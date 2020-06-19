import math
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
from .mlp import get_nonlinearity


KernelParams = namedtuple('KernelParams',
    ['outchannels', 'shape', 'stride', 'padding'],
    defaults=[1, (3, 3), (1, 1), (0, 0)]
)

PoolingParams = namedtuple('PoolingParams',
    ['shape', 'stride', 'padding', 'mode'],
    defaults=[(3, 3), (1, 1), (0,0), 'max']
)


def create_pool(kernel_size, stride, padding, mode):
    if mode == 'avg':
        return nn.AvgPool2d(kernel_size, stride, padding)
    elif mode == 'max':
        return nn.MaxPool2d(kernel_size, stride)
    elif mode == 'adapt':
        return nn.AdaptiveAvgPool2d(kernel_size)
    else:
        raise ValueError('Unrecognised pooling mode {}'.format(mode))


def _pair(s):
    if not isinstance(s, tuple):
        return s, s
    return s


def conv2d_out_shape(in_shape, out_channels, kernel_shape, stride, padding):
    in_shape = in_shape[1:]
    kernel_shape = _pair(kernel_shape)
    stride = _pair(stride)
    padding = _pair(padding)

    hval, wval = zip(in_shape, kernel_shape, stride, padding)

    hout = math.floor((hval[0] -  hval[1] + 2 * hval[3]) / hval[2]) + 1
    wout = math.floor((wval[0] -  wval[1] + 2 * wval[3]) / wval[2]) + 1

    return out_channels, hout, wout


def maxpool2d_out_shape(in_shape, pool_shape, stride, padding):
    in_channels, hout, wout = in_shape
    pool_shape = _pair(pool_shape)
    stride = _pair(stride)
    padding = _pair(padding)

    hval, wval = zip(pool_shape, stride, padding)

    hout = math.floor((hout - hval[0] + 2 * hval[2]) / hval[1]) + 1
    wout = math.floor((wout - wval[0] + 2 * wval[2]) / wval[1]) + 1

    return in_channels, hout, wout


def get_conv_layer_out_shape(input_size, kernels, pools):
    out_shape = input_size

    for k, p in zip(kernels, pools):
        out_shape = conv2d_out_shape(out_shape, *k)
        if p is not None:
            out_shape = maxpool2d_out_shape(out_shape, *p[:-1])

    return out_shape


class CNN(nn.Module):
    def __init__(self, input_size, layer_params):
        super().__init__()

        if isinstance(layer_params, dict):
            layer_params = dict.items()

        in_chann, h, w = output_size = input_size
        cnn_layers = []

        for layer_type, params in layer_params:
            if layer_type == 'conv':
                layer = nn.Conv2d(in_chann, *params)
                output_size = conv2d_out_shape(output_size, *params)
                in_chann = params[0]
            elif layer_type == 'batch_norm':
                layer = nn.BatchNorm2d(in_chann)
            elif layer_type == 'pool':
                layer = create_pool(*params)
                output_size = maxpool2d_out_shape(output_size, *params[:-1])
            elif layer_type == 'dropout':
                layer = nn.Dropout2d(*params)
            else:
                layer = get_nonlinearity(layer_type)(*params)

            cnn_layers.append(layer)

        self.layers = nn.Sequential(*cnn_layers)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, inputs):
        inputs = inputs.view(-1, *self.input_size)
        outputs = self.layers(inputs)
        return torch.flatten(outputs, start_dim=1)


def _bilinear_usample(in_shape, pool):
    usample = nn.UpsamplingBilinear2d(size=in_shape[1:])
    in_shape = maxpool2d_out_shape(
        in_shape, *pool[:-1])
    return usample, in_shape


class TransposedCNN(nn.Module):
    def __init__(self, input_size, layer_params):
        super().__init__()

        if isinstance(layer_params, dict):
            layer_params = layer_params.items()

        in_chann, h, w = output_size = input_size
        tcnn_layers = []

        for layer_type, params in layer_params:
            if layer_type == 'conv':
                layer = nn.ConvTranspose2d(params[0], in_chann, *params[1:])
                output_size = conv2d_out_shape(output_size, *params)
                in_chann = params[0]
            elif layer_type == 'batch_norm':
                layer = nn.BatchNorm2d(in_chann)
            elif layer_type == 'pool':
                layer, output_size = _bilinear_usample(output_size, params)
            elif layer_type == 'dropout':
                layer = nn.Dropout2d(*params)
            else:
                layer = get_nonlinearity(layer_type)(*params)

            tcnn_layers.append(layer)

        tcnn_layers = list(reversed(tcnn_layers))
        while not isinstance(tcnn_layers[-1], nn.ConvTranspose2d):
            tcnn_layers.pop()

        self.layers = nn.Sequential(*tcnn_layers)
        self.input_size = output_size
        self.output_size = input_size

    def forward(self, inputs):
        inputs = inputs.view(-1, *self.input_size)
        outputs = self.layers(inputs)
        return torch.flatten(outputs, start_dim=1)
