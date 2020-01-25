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


def get_conv_layer_out_shape(input_shape, kernels, pools):
    out_shape = input_shape

    for k, p in zip(kernels, pools):
        out_shape = conv2d_out_shape(out_shape, *k)
        out_shape = maxpool2d_out_shape(out_shape, *p[:-1])

    return out_shape


class CNN(nn.Module):
    def __init__(self, input_shape, kernels, pools, 
                 non_linearity='relu', dropout=0.0):
        super().__init__()
        if pools is None:
            pools = len(kernels) * [None]

        elif len(pools) != len(kernels):
            raise ValueError(
                'Number of pooling  and convolution layers do not match')

        in_chann, h, w = input_shape

        conv_layers = []
        for conv_shape, pooling_params in zip(kernels, pools):
            conv_layers.extend([
                nn.Conv2d(in_chann, *conv_shape),
                get_nonlinearity(non_linearity)(),
            ])
            if pooling_params is not None:
                conv_layers.append(create_pool(*pooling_params))

            in_chann = conv_shape[0]

        self.conv_layers = nn.Sequential(*conv_layers)
        self.input_shape = input_shape
        self.out_shape = np.prod(get_conv_layer_out_shape(
            input_shape, kernels, pools))

    def forward(self, inputs):
        inputs = inputs.view(-1, *self.input_shape)
        outputs = self.conv_layers(inputs)
        return torch.flatten(outputs, start_dim=1)


def _bilinear_usample(in_shape, kernel, pool):
    upsample_shape = conv2d_out_shape(in_shape, *kernel)
    usample = nn.UpsamplingBilinear2d(size=upsample_shape[1:])
    in_shape = maxpool2d_out_shape(
        upsample_shape, *pool[:-1])
    return usample, in_shape


class tCNN(nn.Module):
    def __init__(self, input_shape, kernels, pools, 
                 non_linearity='relu', dropout=0.0):
        super().__init__()
        in_chann, h, w = input_shape
        if pools is None:
            pools = len(kernels) * [None]

        conv_layers, layer_input = [], input_shape
        for kernel, pool in zip(kernels, pools):
            out_channel = kernel[0]

            conv = nn.ConvTranspose2d(out_channel, in_chann, *kernel[1:])
            conv_layers.extend([conv, get_nonlinearity(non_linearity)()])

            if pool is not None:
                upsample, layer_input = _bilinear_usample(layer_input,
                                                          kernel, pool)
                conv_layers.append(upsample)
                                                        
            in_chann = out_channel

        self.tconv_layers = nn.Sequential(*reversed(conv_layers))
        self.input_shape = get_conv_layer_out_shape(input_shape, kernels, pools)
        self.out_shape = input_shape

    def forward(self, inputs):
        inputs = inputs.view(-1, *self.input_shape)
        outputs = self.tconv_layers(inputs)
        return torch.flatten(outputs, start_dim=1)
