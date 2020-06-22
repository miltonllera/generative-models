import math
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple


KernelParams = namedtuple('KernelParams',
    ['outchannels', 'shape', 'stride', 'padding'],
    defaults=[1, (3, 3), (1, 1), (0, 0)]
)

PoolingParams = namedtuple('PoolingParams',
    ['shape', 'stride', 'padding', 'mode'],
    defaults=[(3, 3), (1, 1), (0,0), 'max']
)


def get_nonlinearity(nonlinearity):
    if nonlinearity == 'relu':
        return nn.ReLU
    elif nonlinearity == 'sigmoid':
        return nn.Sigmoid
    elif nonlinearity == 'tanh':
        return nn.Tanh
    elif nonlinearity == 'lrelu':
        return nn.LeakyReLU
    elif nonlinearity == 'elu':
        return nn.ELU
    raise ValueError('Unrecognized non linearity: {}'.format(nonlinearity))


def xavier_normal_init_(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        try:
            m.bias.data.zero_()
        except AttributeError:
            pass


def kaiming_normal_init_(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        try:
            m.bias.data.zero_()
        except AttributeError:
            pass

def create_linear(input_size, params, transposed=False):
    if isinstance(input_size, (list, tuple)):
        in_features = input_size[-1]
    else:
        in_features = input_size

    if transposed:
        layer = nn.Linear(params[0], in_features, *params[1:])
    else:
        layer = nn.Linear(in_features, *params)

    if isinstance(input_size, (list, tuple)):
        input_size[-1] = params[0]
    else:
        input_size = params[0]

    return layer, input_size


def creat_batch_norm(ndims, input_size, params):
    if ndims == 1:
        return nn.BatchNorm1d(input_size, *params)
    elif ndims == 2:
        return nn.BatchNorm2d(input_size[0], *params)
    elif ndims == 3:
        return nn.BatchNorm3d(input_size[0], *params)


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


def compute_flattened_size(input_size, start_dim=1, end_dim=-1):
    start_dim -= 1
    if end_dim < 0:
        end_dim = len(input_size) + 1

    if start_dim < 0:
        raise ValueError('Cannot flatten batch dimension')

    output_size = list(input_size[:start_dim])
    output_size.append(np.prod(input_size[start_dim:end_dim]))
    output_size.extend(input_size[end_dim:])

    if len(output_size) == 1:
        return output_size[0]

    return output_size

class Flatten(nn.Flatten):
    def extra_repr(self):
        dims = [str(self.start_dim), str(self.end_dim)]
        return 'start_dim={}, end_dim={}'.format(*dims)


class Unflatten(nn.Module):
    def __init__(self, unflatten_shape):
        super().__init__()
        self.unflatten_shape = unflatten_shape

    def forward(self, inputs):
        return inputs.view(-1, *self.unflatten_shape)

    def extra_repr(self):
        dims = [str(d) for d in self.unflatten_shape]
        return 'batch_size, {}'.format(', '.join(dims))


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.transposed_dims = dim1, dim2

    def forward(self, inputs):
        return inputs.transpose(*self.transposed_dims)

    def extra_repr(self):
        return 'dim1={}, dim2={}'.format(*self.transposed_dims)


def transpose_size(size, dim1, dim2):
    size = list(size)
    size[dim1-1], size[dim2-1] = size[dim2-1], size[dim1-1]
    return size


class FeedForward(nn.Sequential):
    def __init__(self, input_size, layer_params, flatten=True):
        if isinstance(layer_params, dict):
            layer_params = dict.items()

        if isinstance(input_size, (tuple, list)):
            in_chann, _, _ = output_size = input_size
        else:
            in_chann = 1
            output_size = input_size

        cnn_layers = []

        for layer_type, params in layer_params:
            if layer_type == 'linear':
                layer, output_size = create_linear(output_size, params)
            elif layer_type == 'conv':
                layer = nn.Conv2d(in_chann, *params)
                output_size = conv2d_out_shape(output_size, *params)
                in_chann = params[0]
            elif layer_type == 'batch_norm':
                layer = creat_batch_norm(params[0], output_size, params[1:])
            elif layer_type == 'pool':
                layer = create_pool(*params)
                output_size = maxpool2d_out_shape(output_size, *params[:-1])
            elif layer_type == 'dropout':
                layer = nn.Dropout2d(*params)
            elif layer_type == 'flatten':
                layer = Flatten(*params)
                output_size = compute_flattened_size(output_size, *params)
            elif layer_type == 'transpose':
                layer = Transpose(*params)
                output_size = transpose_size(output_size, *params)
            else:
                layer = get_nonlinearity(layer_type)(*params)

            cnn_layers.append(layer)

        super().__init__(*cnn_layers)

        self.input_size = input_size
        self.output_size = output_size
        self.flatten = flatten

    def forward(self, inputs):
        if isinstance(self.input_size, (list, tuple)):
            inputs = inputs.view(-1, *self.input_size)

        outputs = super().forward(inputs)

        if self.flatten:
            outputs = torch.flatten(outputs, start_dim=1)

        return outputs


def _bilinear_usample(in_shape, pool):
    usample = nn.UpsamplingBilinear2d(size=in_shape[1:])
    in_shape = maxpool2d_out_shape(
        in_shape, *pool[:-1])
    return usample, in_shape


class TransposedFF(nn.Sequential):
    def __init__(self, layer_params, input_size, flatten=True):
        if isinstance(layer_params, dict):
            layer_params = layer_params.items()

        layer_params = list(reversed(layer_params))

        if isinstance(input_size, (tuple, list)):
            in_chann, _, _ = output_size = input_size
        else:
            in_chann = 1
            output_size = input_size

        tcnn_layers = []

        for layer_type, params in layer_params:
            if layer_type == 'linear':
                layer, output_size = create_linear(output_size, params,
                                                   transposed=True)
            elif layer_type == 'conv':
                layer = nn.ConvTranspose2d(params[0], in_chann, *params[1:])
                output_size = conv2d_out_shape(output_size, *params)
                in_chann = params[0]
            elif layer_type == 'batch_norm':
                layer = creat_batch_norm(params[0], output_size, params[1:])
            elif layer_type in ['pool', 'upsample']:
                layer, output_size = _bilinear_usample(output_size, params)
            elif layer_type == 'dropout':
                layer = nn.Dropout2d(*params)
            elif layer_type in ['flatten', 'unflatten']:
                layer = Unflatten(output_size)
                output_size = compute_flattened_size(output_size, *params)
            elif layer_type == 'transpose':
                params = list(reversed(params))
                layer = Transpose(*params)
                output_size = transpose_size(output_size, *params)
            else:
                layer = get_nonlinearity(layer_type)(*params)

            tcnn_layers.append(layer)

        tcnn_layers = list(reversed(tcnn_layers))
        while not isinstance(tcnn_layers[-1], (nn.ConvTranspose2d, nn.Linear)):
            tcnn_layers.pop()

        super().__init__(*tcnn_layers)

        self.input_size = output_size
        self.output_size = input_size
        self.flatten = flatten

    def forward(self, inputs):
        # inputs = inputs.view(-1, *self.input_size)
        outputs = super().forward(inputs)

        if self.flatten:
            outputs = torch.flatten(outputs, start_dim=1)

        return outputs
