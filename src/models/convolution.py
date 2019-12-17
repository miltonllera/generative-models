import math
from collections import namedtuple

import torch
import torch.nn as nn


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


def _get_out_features(*args):
    # expand params
    params = [p if isinstance(p, tuple) else (p,p) for p in args]

    hval, wval = zip(*params)

    # new output shape
    hout = math.floor((hval[0] -  hval[1] + 2 * hval[3]) / hval[2]) + 1
    hout = math.floor((hout - hval[4] + 2 * hval[6]) / hval[5]) + 1

    wout = math.floor((wval[0] -  wval[1] + 2 * wval[3]) / wval[2]) + 1
    wout = math.floor((wout - wval[4] + 2 * wval[6]) / wval[5]) + 1

    return hout, wout


def get_conv2D_out_features(input_shape, kernels, pools):
    out_chann, hin, win = input_shape
    out_shape = hin, win

    for k, p in zip(kernels, pools):
        kout, ksize, kstride, kpad = k
        psize, pstride, ppad, _ = p

        out_chann = kout
        out_shape = _get_out_features(
            out_shape,ksize, kstride,
            kpad, psize, pstride, ppad
        )

    return out_chann, out_shape[0], out_shape[1]


def unflatten(batch, shape):
    channels, w, h = shape
    return batch.view(-1, channels, w, h)


def flatten(batch, size):
    return batch.view(-1, size)


def create_conv_layers(input_shape, kernels, pools, dropout=0.0, non_linearity='ReLU'):
    in_chann, h, w = input_shape

    conv_layers = []
    for conv_shape, pool_shape in zip(kernels, pools):
        conv_layers.extend([
            nn.Conv2d(in_chann, *conv_shape),
            nn.ReLU(),
            nn.Dropout(dropout),
            create_pool(*pool_shape)
        ])

        in_chann = conv_shape.outchannels

    return nn.Sequential(*conv_layers)


def init_cnn_layers(layer_type):
    if layer_type == '2-layer':
        kernels = [
            KernelParams(
                outchannels=3, shape=(2, 2), stride=(1, 1), padding=(1, 1)),
        ]

        pools = [
            PoolingParams(shape=(2, 2), stride=(1, 1), padding=(0, 0), mode='max'),
        ]

    elif layer_type == '4-layer':
        kernels = [
            KernelParams(
                outchannels=16, shape=(5, 5), stride=(1, 1), padding=(1, 1)),
            KernelParams(
                outchannels=32, shape=(5, 5), stride=(1, 1), padding=(1, 1)),
        ]

        pools = [
            PoolingParams(shape=(2, 2), stride=(1, 1), padding=(0, 0), mode='max'),
            PoolingParams(shape=(2, 2), stride=(1, 1), padding=(0, 0), mode='max')
        ]

    elif layer_type == '6-layer':
        kernels = [
            KernelParams(outchannels=32, shape=(3, 3), stride=(1, 1), padding=(1, 1)),
            KernelParams(outchannels=64, shape=(5, 5), stride=(1, 1), padding=(1, 1)),
            KernelParams(outchannels=128, shape=(5, 5), stride=(1, 1), padding=(1, 1)),
        ]

        pools = [
            PoolingParams(shape=(2, 2), stride=(1, 1), padding=(0, 0), mode='max'),
            PoolingParams(shape=(2, 2), stride=(1, 1), padding=(0, 0), mode='max'),
            PoolingParams(shape=(2, 2), stride=(1, 1), padding=(0, 0), mode='max')
        ]

    elif layer_type == '8-layer':
        kernels = [
            KernelParams(outchannels=32, shape=(3, 3), stride=(1, 1), padding=(1, 1)),
            KernelParams(outchannels=64, shape=(5, 5), stride=(1, 1), padding=(1, 1)),
            KernelParams(outchannels=128, shape=(5, 5), stride=(1, 1), padding=(1, 1)),
            KernelParams(outchannels=256, shape=(5, 5), stride=(1, 1), padding=(1, 1)),
        ]

        pools = [
            PoolingParams(shape=(2, 2), stride=(1, 1), padding=(0, 0), mode='max'),
            PoolingParams(shape=(2, 2), stride=(1, 1), padding=(0, 0), mode='max'),
            PoolingParams(shape=(2, 2), stride=(1, 1), padding=(0, 0), mode='max'),
            PoolingParams(shape=(2, 2), stride=(1, 1), padding=(0, 0), mode='max'),
    ]

    return kernels, pools
