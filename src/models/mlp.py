import torch
import torch.nn as nn


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


def _init(m, init):
    if type(m) == nn.Linear:
        init(m.weight)
        try:
            m.bias.data.zero_()
        except AttributeError:
            pass

xavier_normal_init_ = lambda m: _init(m, nn.init.xavier_normal_)
kaiming_normal_init_ = lambda m: _init(m, nn.init.kaiming_normal_)


def create_mlp(input_size, hidden_sizes, output_size=None, nonlinearity='relu',
               dropout=0.0, batch_norm=False, init=None):
    sizes = [input_size] + hidden_sizes

    if isinstance(nonlinearity, str):
        nonlinearity = get_nonlinearity(nonlinearity)

    layers = []
    for i in range(1, len(sizes)):
        layers.append(nn.Linear(sizes[i-1], sizes[i]), bias=not batch_norm)

        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[i]))

        layers.append(nonlinearity())

        if not batch_norm and dropout > 0:
            layers.append(nn.Dropout(dropout))

    if output_size:
        layers.append(nn.Linear(sizes[-1], output_size))

    mlp = nn.Sequential(*layers)

    if init is not None:
        mlp.apply(init)

    return mlp
