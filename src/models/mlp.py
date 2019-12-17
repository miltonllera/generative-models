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


def xavier_normal_init_(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.zero_()


def kaiming_normal_init_(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()


def create_mlp(input_size, hidden_sizes, output_size=None, dropout=0.0,
               nonlinearity='relu', init=None):
    sizes = [input_size] + hidden_sizes

    if isinstance(nonlinearity, str):
        nonlinearity = get_nonlinearity(nonlinearity)

    layers = []
    for i in range(1, len(sizes)):
        layers.extend([
            nn.Linear(sizes[i-1], sizes[i]),
            nonlinearity(),
            nn.Dropout(dropout)
        ])

    if output_size:
        layers.append(nn.Linear(sizes[-1], output_size))

    mlp = nn.Sequential(*layers)

    if init is not None:
        mlp.apply(init)

    return mlp
