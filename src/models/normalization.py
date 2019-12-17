import torch
import torch.nn as nn


class DivNorm(nn.Module):
    def __init__(self, norm_bias=0.0, gain=1.0, output_shift=0.0):
        super().__init__()
        self.norm_bias = 0.0
        self.shift = output_shift
        self.gain = gain

    def forward(self, inputs):
        inputs -= inputs.mean(axis=1).reshape(-1, 1)
        norm = torch.sqrt(self.norm_bias + (inputs ** 2).sum(axis=1))
        return self.gain * inputs / norm.reshape(-1, 1) + self.shift
        
        