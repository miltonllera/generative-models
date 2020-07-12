import numpy as np
import torch
import torch.nn as nn
import ignite.metrics as M
from torch.nn.functional import binary_cross_entropy_with_logits as logits_bce
from torch.nn.functional import mse_loss
from torch.nn.modules.loss import _Loss


class VAELoss(_Loss):
    def __init__(self, reconstruction_loss='bce'):
        super().__init__(reduction='batchmean')
        if reconstruction_loss == 'bce':
            recons_loss = logits_bce
        elif reconstruction_loss == 'mse':
            recons_loss = mse_loss
        elif not callable(reconstruction_loss):
            raise ValueError('Unrecognized reconstruction' \
                            'loss {}'.format(reconstruction_loss))

        self.recons_loss = recons_loss

    def forward(self, input, target):
        reconstruction, z, latent_params = input
        target = target.flatten(start_dim=1)

        recons_loss = self.recons_loss(reconstruction, target, reduction='sum')
        recons_loss /= target.size(0)

        kl_div = self.latent_term(z, *latent_params)

        return recons_loss + kl_div

    def latent_term(self):
        raise NotImplementedError()


class ReconstructionNLL(_Loss):
    def __init__(self, loss='bce'):
        super().__init__(reduction='batchmean')
        if loss == 'bce':
            recons_loss = logits_bce
        elif loss == 'mse':
            recons_loss = mse_loss
        elif not callable(reconstruction_loss):
            raise ValueError('Unrecognized reconstruction' \
                             'loss {}'.format(reconstruction_loss))
        self.loss = recons_loss

    def forward(self, input, target):
        if isinstance(input, (tuple, list)):
            reconstruction = input[0]
        else:
            reconstruction = input

        return self.loss(reconstruction, target, reduction='sum') / target.size(0)


class GaussianKLDivergence(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, input, targets):
        _, _, (mu, logvar) = input
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return  kl / targets.size(0)


class GaussianVAELoss(VAELoss):
    def __init__(self, reconstruction_loss='bce', beta=1.0, beta_schedule=None):
        super().__init__(reconstruction_loss)
        self.beta = beta
        self.beta_schedule = beta_schedule
        self.anneal = 1.0

    def latent_term(self, z_sample, mu, logvar):
        kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
        kl_div /= z_sample.size(0)
        return self.anneal * self.beta * kl_div.sum()

    def update_parameters(self, step):
        if self.beta_schedule is not None:
            steps, schedule_type = self.beta_schedule
            delta = 1 / steps

            if schedule_type == 'anneal':
                self.anneal = max(1.0 - step * delta, 0)
            elif schedule_type == 'increase':
                self.anneal = min(delta * step, 1.0)


class BurgessLoss(VAELoss):
    def __init__(self, reconstruction_loss='bce', gamma=100.0,
                 capacity=0.0, capacity_schedule=None):
        super().__init__(reconstruction_loss)

        self.gamma = gamma
        self.capacity = capacity
        self.capacity_schedule = capacity_schedule

    def latent_term(self, z_sample, mu, logvar):
        kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
        kl_div /= z_sample.size(0)
        return self.gamma * (kl_div- self.capacity).abs()

    def update_parameters(self, step):
        if self.capacity_schedule is not None:
            cmin, cmax, increase_steps = self.capacity_schedule
            delta = (cmax - cmin) / increase_steps

        self.capacity = min(cmin + delta * step, cmax)


class QuantizationLoss(AELoss):
    def __init__(self, reconstruction_loss='mse', beta=0.25):
        super().__init__(reconstruction_loss)
        self.beta = beta

    def latent_term(self, quantization, diff):
        return self.beta * diff.pow(2).sum() / quantization.size(0)


def get_loss(loss):
    loss_fn, params = loss['name'], loss['params']
    if loss_fn == 'vae':
        return GaussianVAELoss(**params, beta=1.0)
    elif loss_fn == 'beta-vae':
        return GaussianVAELoss(**params)
    elif loss_fn == 'constrained-beta-vae':
        return BurgessLoss(**params)
    elif loss_fn == 'recons_nll':
        return ReconstructionNLL(**params)
    elif loss_fn == 'bxent':
        return nn.BCEWithLogitsLoss(**params)
    elif loss_fn == 'xent':
        return nn.CrossEntropyLoss(**params)
    else:
        raise ValueError('Unknown loss function {}'.format(loss_fn))


def get_metric(metric):
    name = metric['name']
    params =  metric['params']
    if name == 'mse':
        return M.MeanSquaredError(**params)
    elif name == 'vae':
        return M.Loss(GaussianVAELoss(**params))
    elif name == 'kl-div':
        return M.Loss(GaussianKLDivergence())
    elif name == 'recons_nll':
        return M.Loss(ReconstructionNLL(**params))
    elif name == 'bxent':
        return M.Loss(nn.BCEWithLogitsLoss(**params))
    elif name == 'xent':
        return M.Loss(nn.CrossEntropyLoss(**params))
    elif name == 'acc':
        return M.Accuracy(**params)
    raise ValueError('Unrecognized metric {}.'.format(metric))


def init_metrics(loss, metrics, rate_reg=0.0):
    criterion = get_loss(loss)

    if rate_reg > 0:
        criterion = ComposedLoss(
            terms=[criterion, RateDacay(device=device)],
            decays=[1.0, rate_reg]
        )

    metrics = {m['name']: get_metric(m) for m in metrics}

    return criterion, metrics