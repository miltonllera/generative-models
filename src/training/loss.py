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
        latent_params, reconstruction = input
        target = target.flatten(start_dim=1)

        recons_loss = self.recons_loss(reconstruction, target, reduction='sum')
        kl_div = self.latent_term(*latent_params)

        # print(kl_div.cpu().item() / target.size(0))
        # print(recons_loss.cpu().item() / target.size(0))

        return (recons_loss + kl_div) / target.size(0)

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
        params, reconstruction = input
        return self.loss(reconstruction, target, reduction='sum') / target.size(0)


class GaussianKLDivergence(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, input, targets):
        (mu, logvar), _ = input
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return  kl / targets.size(0)


class GaussianVAELoss(VAELoss):
    def __init__(self, reconstruction_loss='bce', beta=1.0, beta_schedule=None):
        super().__init__(reconstruction_loss)
        self.beta = beta
        self.beta_schedule = beta_schedule

    def latent_term(self, mu, logvar):
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return self.beta * kl_div

    def update_parameters(self, step):
        if self.beta_schedule is None:
            return self.beta

        beta_range = self.beta_schedule[:2]
        steps, schedule_type = self.beta_schedule[2:]

        if schedule_type == 'anneal':
            bmax, bmin = beta_range
            delta = (bmax - bmin) / steps
            self.beta = max(bmax - delta * step, bmin)
        elif schedule_type == 'increase':
            bmin, bmax = beta_range
            delta = (bmax - bmin) / steps
            self.beta = min(bmin + delta * step, bmax)


class BurgessGVAELoss(VAELoss):
    def __init__(self, reconstruction_loss='bce', gamma=100.0,
                 capacity=0.0, capacity_schedule=None):
        super().__init__(reconstruction_loss)

        self.gamma = gamma
        self.capacity = capacity
        self.capacity_schedule = capacity_schedule

    def latent_term(self, mu, logvar):
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return self.gamma * (kl_div - self.capacity).abs()

    def update_parameters(self, step):
        if self.capacity_schedule is None:
            return self.capacity

        cmax, cmin, increase_steps = self.capacity_schedule
        delta = (cmax - cmin) / increase_steps

        self.capacity = cmin + delta * step


def get_loss(loss):
    loss_fn, params = loss['name'], loss['params']
    if loss_fn == 'vae':
        return GaussianVAELoss(**params, beta=1.0)
    elif loss_fn == 'beta-vae':
        return GaussianVAELoss(**params)
    elif loss_fn == 'constrained-beta-vae':
        return BurgessGVAELoss(**params)
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


def init_metrics(loss, metrics, rate_reg=0.0, rnn_eval=False, loss_fn_parameters=None):
    criterion = get_loss(loss)

    # if rnn_eval:
    #     criterion = RNNLossWrapper(criterion)

    if rate_reg > 0:
        criterion = ComposedLoss(
            terms=[criterion, RateDacay(device=device)],
            decays=[1.0, rate_reg]
        )

    metrics = {m['name']: get_metric(m) for m in metrics}

    return criterion, metrics