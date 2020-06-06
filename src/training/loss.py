import numpy as np
import torch
import torch.nn as nn
import ignite.metrics as M
from torch.nn.functional import binary_cross_entropy_with_logits as logits_bce
from torch.nn.functional import mse_loss
from torch.nn.modules.loss import _Loss


class ConstrainedELBO(_Loss):
    def __init__(self, reconstruction_loss='bce', gamma=1000.0, capacity=0.0):
        super().__init__(reduction='batchmean')
        if reconstruction_loss == 'bce':
            recons_loss = logits_bce
        elif reconstruction_loss == 'mse':
            recons_loss = mse_loss
        elif not callable(reconstruction_loss):
            raise ValueError('Unrecognized reconstruction' \
                             'loss {}'.format(reconstruction_loss))

        self.recons_loss = recons_loss
        self.gamma = gamma
        self.capacity= capacity

    def forward(self, input, target):
        (mu, logvar), reconstruction = input
        target = target.flatten(start_dim=1)

        recons_loss = self.recons_loss(reconstruction, target, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recons_loss + self.gamma * (KLD - self.capacity).abs())/target.size(0)


class VQELBO(ConstrainedELBO):
    def forward(self, input, target):
        params, reconstruction = input
        return self.recons_loss(reconstruction, target, reduction='sum')


class ELBO(ConstrainedELBO):
    def __init__(self, reconstruction_loss='bce'):
        super().__init__(reconstruction_loss, 1.0, 0.0)


class BetaELBO(ConstrainedELBO):
    def __init__(self, reconstruction_loss='bce', beta=4.0):
        super().__init__(reconstruction_loss, beta, 0.0)


class CapacityScheduler:
    def __init__(self, elbo, capacity_range, patience):
        self.schedule = iter(np.linspace(*capacity_range))
        self.elbo = elbo
        self.patience = patience

    def __call__(self, epoch):
        if (epoch % self.patience) == 0:
            try:
                self.elbo.capacity = next(self.schedule)
            except StopIteration:
                pass


class BetaScheduler:
    def __init__(self, elbo, beta_range, patience):
        self.schedule = iter(np.linspace(*beta_range))
        self.elbo = elbo
        self.patience = patience

    def __call__(self, epoch):
        if (epoch % self.patience) == 0:
            try:
                self.elbo.gamma = next(self.schedule)
            except StopIteration:
                pass


def get_loss(loss):
    loss_fn, params = loss
    if loss_fn == 'elbo':
        return ELBO(**params)
    elif loss_fn == 'beta-elbo':
        return BetaELBO(**params)
    elif loss_fn == 'constrained-elbo':
        return ConstrainedELBO(**params)
    elif loss_fn == 'reconstruction':
        return ConstrainedELBO(**params, gamma=0.0)
    elif loss_fn == 'bxent':
        return nn.BCEWithLogitsLoss(**params)
    elif loss_fn == 'xent':
        return nn.CrossEntropyLoss(**params)
    elif loss_fn == 'mse':
        return nn.MSELoss(**params)
    else:
        raise ValueError('Unknown loss function {}'.format(loss_fn))


def get_metric(metric):
    metric, params =  metric
    if metric == 'mse':
        return M.MeanSquaredError(**params)
    elif metric == 'elbo':
        return M.Loss(BetaELBO(**params))
    elif metric == 'reconstruction':
        return M.Loss(ConstrainedELBO(**params, gamma=0.0))
    elif metric == 'bxent':
        return M.Loss(nn.BCEWithLogitsLoss(**params))
    elif metric == 'xent':
        return M.Loss(nn.CrossEntropyLoss(**params))
    elif metric == 'acc':
        return M.Accuracy(**params)
    raise ValueError('Unrecognized metric {}.'.format(metric))


def init_metrics(training_loss, metrics, rate_reg=0.0):
    criterion = get_loss(training_loss)

    # if rnn_eval:
    #     criterion = RNNLossWrapper(criterion)

    if rate_reg > 0:
        criterion = ComposedLoss(
            terms=[criterion, RateDacay(device=device)],
            decays=[1.0, rate_reg]
        )

    metrics = {m[0]: get_metric(m) for m in metrics}

    return criterion, metrics