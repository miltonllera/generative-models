import torch
from torch.nn.functional import binary_cross_entropy_with_logits as logits_bce
from torch.nn.functional import mse_loss
from torch.nn.modules.loss import _Loss    


class BetaELBO(_Loss):
    def __init__(self, reconstruction_loss='bce', gamma=10.0, capacity=0.5):
        super().__init__(reduction='batchmean')
        if reconstruction_loss == 'bce':
            recons_loss = logits_bce
        elif reconstruction_loss == 'mse':
            recons_loss = mse_loss
        elif not callable(recons_loss):
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
        return (recons_loss + self.gamma * (KLD - self.capacity))/target.size(0)

    
class ELBO(BetaELBO):
    def __init__(self, reconstruction_loss='bce'):
        super().__init__(reconstruction_loss, 1.0, 0.0)


class CapacityScheduler:
    def __init__(self, elbo, capcity_range, patience):
        self.schedule = iter(range(*capcity_range))
        self.elbo = elbo
        self.patience = patience

    def __call__(self, engine):
        if (engine.state.iteration % self.patience) == 0:
            elbo.capacity = next(self.schedule)

    def attach(self, engine):
        engine.add_event_handler(Events.COMPLETED, self)
        return self
