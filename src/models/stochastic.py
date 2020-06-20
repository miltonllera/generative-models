import torch
import torch.nn as nn


class StochasticLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def reparam(self, *params):
        raise NotImplementedError()

    def sample(self, inputs=None, nsamples=1):
        raise  NotImplementedError()

    def forward(self, inputs):
        raise NotImplementedError()


class DiagonalGaussian(StochasticLayer):
    def __init__(self, latent_size):
        super().__init__()
        self.size = latent_size

    def reparam(self, mu, logvar, random_eval=False):
        if self.training or random_eval:
            # std = exp(log(var))^0.5
            std = logvar.mul(0.5).exp()
            eps = torch.randn_like(std)
            # z = mu + std * eps
            return mu.addcmul(std, eps)
        return mu

    def sample(self, inputs=None, n_samples=1, device='cpu'):
        mu, logvar = inputs.chunk(2, dim=1)
        mu = mu.unsqueeze_(1).expand(-1, n_samples, -1)
        logvar = logvar.unsqueeze_(1).expand(-1, n_samples, -1)

        return self.reparam(mu, logvar, random_eval=True)

    def forward(self, inputs):
        mu, logvar = inputs.chunk(2, dim=1)
        return (mu, logvar), self.reparam(mu, logvar)

    def extra_repr(self):
        return 'size={}'.format(self.size)