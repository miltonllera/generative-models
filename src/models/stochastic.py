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


class GaussianLayer(StochasticLayer):
    def reparam(self, mu, logvar, random_eval=False):
        if self.training or random_eval:
            # std = exp(log(var))^0.5
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            # z = mu + std * eps
            return mu.addcmul(std, eps)
        return mu

    def sample(self, inputs=None, n_samples=1):
        if inputs is None:
            mu, logvar = self.mu_z.weight.new_zeros((2, 1, self.size)).unbind(0)
        else:
            mu, logvar = self.mu_z(inputs), self.logvar_z(inputs)

        mu = mu.unsqueeze_(1).expand(-1, n_samples, -1)
        logvar = logvar.unsqueeze_(1).expand(-1, n_samples, -1)

        return self.reparam(mu, logvar, random_eval=True)

    def forward(self, inputs):
        mu, logvar = self.mu_z(inputs), self.logvar_z(inputs)
        return (mu, logvar), self.reparam(mu, logvar)


class DiagonalGaussian(GaussianLayer):
    def __init__(self, input_size, latent_size):
        super().__init__()

        self.mu_z = nn.Linear(input_size, latent_size)
        self.logvar_z = nn.Linear(input_size, latent_size)
        self.size = latent_size
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.mu_z.weight)
        nn.init.kaiming_normal_(self.logvar_z.weight)


class HomoscedasticGaussian(GaussianLayer):
    def __init__(self, input_size, latent_size):
        super().__init__()

        self.mu_z = nn.Linear(input_size, latent_size)
        self._logvar_z = nn.Parameter(torch.zeros_like(self.mu_z.bias))
        self.size = latent_size

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.mu_z.weight)
        nn.init.normal_(self._logvar_z)

    def logvar_z(self, inputs):
        return self._logvar_z.unsqueeze(0).expand(len(inputs), -1)
