import torch
import torch.nn as nn
from .stochastic import DiagonalGaussian
from .quantization import Quantization
from .feedforward import xavier_normal_init_, kaiming_normal_init_


class AutoEncoder(nn.Module):
    def __init__(self, latent, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.latent = latent
        self.decoder = decoder

        self.reset_parameter()

    @property
    def nlayers(self):
        return len(self.encoder)

    @property
    def latent_size(self):
        return self.latent.size

    def encode(self, inputs):
        """
        Return the posterior distribution given the inputs
        """
        h = self.encoder(inputs)
        return self.latent(h)[0]

    def decode(self, z):
        return self.decoder(z)

    def embed(self, inputs):
        """Embed a batch of data points, x, into their z representations."""
        h = self.encoder(inputs)
        return self.latent(h)[1]

    def forward(self, inputs):
        """
        Takes a batch of samples, encodes them, and then decodes them again.
        Returns the parameters of the posterior to enable ELBO computation.
        """
        h = self.encoder(inputs)
        z_params, z = self.latent(h) # z_params = (mu, logvar)
        return z_params, self.decoder(z)


class VAE(AutoEncoder):
    def __init__(self, latent_size, encoder, decoder):
        latent = DiagonalGaussian(latent_size)
        super().__init__(latent, encoder, decoder)

        self.reset_parameter()

    def reset_parameter(self):
        self.encoder.apply(kaiming_normal_init_)
        self.latent.apply(kaiming_normal_init_)
        self.decoder.apply(kaiming_normal_init_)

    def sample(self, inputs=None, n_samples=1):
        """
        Sample from the prior distribution or the conditional posterior
        learned by the model. If no input is given the output will have
        size (n_samples, latent_size) and if mu and logvar are given it
        will have size (batch_size, n_samples, latent size)
        """
        zsamples = self.sample_latent(inputs, n_samples)
        return self.decode(zsamples)

    def sample_latent(self, inputs=None, n_samples=1):
        if inputs is None:
            h = encoder[-1].bias.new_zeros((2, 1, self.latent_size))
        else:
            h = self.encoder(inputs)
        return self.latent.sample(h, n_samples)


class VQAE(AutoEncoder):
    def __init__(self, latent_size, encoder, decoder,
                 beta=0.25, batch_norm=False, raw_output=True):
        latent = Quantization(*latent_size, beta=beta)
        # encoder_sizes.append(code_size)
        super().__init__(latent, encoder, decoder)

    def reset_parameter(self):
        self.encoder.apply(kaiming_normal_init_)
        self.latent.reset_parameters()
        self.decoder.apply(kaiming_normal_init_)

    def forward(self, input):
        h = self.encoder(input)
        z = self.latent(h)
        return z, self.decoder(z)

    def embed(self, input):
        h = self.encoder(input)
        return self.latent(h)

    sample = forward
    sample_latent = embed