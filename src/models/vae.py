import torch
import torch.nn as nn
from .stochastic import DiagonalGaussian
from .quantization import Quantization
from .mlp import create_mlp, xavier_normal_init_, kaiming_normal_init_


def get_latent(latent_type):
    if latent_type == 'diag-gauss':
        return DiagonalGaussian
    elif latent_type == 'homoscedastic':
        return HomoscedasticGaussian
    raise ValueError('Unrecognized latent layer {}'.format(latent_type))


class AutoEncoder(nn.Module):
    def __init__(self, input_size, encoder_sizes, latent_size,
                 latent_type, batch_norm=False, raw_output=True):

        super().__init__()

        if isinstance(encoder_sizes, int) and encoder_sizes:
            encoder_sizes = [encoder_sizes]
        elif not encoder_sizes:
            encoder_sizes = []

        # Encoder MLP
        self.encoder = create_mlp(input_size, encoder_sizes,
                                  batch_norm=batch_norm,
                                  init=kaiming_normal_init_)

        # Latent representation
        latent_constructor = get_latent(latent_type)
        self.latent = latent_constructor(
            encoder_sizes[-1] if len(encoder_sizes) else input_size,
            latent_size)

        if raw_output:
            decoder_sizes = list(reversed(encoder_sizes))
            output_size = input_size
        else:
            decoder_sizes = list(reversed(encoder_sizes)) + [input_size]
            output_size = None

        # Decoder MLP
        self.decoder = create_mlp(latent_size, decoder_sizes, output_size,
                                  batch_norm=batch_norm,
                                  init=kaiming_normal_init_)

        self.input_size = input_size

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
        params, z = self.latent(h)
        return params, self.decoder(z)


class VAE(nn.Module):
    def __init__(self, input_size, encoder_sizes, latent_size,
                 batch_norm=False, raw_output=True):

        super().__init__(input_size, encoder_sizes, latent_size,
                         batch_norm, 'diag-gauss', raw_output)

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
        h = None if inputs is None else self.encoder(inputs)
        return self.latent.sample(h, n_samples)


class VQAE(AutoEncoder):
    def __init__(self, input_size, encoder_sizes, latent_size,
                 batch_norm=False, raw_output=True):

        super().__init__(input_size, encoder_sizes, latent_size,
                         batch_norm, 'quantized', raw_output)

    def reset_parameter(self):
        self.encoder.apply(kaiming_normal_init_)
        self.latent.reset_parameters()
        self.decoder.apply(kaiming_normal_init_)

    def forward(self, input):
        h = self.encoder(input)
        z = self.latent(h)
        return self.decoder(z)

    def embed(self, input):
        h = self.encoder(input)
        return self.latent(h)

    sample = forward
    sample_latent = embed