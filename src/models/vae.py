import torch
import torch.nn as nn
from .stochastic import DiagonalGaussian, HomoscedasticGaussian, PosteriorGaussian
from .convolution import CNN, tCNN
from .mlp import create_mlp, xavier_normal_init_, kaiming_normal_init_
from .utils import load_model as load
from .normalization import DivNorm


def get_latent(latent_type):
    if latent_type == 'diagonal':
        return DiagonalGaussian
    elif latent_type == 'homoscedastic':
        return HomoscedasticGaussian
    raise ValueError('Unrecognized latent layer {}'.format(latent_type))


class VAE(nn.Module):
    def __init__(self, input_size, encoder_sizes, latent_size,
                 batch_norm=False, latent_type='diagonal'):

        super(VAE, self).__init__()

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

        # Decoder MLP
        decoder_sizes = list(reversed(encoder_sizes))
        self.decoder = create_mlp(latent_size, decoder_sizes, input_size,
                                  batch_norm=batch_norm,
                                  init=kaiming_normal_init_)

        self.input_size = input_size

    def reset_parameter(self):
        self.encoder.apply(kaiming_normal_init_)
        self.latent.apply(kaiming_normal_init_)
        self.decoder.apply(kaiming_normal_init_)

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


class LadderVAE(nn.Module):
    def __init__(self, input_size, encoder_sizes, latent_sizes,
                 batch_norm=False):
        super().__init__()
        self.input_size = input_size
        self.latent_sizes = latent_sizes

        encoders, decoders, latents = [], [], []

        for i, sizes in enumerate(encoder_sizes):
            enc = create_mlp(input_size, sizes)

            if i + 1 < len(latent_sizes):
                lat = PosteriorGaussian(sizes[-1], latent_sizes[i])
            else:
                lat = DiagonalGaussian(sizes[-1], latent_sizes[i])

            input_size = sizes[-1]

            if i == 0:
                sizes = list(reversed(sizes))
                dec = create_mlp(latent_sizes[i], sizes, input_size,
                                 batch_norm=batch_norm)
            else:
                sizes = list(reversed(sizes)) + [2 * latent_sizes[-1]]
                dec = create_mlp(latent_sizes[i], sizes, batch_norm=batch_norm)

            encoders.append(enc)
            decoders.append(dec)
            latents.append(lat)

        self.encoders = nn.ModuleList(encoders)
        self.latents = nn.ModuleList(latents)
        self.decoders = nn.ModuleList(decoders)

    def forward(self, inputs):
        hidden, params = [], []

        for enc in self.encoders:
            inputs = enc(inputs)
            hidden.append(inputs)

        p, z = self.latents[-1](hidden[-1])
        h_tdown = self.decoders[-1](z)
        params.append(p)

        for i in range(len(self.latents)):
            lat = self.latents[-(2 + i)]
            dec = self.decoders[-(2 + i)]
            h_bup = hidden[-(2 + i)]

            p, z = lat((h_bup, h_tdown.chunk(2, 1)))
            h_tdown = dec(z)
            params.append(p)

        return params, h_tdown


class cVAE(nn.Module):
    def __init__(self, input_size, kernels, pools, encoder_sizes,
                 latent_size, batch_norm=False, latent_type='diagonal'):
        super().__init__()

        if isinstance(encoder_sizes, int) and encoder_sizes:
            encoder_sizes = [encoder_sizes]
        elif not encoder_sizes:
            encoder_sizes = []

        conv = CNN(input_size, kernels, pools,
               batch_norm=batch_norm, non_linearity='relu')

        conv_out_features = conv.out_shape

        # Encoder MLP
        encoder = create_mlp(conv_out_features, encoder_sizes,
                             batch_norm=batch_norm, init=kaiming_normal_init_)

        self.encoder = nn.Sequential(conv, encoder)

        # Latent representation
        latent_constructor = get_latent(latent_type)
        self.latent = latent_constructor(
            encoder_sizes[-1] if len(encoder_sizes) else input_size,
            latent_size)

        # Decoder
        dec_sizes = list(reversed(encoder_sizes)) + [conv_out_features]
        decoder = create_mlp(latent_size, dec_sizes, batch_norm=batch_norm,
                             init=kaiming_normal_init_)

        deconv = tCNN(input_size, kernels, pools,
                      batch_norm=batch_norm, non_linearity='relu')

        self.decoder = nn.Sequential(decoder, deconv)
        self.input_size = input_size

    def reset_parameter(self):
        self.encoder.apply(kaiming_normal_init_)
        self.latent.apply(kaiming_normal_init_)
        self.decoder.apply(kaiming_normal_init_)

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



def init_vae(model_type, input_size, encoder_sizes, latent_size,
             latent_type='diagonal', kernels=None, pools=None,
             device='cpu'):
    if model_type == 'vae':
        model = VAE(input_size, encoder_sizes, latent_size, latent_type)
    if model_type == 'cvae':
        model = ConvVAE(input_size, kernels, pools, encoder_sizes,
                     latent_size, latent_type)
    return model.to(device=device)


def load_vae(params, state, device):
    m = init_vae(device=device, **params)
    m.load_state_dict(state)
    return m


load_from_file = lambda folder, device: load(folder, load_vae, device='cuda')