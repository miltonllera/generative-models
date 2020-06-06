from .vae import VAE, VQAE


def init_vae(autoencoder, input_size, encoder_sizes, latent_size,
             latent_type=None, batch_norm=False, raw_output=True):
    if autoencoder == 'variational':
        model = VAE(input_size, encoder_sizes, latent_size,
                    batch_norm, raw_output)
    elif autoencoder == 'quantized':
        model = VQAE(input_size, encoder_sizes, latent_size,
                     batch_norm, raw_output)
    return model
