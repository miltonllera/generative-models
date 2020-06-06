from .vae import VAE, VQAE


def init_vae(autoencoder, input_size, encoder_sizes, latent_size,
             latent_type, batch_norm=False, raw_output=True):
    if autoencoder == 'variational':
        model = VAE(input_size, encoder_sizes, latent_size,
                    batch_norm, raw_output)
    elif vae_type == 'quantized':
        model = VQAE(input_size, encoder_sizes, latent_size,
                     batch_norm, raw_output)
    return model
