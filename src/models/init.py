from .vae import VAE


def init_vae(vae_type, input_size, encoder_sizes, latent_size,
             latent_type='diagonal', batch_norm=False, raw_output=True):
    if vae_type == 'vae':
        model = VAE(input_size, encoder_sizes, latent_size,
                    batch_norm, latent_type, raw_output)
    return model
