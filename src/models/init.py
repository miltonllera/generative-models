from .vae import VAE


def init_model(model_type, **kwargs):
    if model_type == 'VAE':
        model = VAE(**kwargs)
    else:
        raise ValueError('Unrecognized model {}'.format(model_type))
    return model