import sys
import torch
import torch.nn as nn
import numpy as np
from io import BytesIO
from sacred import Ingredient
from incense import ExperimentLoader

if sys.path[0] != '../src':
    sys.path.insert(0, '../src')

from models.feedforward import FeedForward, TransposedFF
from models.autoencoder import VQAE, VAE


model = Ingredient('model')

# Basic init functions, add more later
init_vae = model.capture(VAE)
init_vqae = model.capture(VQAE)


def get_init(autoencoder):
    if autoencoder == 'variational':
        return init_vae
    elif autoencoder == 'quantized':
        return init_vqae


# General model initialization function, init_fn will depend on the experiment
@model.capture
def init_model(init_fn, device='cpu'):
    model = init_fn()
    return model.to(device=device)


# Function for loading a state dictionary from the database
@model.capture
def load_from_db(db, exp_id, mongo_uri='127.0.0.1'):
    loader = ExperimentLoader(mongo_uri=mongo_uri, db_name=db)
    exp = loader.find_by_id(exp_id)

    model_state = BytesIO(exp.artifacts['trained-model'].content)
    model_state = torch.load(model_state)

    return model_state

# Print the model
@model.command(unobserved=True)
def show():
    model = init_model(init_fn=init_cnn_autoencoder)
    print(model)


# Convolutional AutoEncoder
class VAEPredictor(nn.Module):
    def __init__(self, vae, output_size, pretrained=True):
        super().__init__()

        if pretrained:
            for p in vae.parameters():
                p.requires_grad=False

        self.vae = vae
        self.linear = nn.Linear(vae.latent_size, output_size)

    def forward(self, inputs):
        z = self.vae.embed(inputs)
        return self.linear(z)


@model.capture
def init_cnn_autoencoder(autoencoder, input_size, encoder_layers,
                         latent_size, decoder_layers=None):
    if autoencoder == 'variational':
        latent_input = 2 * latent_size
        latent_output = latent_size
    elif autoencoder == 'quantized':
        latent_input = latent_output = latent_size[1]

    encoder_layers += [('linear', [latent_input])]
    encoder = FeedForward(input_size, encoder_layers, flatten=False)

    if decoder_layers is None:
        decoder_layers = encoder_layers[:-1]
        decoder_layers.append(('linear', [latent_output]))

        decoder_layers = list(reversed(decoder_layers))
        decoder = TransposedFF(decoder_layers, input_size, flatten=True)

    else:
        decoder.append(('linear', [np.prod(input_size)]))
        decoder = FeedForward(latent_size, decoder_layers, flatten=True)


    return get_init(autoencoder)(latent_size, encoder=encoder, decoder=decoder)
