import sys
import torch
import torch.nn as nn
import numpy as np
from io import BytesIO
from sacred import Ingredient
from incense import ExperimentLoader

if sys.path[0] != '../src':
    sys.path.insert(0, '../src')

from models.init import init_vae
from models.convolution import CNN, TransposedCNN
from models.mlp import create_mlp, kaiming_normal_init_


model = Ingredient('model')

# Basic init functions, add more later
_init_vae = model.capture(init_vae)
_init_cnn = model.capture(CNN)
_init_tcnn = model.capture(TransposedCNN)
_init_mlp = model.capture(create_mlp)


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
    model = init_model()
    print(model)


# Convolutional VAE
class ConvVAE(nn.Module):
    def __init__(self, vae, cnn, transposed_cnn):
        super().__init__()
        self.cnn = cnn
        self.vae = vae
        self.transposed_cnn = transposed_cnn

    @property
    def latent_size(self):
        return self.vae.latent_size

    def forward(self, inputs):
        inputs = self.cnn(inputs)
        params, z = self.vae(inputs)
        recons = self.transposed_cnn(z)

        return params, recons

    def embed(self, inputs):
        h = self.cnn(inputs)
        return self.vae.embed(h)

    def decode(self, z):
        h_prime = self.vae.decode(z)
        return self.transposed_cnn(h_prime)

    def sample_latent(self, inputs, max_samples):
        h_prime = self.cnn(inputs)
        return self.vae.sample_latent(h_prime, max_samples)


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
def create_conv_vae(vae_type, input_size, kernels, pools, encoder_sizes,
                    latent_size, batch_norm=False):
    cnn = CNN(input_size, kernels, pools, batch_norm=batch_norm)

    vae = init_vae(vae_type, np.prod(cnn.output_size), encoder_sizes, latent_size,
                   batch_norm=batch_norm, raw_output=False)

    tCNN = TransposedCNN(input_size, kernels, pools, batch_norm=batch_norm)

    return ConvVAE(vae, cnn, tCNN)
