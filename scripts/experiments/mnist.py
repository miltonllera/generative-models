import sys
import numpy as np
import torch
from ignite.engine import Events
from ignite.engine import create_supervised_evaluator, create_supervised_trainer
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver

# Load experiment ingredients and their respective configs.
from ingredients.dataset import dataset, get_dataset_loader
from ingredients.models import model, init_model, create_conv_vae, VAEPredictor
from ingredients.training import training, init_metrics, init_optimizer
from ingredients.training import add_beta_annealing, add_capacity_scheduling
from ingredients.training import vae_loss, reconstruction_loss
from ingredients.training import attach_lr_scheduler, attach_model_checkpointing, \
                                 attach_early_stopper, Tracer

# Load configs
from configs.cnnvae import cnn_vae


# Set up experiment
ex = Experiment(name='gm-mnist', ingredients=[dataset, model, training])
ex.add_config(no_cuda=False, save_folder = '../data/sims/temp')
ex.add_package_dependency('torch', torch.__version__)

ex.observers.append(FileStorageObserver.create('../data/sims/mnist'))

# Functions
@ex.capture
def set_seed_and_device(seed, no_cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and not no_cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device

# Dataset configs
dataset.add_config('configs/mnist.yaml')

# Training configs
training.add_config('configs/training.yaml')
training.add_config(loss=vae_loss, metrics=[reconstruction_loss])
# training.add_named_config('banneal', 'configs/beta-annealing.yaml')

# Model configs
model.named_config(cnn_vae)


# Run experiment
@ex.automain
def main(_config, save_folder):
    max_epochs = _config['training']['max_epochs']
    batch_size = _config['training']['batch_size']

    device = set_seed_and_device()

    # Load data
    load_dataset = get_dataset_loader('mnist')
    training_set, validation_set = load_dataset(train=True, supervised=False,
                                                batch_size=batch_size)

    # Init model
    vae = init_model(init_fn=create_conv_vae, device=device)

    # Init metrics
    loss, metrics = init_metrics(vae_loss, [reconstruction_loss])
    optimizer = init_optimizer(params=vae.parameters())

    # Init engines
    trainer = create_supervised_trainer(vae, optimizer, loss, device=device)
    validator = create_supervised_evaluator(vae, metrics, device=device)
    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        validator.run(validation_set)

    # Add beta scheduling
    add_capacity_scheduling(trainer, loss)
    add_beta_annealing(trainer, loss)

    # Record training progression
    tracer = Tracer(metrics).attach(trainer)
    def log_training(engine):
        ex.log_scalar('training_elbo', tracer.loss[-1])

    def log_validation(engine):
        for metric, value in engine.state.metrics.items():
            ex.log_scalar('val_{}'.format(metric), value)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training)
    validator.add_event_handler(Events.EPOCH_COMPLETED, log_validation)

    # Save best model
    model_checkpoint = attach_model_checkpointing(validator, 'reconstruction',
                                                  vae, save_folder)

    trainer.run(training_set, max_epochs=max_epochs)

    # Select best model
    vae.load_state_dict(model_checkpoint.best_model)

    # Run on test data
    test_set = load_dataset(train=False, supervised=False, batch_size=batch_size)

    tester = create_supervised_evaluator(vae, metrics, device=device)
    test_metrics = tester.run(test_set).metrics

    # Save best model performance and state
    for metric, value in test_metrics.items():
        ex.log_scalar('test_{}'.format(metric), value)

    ex.add_artifact(model_checkpoint._saved[-1][1][0], 'trained-model')
