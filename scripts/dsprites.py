import sys
import os
import argparse
import datetime
import yaml

import numpy as np
import torch
import torch.nn as nn

from ignite.engine import Events
from ignite.metrics import Loss

sys.path.insert(0, '../src')

from models.vae import cVAE
from dataset.mnist import load_mnist
from training.loss import BetaELB, CapacityScheduler
from training.optimizer import init_optimizer, init_lr_scheduler
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
import training.handlers as hdlr


###############################################################################
# PARSE THE INPUT
###############################################################################

parser = argparse.ArgumentParser(
    description='Dsprite data to test disentangled representations')

# Model parameters
parser.add_argument('--latent-size', type=int, default=50,
                    help='dimensionality of the latent representation')
parser.add_argument('--latent-type', type=str, default='diagonal',
                    help='stochastic layer parametrization,'
                    'one of diagonal|homoscedastic')
parser.add_argument('--encoder-sizes', nargs='+', type=int, default=[100],
                    help='the encoder layer size. decoder sizes are inferred'
                    'from this parameter.')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='the beta term penalising the KL term in the ELBO')
parser.add_argument('--capacity-range', nargs='+', type=float, default=(0,),
                    help='Initial, target and step values for varying the capacity')
parser.add_argument('--capacity-patience', type=int, default=0,
                    help='Number of epochs between capacity increases')

# Data parameters
parser.add_argument('--data-path', type=str, default='../data/',
                    help='location of the data set')
parser.add_argument('--val-split', type=float, default=0.2,
                    help='proportion of trainig data used for validation')
parser.add_argument('--shuffle', action='store_true',
                    help='shuffle the data at the start of each epoch.')

# Training parameters
parser.add_argument('--epochs', type=int, default=40,
                    help='max number of training epochs')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='batch size')
parser.add_argument('--recons-loss', type=str, default='bce',
                    help='reconstruction loss used, supports: bce|mse')
parser.add_argument('--optim', type=str, default='rmsprop',
                    help='learning rule, supports:'
                    'adam|sparseadam|adamax|rmsprop|sgd|adagrad|adadelta')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--l2-norm', type=float, default=0,
                    help='weight of L2 norm')
parser.add_argument('--early-stopping', action='store_true',
                    help='use early stopping')

# Replicability and storage
parser.add_argument('--save', type=str,  default='../data/sims/test',
                    help='path to save the final model')
parser.add_argument('--seed', type=int, default=18092,
                    help='random seed')

# CUDA
parser.add_argument('--no-cuda', action='store_true', help='disable CUDA')

# Print options
parser.add_argument('--verbose', action='store_true',
                    help='print the progress of training to std output.')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='report interval')

args = parser.parse_args()

###############################################################################
# SETTING UP THE DIVICE AND SEED
###############################################################################

torch.manual_seed(args.seed)
if not args.no_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

###############################################################################
# LOAD DATA
###############################################################################

input_size = 64 * 64

 # Load training data
train_data, validation_data, test_data = load_mnist(
    args.data_path, input_size, args.batch_size, args.val_split,
    shuffle=args.shuffle, download=True, supervised=False
)

###############################################################################
# CREATE THE MODEL
###############################################################################


encoder_sizes, latent_size = args.encoder_sizes, args.latent_size

if len(encoder_sizes) == 1 and encoder_sizes[0] == 0:
    encoder_sizes = []

vae = cVAE(
    input_size=input_size,
    encoder_sizes=encoder_sizes,
    latent_size=latent_size,
    latent_type=args.latent_type
)

###############################################################################
# SET UP OPTIMIZER & OBJECTIVE FUNCTION
###############################################################################

epochs, log_interval, save_path = args.epochs, args.log_interval, args.save

criterion = BetaELB(reconstruction_loss=args.recons_loss, 
                    gamma=args.gamma, capacity=args.capacity_range[0])
                
metrics = {'elbo': Loss(criterion)}

optimizer = init_optimizer(args.optim, vae.parameters(), args.lr, args.l2_norm)

trainer = create_supervised_trainer(vae, optimizer, criterion, device=device)
validator = create_supervised_evaluator(vae, metrics, device=device)

@trainer.on(Events.EPOCH_COMPLETED)
def validate(engine):
    validator.run(validation_data)

if len(args.capacity_range) == 3:
    capschedule = CapacityScheduler(
        criterion, args.capacity_range, args.capacity_patience)
    trainer.add_event_handler(Events.STARTED, capschedule)

# Tracing
tracer = hdlr.TracerHandler(
    metrics.keys(), args.save, save_interval=1
).attach(trainer, validator)

if args.verbose:
    logger = hdlr.LoggerHandler(train_data, args.log_interval
                                ).attach(trainer, validator, metrics.keys())

# Early stopping and model checkpoint


def score_fn(engine):
    return -engine.state.metrics['elbo']


if args.early_stopping:
    stopper = hdlr.EarlyStopping(
        patience=100,
        score_function=score_fn,
        trainer=trainer
    )
    validator.add_event_handler(Events.COMPLETED, stopper)

checkpoint = hdlr.ModelCheckpoint(
    dirname=save_path,
    filename_prefix='',
    score_function=score_fn,
    create_dir=True,
    require_empty=False,
    save_as_state_dict=True
)

validator.add_event_handler(Events.COMPLETED, checkpoint, {'model': vae})

# Training time
timer = hdlr.Timer(average=False)
timer.attach(trainer)


###############################################################################
# TRAIN MODEL
###############################################################################

# Train model

trainer.run(train_data, max_epochs=epochs)

# Test
best_model_path = str(checkpoint._saved[-1][1][0])
with open(best_model_path, mode='rb') as f:
    state_dict = torch.load(f)
vae.load_state_dict(state_dict)

tester = create_supervised_evaluator(vae, metrics, device=device)
tester.run(test_data)

# Testing preformance
test_loss = tester.state.metrics['elbo']

print('Training ended: test loss {:5.4f}'.format(test_loss))

###############################################################################
# Save metadata
###############################################################################

print('Saving results....')

data_params = {
    'dataset-type': args.dataset_type,
    'training-size': args.training_size,
    'train_contrast': args.train_contrast,
    'train-noise': args.train_noise,
    'val-ratio': args.val_precentage,
    'test-size': args.test_size,
    'test_contrast': args.test_contrast,
    'test-noise': args.test_noise,
    'denoise': args.denoise if  args.dataset_type == 'unsupervised' else False
}

model_params = {
    'model_type': 'vae',
    'input_size': input_size,
    'encoder_sizes': encoder_sizes,
    'latent_size': latent_size,
    'latent_type': args.latent_type
}

learning_params = {
    'optimizer': args.optim,
    'learning-rate': args.lr,
    'l2-norm': args.l2_norm,
    'criterion': 'elbo',
    'batch-size': batch_size,
}

meta = {
    'data-params': data_params,
    'model-params': model_params,
    'learning-params': learning_params,
    'info': {
        'test-score': test_loss,
        'training-time': timer.value(),
        'timestamp': datetime.datetime.now()
    }
}

with open(save_path + '/meta.yaml', mode='w') as f:
    yaml.dump(meta, f)

print('Done.')
