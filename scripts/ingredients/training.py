import sys
import torch.nn as nn
from sacred import Ingredient
from ignite.engine import Events

if sys.path[0] != '../src':
    sys.path.insert(0, '../src')

from training.loss import CapacityScheduler, BetaScheduler
from training.handlers import EarlyStopping, ModelCheckpoint, Tracer
from training.optimizer import init_optimizer, init_lr_scheduler
from training.loss import init_metrics


def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = (y_pred.sigmoid() > 0.5).to(dtype=y.dtype)
    return y_pred, y


vae_loss = 'elbo', {'reconstruction_loss': 'bce'}
reconstruction_loss = 'recons_nll', {'loss': 'bce'}
bvae_loss = 'beta-elbo', {'reconstruction_loss': 'bce', 'beta': 4.0}
bxent_loss = 'bxent', {}
xent_loss = 'xent', {}
accuracy = 'acc', {'output_transform': thresholded_output_transform}
mse_loss = 'mse', {}


training = Ingredient('training')
training.add_named_config('vae', loss=vae_loss, metrics=[reconstruction_loss])
training.add_named_config('bvae', loss=bvae_loss, metrics=[reconstruction_loss])
training.add_named_config('binclass', loss=bxent_loss, metrics=[bxent_loss, accuracy])
training.add_named_config('class', loss=xent_loss, metrics=[xent_loss, accuracy])
training.add_named_config('decode_bce', loss=bxent_loss, metrics=[bxent_loss])
training.add_named_config('decode_mse', loss=mse_loss, metrics=[mse_loss])
training.add_named_config('recons_nll', loss=reconstruction_loss,
                                        metrics=[reconstruction_loss])


init_optimizer =  training.capture(init_optimizer)
init_metrics = training.capture(init_metrics)


@training.capture
def add_capacity_scheduling(trainer, loss, capacity_range=None,
                            capacity_patience=None):
    if capacity_range is not None:
        scheduler = CapacityScheduler(loss, capacity_range,capacity_patience)
        def reduce_capacity(engine):
            scheduler(engine.state.iteration)

        trainer.add_event_handler(Events.EPOCH_COMPLETED, reduce_capacity)


@training.capture
def add_beta_annealing(trainer, loss, beta_range=None, beta_patience=None):
    if beta_patience is not None:
        scheduler = BetaScheduler(loss, beta_range, beta_patience)
        def anneal_beta(engine):
            scheduler(engine.state.iteration)

        trainer.add_event_handler(Events.EPOCH_COMPLETED, anneal_beta)


@training.capture
def attach_lr_scheduler(optimizer, metric, validator,
                        scaling_mode='reduce-on-plateau',
                        lr_decay=0.01, lr_decay_patience=10):
    scheduler = init_lr_scheduler(optimizer, 'reduce-on-plateau',
                                    lr_decay=lr_scale,
                                    patience=lr_decay_patience)

    handler = LRScheduler(loss=metric, scheduler=scheduler)
    handler.attach(validator)

    return handler


@training.capture
def attach_model_checkpointing(validator, metric_name, model, save):
    # Model checkpoint and early stopping
    def score_fn(engine):
        return -engine.state.metrics[metric_name]

    checkpoint = ModelCheckpoint(
        dirname=save,
        filename_prefix='',
        score_function=score_fn,
        create_dir=True,
        require_empty=False,
        save_as_state_dict=True
    )
    validator.add_event_handler(
        Events.COMPLETED, checkpoint, {'model': model})

    return checkpoint


@training.capture
def attach_early_stopper(trainer, validator, metric_name, patience):
    def score_fn(engine):
        return -engine.state.metrics[metric_name]

    stopper = EarlyStopping(patience=patience, score_function=score_fn,
                            trainer=trainer)

    validator.add_event_handler(Events.COMPLETED, stopper)

    return stopper

def setup_training(
        model, validation_data, optim, loss, metrics, lr, l2_norm,
        rate_reg, clip, early_stopping, decay_lr, lr_scale,
        lr_decay_patience, save, device, trace=False, time=False
    ):

    metric_name = metrics[0][0]
    loss, metrics = init_metrics(loss, metrics)



    trainer = create_supervised_trainer(
        model, optimizer, loss, # grad_clip=clip,
        device=device, # reset_hidden=(not keep_hidden)
    )

    validator = create_supervised_evaluator(model, metrics, device=device)
    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        validator.run(validation_data)
