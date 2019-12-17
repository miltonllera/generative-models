import csv
import numpy as np
from ignite.engine import Events
from ignite.handlers import *
from tqdm import tqdm


class LRSchedulerHandler(object):
    def __init__(self, scheduler, loss):
        self.scheduler = scheduler
        self.loss = loss

    def __call__(self, engine):
        loss_val = engine.state.metrics[self.loss]
        self.scheduler.step(loss_val)

    def attach(self, engine):
        engine.add_event_handler(Events.COMPLETED, self)
        return self


class TracerHandler(object):
    def __init__(self, val_metrics, save_path=None, save_interval=1):
        self.metrics = ['loss']
        self.loss = []
        self._batch_trace = []
        self.save_path = save_path
        self.save_interval = save_interval

        template = 'val_{}'
        for k in val_metrics:
            name = template.format(k)
            setattr(self, name, [])
            self.metrics.append(name)

    def _initalize_traces(self, engine):
        for k in self.metrics:
            getattr(self, k).clear()

    def _save_batch_loss(self, engine):
        self._batch_trace.append(engine.state.output)

    def _trace_training_loss(self, engine):
        avg_loss = np.mean(self._batch_trace)
        self.loss.append(avg_loss)
        self._batch_trace.clear()

    def _trace_validation(self, engine):
        metrics = engine.state.metrics
        template = 'val_{}'
        for k, v in metrics.items():
            trace = getattr(self, template.format(k))
            trace.append(v)

    def attach(self, trainer, evaluator=None):
        trainer.add_event_handler(Events.STARTED, self._initalize_traces)
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, self._save_batch_loss)
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, self._trace_training_loss)

        if evaluator is not None:
            evaluator.add_event_handler(
                Events.COMPLETED, self._trace_validation)

        if self.save_path is not None:
            trainer.add_event_handler(
                Events.EPOCH_COMPLETED, self._save_at_interval)

        return self

    def _save_at_interval(self, engine):
        if engine.state.iteration % self.save_interval == 0:
            self.save_traces()

    def save_traces(self):
        for loss in self.metrics:
            trace = getattr(self, loss)
            with open('{}/{}.csv'.format(self.save_path, loss), mode='w') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                for i, v in enumerate(trace):
                    wr.writerow([i + 1, v])


class LoggerHandler(object):
    def __init__(self, loader, log_interval, pbar=None, desc=None):
        n_batches = len(loader)
        self.desc = 'iteration-loss: {:.5f}' if desc is None else desc
        self.pbar = pbar or tqdm(
            initial=0, leave=False, total=n_batches,
            desc=self.desc.format(0)
        )
        self.log_interval = log_interval
        self.running_loss = 0
        self.n_batches = n_batches

    def _log_batch(self, engine):
        self.running_loss += engine.state.output

        iter = (engine.state.iteration - 1) % self.n_batches + 1
        if iter % self.log_interval == 0:
            self.pbar.desc = self.desc.format(
                engine.state.output)
            self.pbar.update(self.log_interval)

    def _log_epoch(self, engine):
        self.pbar.refresh()
        tqdm.write("Epoch: {} - avg loss: {:.5f}"
            .format(engine.state.epoch, self.running_loss / self.n_batches))
        self.running_loss = 0
        self.pbar.n = self.pbar.last_print_n = 0

    def _log_validation(self, engine):
        metrics = self.evaluator.state.metrics

        message = []
        for k, v in metrics.items():
            message.append("{}: {:.5f}".format(k, v))
        tqdm.write('\tvalidation: ' + ' - '.join(message))

    def attach(self, trainer, evaluator=None, metrics=None):
        trainer.add_event_handler(Events.ITERATION_COMPLETED, self._log_batch)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self._log_epoch)
        trainer.add_event_handler(Events.COMPLETED, lambda x: self.pbar.close())

        if evaluator is not None and metrics is None:
            raise ValueError('')

        if evaluator is not None:
            self.evaluator = evaluator
            trainer.add_event_handler(Events.EPOCH_COMPLETED, self._log_validation)

        return self