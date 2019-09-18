import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from os import path as osp

import numpy as np
import ray
import torch
from ray.tune import Trainable
from ray.tune.resources import Resources
from tqdm import trange

from . import utils
from .logger import PRLogger, pr_logger_creator

logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class PyTorchTrainable(Trainable):
    def __init__(self, config=None, logger_creator=None):
        if logger_creator is None:
            logger_creator = pr_logger_creator
        super().__init__(config, logger_creator)

    def _setup(self, config):
        runner_creator = config['runner_creator']
        runner_config = config['runner_config']
        runner_config.update({'num_gpus': config['num_gpus']})
        self._runner = runner_creator(config=runner_config)

        self._runner.log_graph(self._result_logger)

        if 'restore_from' in config:
            self.restore(config['restore_from'])

    def _train(self):
        tng_stats = self._runner.tng()
        if (self.epoch == 1 or self.epoch % self.config.get('val_freq', 1) == 0
                or self.epoch == self.config['max_epochs']):
            val_stats = self._runner.val()
        else:
            val_stats = {}

        if (self.epoch == 1
                or self.epoch % self.config.get('save_freq', 1) == 0
                or self.epoch == self.config['max_epochs']):
            self.save()

        keys = set(list(tng_stats.keys()) + list(val_stats.keys()))
        stats = {
            k: {
                **tng_stats.get(k, {}),
                **val_stats.get(k, {})
            }
            for k in keys
        }

        if isinstance(self._result_logger, PRLogger):
            self._result_logger.log(stats, self.epoch)
        return stats['scalar']

    def val(self):
        return self._runner.val()['scalar']

    def _save(self, checkpoint_dir):
        checkpoint = osp.join(checkpoint_dir, "model.pth")
        state = self._runner.get_state()
        torch.save(state, checkpoint)
        return checkpoint

    def _restore(self, checkpoint):
        state = torch.load(checkpoint)
        return self._runner.set_state(state)

    def _stop(self):
        self._runner.shutdown()

    @property
    def epoch(self):
        return self._runner.epoch

    def fit(self):
        for i in trange(self.config['max_epochs']):
            _ = self.train()

    @classmethod
    def default_resource_request(cls, config):
        return Resources(
            cpu=config.get("num_cpus", 2), gpu=config.get("num_gpus", 0))


class PyTorchRunner(ABC):
    """Manages a PyTorch model for training."""

    def __init__(self, config=None):
        """Initializes the runner.

        Args:
            model_creator (dict -> torch.nn.Module): see pytorch_trainer.py.
            data_creator (dict -> Dataset, Dataset): see pytorch_trainer.py.
            optimizer_creator (torch.nn.Module, dict -> loss, optimizer):
                see pytorch_trainer.py.
            config (dict): see pytorch_trainer.py.
            batch_size (int): see pytorch_trainer.py.
        """

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.tng_set = None
        self.val_set = None
        self.tng_loader = None
        self.val_loader = None
        self.config = {} if config is None else config
        self.batch_size = self.config.get('batch_size', 16)
        self.num_gpus = self.config.get('num_gpus', 0)
        self.verbose = True

        self._epoch = 0
        self._timers = {
            k: utils.TimerStat(window_size=1)
            for k in [
                "setup_proc", "setup_model", "get_state", "set_state",
                "validation", "training", "inference"
            ]
        }

        self._setup(self.config)
        self._bests = {}

    @property
    def epoch(self):
        return self._epoch

    @abstractmethod
    def model_creator(self, config):
        pass

    @abstractmethod
    def data_creator(self, config):
        pass

    @abstractmethod
    def optimizer_creator(self, model, config):
        pass

    def _setup(self, config):
        """Initializes the model."""
        logger.debug("Creating model")
        self.model = self.model_creator(config)
        if self.num_gpus:
            self.model = self.model.cuda()

        logger.debug("Creating optimizer")
        self.criterion, self.optimizer = self.optimizer_creator(
            self.model, config)
        if self.num_gpus:
            self.criterion = self.criterion.cuda()

        logger.debug("Creating dataset")
        self.tng_set, self.val_set = self.data_creator(config)

        self.tng_loader = torch.utils.data.DataLoader(
            self.tng_set,
            batch_size=config.get('batch_size', 4),
            shuffle=True,
            num_workers=config.get('ds_workers', 2),
            pin_memory=False)

        self.val_loader = torch.utils.data.DataLoader(
            self.val_set,
            batch_size=config.get('batch_size', 4),
            shuffle=False,
            num_workers=config.get('ds_workers', 2),
            pin_memory=False)

        logging.info(f'DataSet(tng) len: {len(self.tng_set)}')
        logging.info(f'DataLoader(tng) len: {len(self.tng_loader)}')
        logging.info(f'DataSet(val) len: {len(self.val_set)}')
        logging.info(f'DataLoader(val) len: {len(self.val_loader)}')

    def log_graph(self, tb_logger):
        self.model.eval()

        samples = next(iter(self.val_loader))
        if self.num_gpus:
            samples = [s.cuda(non_blocking=True) for s in samples]
        self._log_graph(tb_logger, samples)

    def _log_graph(self, tb_logger, samples):
        pass

    def _step(self, data_loader, phase):
        """Runs 1 training epoch"""
        # batch_time = utils.AverageMeter()
        # data_time = utils.AverageMeter()
        # losses = utils.AverageMeter()
        batch_outputs = []

        timers = {
            k: utils.TimerStat()
            for k in ["d2h", "fwd", "grad", "apply"]
        }

        if phase == 'tng':
            self.model.train()
            step_fcn = self.tng_step
            post_step_fcn = self.post_tng_step
            self._epoch += 1
        elif phase == 'val':
            self.model.eval()
            step_fcn = self.val_step
            post_step_fcn = self.post_val_step
        else:
            raise NotImplementedError(f'Phase {phase} not implemeneted')

        # end = time.time()

        for i, samples in enumerate(data_loader):
            # measure data loading time
            # data_time.update(time.time() - end)

            # Create non_blocking tensors for distributed training
            with timers["d2h"]:
                if self.num_gpus:
                    samples = [s.cuda(non_blocking=True) for s in samples]

            # compute output
            with timers["fwd"]:
                loss, outputs = step_fcn(samples)

                assert isinstance(outputs, dict)
                outputs = {k: utils.to_numpy(v) for k, v in outputs.items()}
                outputs.update({'batch_num': i})
                batch_outputs.append(outputs)

            if phase != 'tng':
                continue

            with timers["grad"]:
                # compute gradients in a backward pass
                self.optimizer.zero_grad()
                loss.backward()

            with timers["apply"]:
                # Call step of optimizer to update model params
                self.optimizer.step()

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()

        metrics = post_step_fcn(batch_outputs)
        # stats = {
        #     "batch_time": batch_time.avg,
        #     "batch_processed": losses.count,
        #     "tng_loss": losses.avg,
        #     "data_time": data_time.avg,
        # }
        if phase == 'tng':
            if 'histogram' not in metrics:
                metrics.update({'histogram': {}})
            metrics['histogram'].update(self._get_model_params())

        return metrics

    @abstractmethod
    def tng_step(self, samples):
        pass

    @abstractmethod
    def post_tng_step(self, batch_outputs):
        pass

    @abstractmethod
    def val_step(self, samples):
        pass

    @abstractmethod
    def post_val_step(self, batch_outputs):
        pass

    def tng(self):
        return self._step(self.tng_loader, 'tng')

    def val(self):
        return self._step(self.val_loader, 'val')

    def _get_model_params(self):
        params = {
            'param/' + name: param
            for name, param in self.model.named_parameters()
        }
        params.update({
            'grad/' + name: param.grad
            for name, param in self.model.named_parameters()
        })
        return params

    def get_state(self):
        """Returns the state of the runner."""
        return {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def set_state(self, state):
        """Sets the state of the model."""
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._epoch = state["epoch"]

    def shutdown(self):
        """Attempts to shut down the worker."""
        del self.val_loader
        del self.val_set
        del self.tng_loader
        del self.tng_set
        del self.criterion
        del self.optimizer
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
