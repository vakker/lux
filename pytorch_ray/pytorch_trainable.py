import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from os import path as osp

import numpy as np
import ray
import torch
from ray.tune import Trainable
from ray.tune.resources import Resources

from . import utils

logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class PyTorchTrainable(Trainable):
    @classmethod
    def default_resource_request(cls, config):
        return Resources(
            cpu=config.get("num_cpus", 2), gpu=config.get("num_gpus", 0))

    def _setup(self, config):
        runner_creator = config['runner_creator']
        runner_config = config['runner_config']
        self._runner = runner_creator(config=runner_config)

    def _train(self):
        tng_stats = self._runner.train()
        tng_stats['exp_id'] = self._experiment_id
        return tng_stats

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

    def _tng(self):
        """Runs 1 training epoch"""
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        losses = utils.AverageMeter()

        timers = {
            k: utils.TimerStat()
            for k in ["d2h", "fwd", "grad", "apply"]
        }

        # switch to train mode
        self.model.train()

        end = time.time()

        for i, (features, target) in enumerate(self.tng_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # Create non_blocking tensors for distributed training
            with timers["d2h"]:
                if self.num_gpus:
                    features = features.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

            # compute output
            with timers["fwd"]:
                output = self.model(features)
                loss = self.criterion(output, target)

                # measure accuracy and record loss
                losses.update(loss.item(), features.size(0))

            with timers["grad"]:
                # compute gradients in a backward pass
                self.optimizer.zero_grad()
                loss.backward()

            with timers["apply"]:
                # Call step of optimizer to update model params
                self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        stats = {
            "batch_time": batch_time.avg,
            "batch_processed": losses.count,
            "tng_loss": losses.avg,
            "data_time": data_time.avg,
        }
        stats.update({k: t.mean for k, t in timers.items()})
        return stats

    def _val(self):
        batch_time = utils.AverageMeter()
        losses = utils.AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (features, target) in enumerate(self.val_loader):

                if self.num_gpus:
                    features = features.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                # compute output
                output = self.model(features)
                loss = self.criterion(output, target)

                # measure accuracy and record loss
                losses.update(loss.item(), features.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        stats = {"batch_time": batch_time.avg, "val_loss": losses.avg}
        return stats

    def _inf(self):
        raise NotImplementedError
        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            outputs = []
            for i, (features, target) in enumerate(self.inf_loader):

                if self.num_gpus:
                    features = features.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                # compute output
                output = self.model(features)
                outputs.append(output)

        return utils.to_numpy(torch.cat(outputs, dim=0))

    def tng_step(self):
        """Runs a training epoch and updates the model parameters."""
        logger.debug(f"Begin Training Epoch {self.epoch + 1}")
        with self._timers["training"]:
            train_stats = self._tng()
            train_stats["epoch"] = self.epoch

        self._epoch += 1

        train_stats.update(self.stats())
        return train_stats

    def val_step(self):
        """Evaluates the model on the validation data set."""
        with self._timers["validation"]:
            validation_stats = self._val()

        validation_stats.update(self.stats())
        return validation_stats

    def train(self):
        train_stats = self.tng_step()
        validation_stats = self.val_step()

        train_stats.update(validation_stats)
        train_stats.update({'hparams': self.config['hparams']})
        return train_stats

    def inference(self):
        """Evaluates the model on the validation data set."""
        with self._timers["inference"]:
            outputs = self._inf()

        return outputs

    def stats(self):
        """Returns a dictionary of statistics collected."""
        stats = {"epoch": self.epoch}
        for k, t in self._timers.items():
            stats[k + "_time_mean"] = t.mean
            stats[k + "_time_total"] = t.sum
            t.reset()
        return stats

    def get_state(self):
        """Returns the state of the runner."""
        return {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "stats": self.stats()
        }

    def set_state(self, state):
        """Sets the state of the model."""
        # TODO: restore timer stats
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epoch = state["stats"]["epoch"]

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
