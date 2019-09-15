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
from ray.tune.logger import UnifiedLogger
from ray.tune.resources import Resources
from tqdm import trange

from . import utils

logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def logger_creator(config):
    return UnifiedLogger(config, config['logdir'])


class PyTorchTrainable(Trainable):
    @classmethod
    def default_resource_request(cls, config):
        return Resources(
            cpu=config.get("num_cpus", 2), gpu=config.get("num_gpus", 0))

    def _setup(self, config):
        runner_creator = config['runner_creator']
        runner_config = config['runner_config']
        runner_config.update({'num_gpus': config['num_gpus']})
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

    def _step(self, data_loader, phase):
        """Runs 1 training epoch"""
        # batch_time = utils.AverageMeter()
        # data_time = utils.AverageMeter()
        # losses = utils.AverageMeter()
        metrics = defaultdict(list)

        timers = {
            k: utils.TimerStat()
            for k in ["d2h", "fwd", "grad", "apply"]
        }

        if phase == 'tng':
            self.model.train()
            step_fcn = self.tng_step
            self._epoch += 1
        elif phase == 'val':
            self.model.eval()
            step_fcn = self.val_step
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
                metrics_ = step_fcn(samples)
                loss = metrics_['loss']
                for k, m in metrics_.items():
                    metrics[k].append(utils.to_numpy(m))

                # measure accuracy and record loss
                # losses.update(loss.item(), target.size(0))

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

        # stats = {
        #     "batch_time": batch_time.avg,
        #     "batch_processed": losses.count,
        #     "tng_loss": losses.avg,
        #     "data_time": data_time.avg,
        # }
        metrics = {f'{phase}/{k}': np.mean(v) for k, v in metrics.items()}
        # stats.update({k: t.mean for k, t in timers.items()})
        return metrics

    # def _val(self):
    #     batch_time = utils.AverageMeter()
    #     losses = utils.AverageMeter()

    #     # switch to evaluate mode
    #     self.model.eval()

    #     with torch.no_grad():
    #         end = time.time()
    #         for i, samples in enumerate(self.val_loader):

    #             if self.num_gpus:
    #                 samples = [s.cuda(non_blocking=True) for s in samples]

    #             # compute output
    #             output = self.model(samples)
    #             target = samples[-1]
    #             loss = self.criterion(output, target)

    #             # measure accuracy and record loss
    #             losses.update(loss.item(), target.size(0))

    #             # measure elapsed time
    #             batch_time.update(time.time() - end)
    #             end = time.time()

    #     stats = {"batch_time": batch_time.avg, "val_loss": losses.avg}
    #     return stats

    # def _inf(self):
    #     raise NotImplementedError
    #     # switch to evaluate mode
    #     self.model.eval()

    #     with torch.no_grad():
    #         outputs = []
    #         for i, (features, target) in enumerate(self.inf_loader):

    #             if self.num_gpus:
    #                 features = features.cuda(non_blocking=True)
    #                 target = target.cuda(non_blocking=True)

    #             # compute output
    #             output = self.model(features)
    #             outputs.append(output)

    #     return utils.to_numpy(torch.cat(outputs, dim=0))

    @abstractmethod
    def tng_step(self, samples):
        pass

    @abstractmethod
    def val_step(self, samples):
        pass

    def train(self):
        tng_stats = self._step(self.tng_loader, 'tng')
        val_stats = self._step(self.val_loader, 'val')

        tng_stats.update(val_stats)
        tng_stats.update({'epoch': self.epoch})
        return tng_stats

    # def inference(self):
    #     """Evaluates the model on the validation data set."""
    #     with self._timers["inference"]:
    #         outputs = self._inf()

    #     return outputs

    # def stats(self):
    #     """Returns a dictionary of statistics collected."""
    #     stats = {"epoch": self.epoch}
    #     for k, t in self._timers.items():
    #         stats[k + "_time_mean"] = t.mean
    #         stats[k + "_time_total"] = t.sum
    #         t.reset()
    #     return stats

    def get_state(self):
        """Returns the state of the runner."""
        return {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            # "stats": self.stats()
        }

    def set_state(self, state):
        """Sets the state of the model."""
        # TODO: restore timer stats
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epoch = state["epoch"]

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
