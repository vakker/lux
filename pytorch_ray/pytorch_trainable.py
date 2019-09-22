import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from glob import glob
from os import path as osp

import numpy as np
import ray
import torch
from ray.tune import Trainable
from ray.tune.resources import Resources
from ray.tune.result import (DEFAULT_RESULTS_DIR, DONE, EPISODES_THIS_ITER,
                             EPISODES_TOTAL, RESULT_DUPLICATE,
                             TIME_THIS_ITER_S, TIMESTEPS_THIS_ITER,
                             TIMESTEPS_TOTAL, TRAINING_ITERATION)
from tqdm import tqdm, trange

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

        # self._runner.log_graph(self._result_logger)

        if config.get('reload'):
            dirs = glob(osp.join(self.logdir, 'checkpoint_*'))
            dirs = sorted(dirs, key=lambda x: int(x.split('_')[-1]))
            self.restore(osp.join(dirs[-1], 'model.pth'))
        elif config.get('restore_from'):
            self.restore(config['restore_from'])

    def _train(self):
        timers = defaultdict(utils.TimerStat)

        with timers["tng"]:
            tng_stats = self._runner.tng()

        with timers["val"]:
            if (self.epoch == 1
                    or self.epoch % self.config.get('val_freq', 1) == 0
                    or self.epoch == self.config['max_epochs']):
                val_stats = self._runner.val()
            else:
                val_stats = {}

        with timers["save"]:
            if (self.epoch == 1
                    or self.epoch % self.config.get('save_freq', 1) == 0
                    or self.epoch == self.config['max_epochs']):
                self.save()

        with timers["log"]:
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

        logging.debug('#' * 10)
        logging.debug('Trainable')
        for k, v in timers.items():
            t_mean = v.mean
            if t_mean is not None:
                logging.debug(f'{k}: {t_mean:.4f}')
        logging.debug('#' * 10)
        return stats

    def train(self):
        start = time.time()
        result_all = self._train()
        assert isinstance(result_all, dict), "_train() needs to return a dict."
        result = result_all['scalar']
        assert isinstance(result, dict), "_train() needs to return a dict."

        # We do not modify internal state nor update this result if duplicate.
        if RESULT_DUPLICATE in result:
            return result

        result = result.copy()

        self._iteration += 1
        self._iterations_since_restore += 1

        if result.get(TIME_THIS_ITER_S) is not None:
            time_this_iter = result[TIME_THIS_ITER_S]
        else:
            time_this_iter = time.time() - start
        self._time_total += time_this_iter
        self._time_since_restore += time_this_iter

        result.setdefault(DONE, False)

        # self._timesteps_total should only be tracked if increments provided
        if result.get(TIMESTEPS_THIS_ITER) is not None:
            if self._timesteps_total is None:
                self._timesteps_total = 0
            self._timesteps_total += result[TIMESTEPS_THIS_ITER]
            self._timesteps_since_restore += result[TIMESTEPS_THIS_ITER]

        # self._episodes_total should only be tracked if increments provided
        if result.get(EPISODES_THIS_ITER) is not None:
            if self._episodes_total is None:
                self._episodes_total = 0
            self._episodes_total += result[EPISODES_THIS_ITER]

        # self._timesteps_total should not override user-provided total
        result.setdefault(TIMESTEPS_TOTAL, self._timesteps_total)
        result.setdefault(EPISODES_TOTAL, self._episodes_total)
        result.setdefault(TRAINING_ITERATION, self._iteration)
        result_all.setdefault(TIMESTEPS_TOTAL, self._timesteps_total)
        result_all.setdefault(EPISODES_TOTAL, self._episodes_total)
        result_all.setdefault(TRAINING_ITERATION, self._iteration)

        # Provides auto-filled neg_mean_loss for avoiding regressions
        if result.get("mean_loss"):
            result.setdefault("neg_mean_loss", -result["mean_loss"])

        now = datetime.today()
        result.update(
            experiment_id=self._experiment_id,
            date=now.strftime("%Y-%m-%d_%H-%M-%S"),
            timestamp=int(time.mktime(now.timetuple())),
            time_this_iter_s=time_this_iter,
            time_total_s=self._time_total,
            pid=os.getpid(),
            hostname=os.uname()[1],
            node_ip=self._local_ip,
            config=self.config,
            time_since_restore=self._time_since_restore,
            timesteps_since_restore=self._timesteps_since_restore,
            iterations_since_restore=self._iterations_since_restore)

        monitor_data = self._monitor.get_data()
        if monitor_data:
            result.update(monitor_data)
            result_all.update(monitor_data)

        self._log_result(result_all)

        return result

    def val(self):
        return self._runner.val()['scalar']

    def inf(self, subset='val'):
        return self._runner.inf(subset)

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
        for i in trange(self.config['max_epochs'], initial=self.epoch):
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
            shuffle=config.get('shuff', False),
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

    # def log_graph(self, tb_logger):
    #     self.model.eval()

    #     samples = next(iter(self.val_loader))
    #     if self.num_gpus:
    #         samples = [s.cuda(non_blocking=True) for s in samples]
    #     self._log_graph(tb_logger, samples)

    # def _log_graph(self, tb_logger, samples):
    #     pass

    def _step(self, data_loader, phase):
        """Runs 1 training epoch"""
        # batch_time = utils.AverageMeter()
        data_time = []
        # losses = utils.AverageMeter()
        batch_outputs = []

        timers = {
            k: utils.TimerStat()
            for k in ["d2h", "fwd", "log", "grad", "apply", "log2"]
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
        elif phase == 'inf':
            self.model.eval()
            step_fcn = self.inf_step
            post_step_fcn = self.post_inf_step
        else:
            raise NotImplementedError(f'Phase {phase} not implemeneted')

        end = time.time()

        for i, samples in enumerate(data_loader):
            # measure data loading time
            data_time.append(time.time() - end)

            # Create non_blocking tensors for distributed training
            with timers["d2h"]:
                if self.num_gpus:
                    samples = [s.cuda(non_blocking=True) for s in samples]

            # compute output
            with timers["fwd"]:
                loss, outputs = step_fcn(samples)

            with timers["log"]:
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
            end = time.time()

        with timers["log2"]:
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
        for k, v in timers.items():
            t_mean = v.mean
            if t_mean is not None:
                logging.debug(f'{k}: {t_mean:.4f}')
        t_mean = np.mean(data_time)
        logging.debug(f'data_time: {t_mean:.4f}')
        logging.debug('#' * 10)

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

    def inf_step(self, samples):
        raise NotImplementedError

    def post_inf_step(self, batch_outputs):
        raise NotImplementedError

    def tng(self):
        return self._step(self.tng_loader, 'tng')

    def val(self):
        return self._step(self.val_loader, 'val')

    def inf(self, subset='val'):
        if subset == 'val':
            return self._step(self.val_loader, 'inf')
        elif subset == 'tng':
            return self._step(self.tng_loader, 'inf')
        else:
            raise NotImplementedError

    def _get_model_params(self):
        params = {
            'param/' + name: utils.to_numpy(param)
            for name, param in self.model.named_parameters()
        }
        params.update({
            'grad/' + name: utils.to_numpy(param.grad)
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
