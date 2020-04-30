import logging
import re
from os import path as osp

import numpy as np
import tensorflow as tf
from ray.tune.logger import DEFAULT_LOGGERS, CSVLogger, JsonLogger, Logger
from ray.tune.result import TIMESTEPS_TOTAL, TRAINING_ITERATION
from ray.tune.utils import flatten_dict
from tensorboard.plugins.hparams import api as hp
from tensorflow.python.eager import context
from torch.utils.tensorboard import SummaryWriter


def tb_logger_creator(config, logdir=None, trial=None):
    if logdir is None:
        logdir = config['logdir']
    return TBLogger(config, logdir, trial)


def mlf_logger_creator(config, logdir=None, trial=None):
    if logdir is None:
        logdir = config['logdir']

    return MLFLogger(config, logdir, trial)


LOGGERS = [JsonLogger, CSVLogger] + [tb_logger_creator, mlf_logger_creator]
# LOGGERS = DEFAULT_LOGGERS + ['MLFLowLogger']

# class PRLogger(Logger):
#     def _init(self):
#         if self.config.get('restore_from') or self.config.get('reload'):
#             purge_step = None
#         else:
#             purge_step = 0

#         self._tb_writer = SummaryWriter(
#             log_dir=osp.join(self.logdir, 'logs'), purge_step=purge_step)

#         self._hp_logged = False

#     def on_result(self, result):
#         step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
#         self.log(result, step)

#     def log(self, result, step):
#         tmp = result.copy()

#         if not self._hp_logged:
#             self._log_hparams(tmp['scalar'])
#             self._hp_logged = True

#         log_types = ['scalar', 'histogram', 'image', 'figure', 'text']
#         for t in log_types:
#             for k, v in tmp.get(t, {}).items():
#                 log_fcn = getattr(self._tb_writer, f'add_{t}')
#                 log_fcn(k, v, global_step=step)

#         self._tb_writer.flush()

#     def _log_hparams(self, metric_dict):
#         if hasattr(self, 'trial'):
#             if self.trial and self.trial.evaluated_params:
#                 ep = {
#                     p.replace('runner_config', '').replace(
#                         'hparams', '').replace('//', '/').strip('/'): v
#                     for p, v in self.trial.evaluated_params.items()
#                 }
#                 self._tb_writer.add_hparams(
#                     hparam_dict=ep, metric_dict=metric_dict)

#     def add_graph(self, model, inputs):
#         self._tb_writer.add_graph(model, inputs)


class TBLogger(Logger):
    def _init(self):
        self._file_writer = None
        self._hp_logged = False

    def on_result(self, result):
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        self.log(result, step)

    def log(self, result, step):
        if self._file_writer is None:
            self._context = context
            self._file_writer = tf.summary.create_file_writer(self.logdir)
        with tf.device("/CPU:0"), self._context.eager_mode():
            with tf.summary.record_if(True), self._file_writer.as_default():

                tmp = result.copy()
                if not self._hp_logged:
                    self._log_hparams()
                    self._hp_logged = True

                log_types = ['scalar', 'histogram', 'image', 'figure', 'text']
                for t in log_types:
                    for k, v in tmp.items():
                        if k.startswith(t + '/'):
                            log_fcn = getattr(tf.summary, t)
                            log_fcn(k[len(t + '/'):], v, step=step)

        self._file_writer.flush()

    def _log_hparams(self):
        if hasattr(self, 'trial'):
            if self.trial and self.trial.evaluated_params:
                ep = flatten_dict(self.trial.evaluated_params, '/')
                ep = {format_keys(p): v for p, v in ep.items()}
                hp.hparams(ep, trial_id=self.trial.trial_id)

    def flush(self):
        if self._file_writer is not None:
            self._file_writer.flush()

    def close(self):
        if self._file_writer is not None:
            self._file_writer.close()


class MLFLogger(Logger):
    def _init(self):
        from mlflow.tracking import MlflowClient
        uri = osp.join(osp.dirname(self.logdir), 'mlruns')
        # print(uri)
        # import ipdb
        # ipdb.set_trace()
        # raise RuntimeError
        client = MlflowClient(tracking_uri=uri)
        experiments = [e.name for e in client.list_experiments()]
        exp_name = self.config.get("mlflow_experiment", "test")
        if exp_name in experiments:
            experiment_id = client.get_experiment_by_name(exp_name)
        else:
            experiment_id = client.create_experiment(exp_name)
        run = client.create_run(experiment_id.experiment_id,
                                tags={'mlflow.runName': self.trial.trial_id})
        self._run_id = run.info.run_id

        self.client = client
        self._log_hparams()

    def on_result(self, result):
        for key, value in result.items():
            if not isinstance(value, (np.floating, float)):
                continue
            if not key.startswith('scalar/'):
                continue
            self.client.log_metric(self._run_id,
                                   key.replace('scalar/', ''),
                                   value,
                                   step=result.get(TRAINING_ITERATION))

    def close(self):
        self.client.set_terminated(self._run_id)

    def _log_hparams(self):
        if hasattr(self, 'trial'):
            if self.trial and self.trial.evaluated_params:
                ep = flatten_dict(self.trial.evaluated_params, '/')
                ep = {format_keys(p): v for p, v in ep.items()}
                for key, value in ep.items():
                    self.client.log_param(self._run_id, key, value)


def format_keys(key):
    key = re.sub('runner_config|candidate|hparams', '', key)
    return re.sub('^/|/$', '', re.sub(r'/{2,}', '/', key))
