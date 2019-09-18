from os import path as osp

from ray.tune.logger import Logger
from ray.tune.result import TIMESTEPS_TOTAL, TRAINING_ITERATION
from torch.utils.tensorboard import SummaryWriter


def pr_logger_creator(config):
    return PRLogger(config, config['logdir'])


class PRLogger(Logger):
    def _init(self):
        purge_step = self.config.get('start_from', 0)

        self._tb_writer = SummaryWriter(
            log_dir=osp.join(self.logdir, 'logs'), purge_step=purge_step)

        self._log_hparams()

    def on_result(self, result):
        pass

    def log(self, result, step):
        tmp = result.copy()

        log_types = ['scalar', 'histogram', 'image', 'figure', 'text']
        for t in log_types:
            for k, v in tmp.get(t, {}).items():
                log_fcn = getattr(self._tb_writer, f'add_{t}')
                log_fcn(k, v, global_step=step)

        self._tb_writer.flush()

    def _log_hparams(self):
        if self.trial and self.trial.evaluated_params:
            self._tb_writer.add_hparams(
                hparam_dict=self.trial.evaluated_params)

    def add_graph(self, model, inputs):
        self._tb_writer.add_graph(model, inputs)
