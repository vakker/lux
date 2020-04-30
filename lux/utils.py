import collections
import hashlib
import json
import logging
import os
import random
import time
from os import path as osp

import numpy as np
import pandas as pd
import torch
import yaml


def get_dict_hash(d):
    return get_hash(json.dumps(d, sort_keys=True, ensure_ascii=True).encode())


def get_hash(string):
    if not isinstance(string, bytes):
        string = string.encode()
    return hashlib.sha256(string).hexdigest()


def trial_str_creator(trial):
    return "{}_{}".format(trial.trainable_name, trial.trial_id)


def update_keys(base_dict, update):
    if update is None:
        return base_dict

    tmp = base_dict.copy()
    for key in tmp:
        if key in update:
            tmp[key] = update[key]
    return tmp


def init_random(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def to_numpy(input_arr):
    if isinstance(input_arr, torch.Tensor):
        return input_arr.cpu().detach().numpy()
    elif isinstance(input_arr, (pd.DataFrame, pd.Series)):
        return input_arr.values
    elif isinstance(input_arr, np.ndarray):
        return input_arr
    elif isinstance(input_arr, (int, float)):
        return input_arr
    elif isinstance(input_arr, list):
        return np.array(input_arr)
    else:
        raise ValueError("Cannot convert %s to Numpy" % (type(input_arr)))


def read_results(trial_dir):
    trial = trial_dir.split('/')[-1]
    result_json = osp.join(trial_dir, 'result.json')
    result = []
    with open(result_json, 'r') as f:
        for row in f:
            result.append(json.loads(row))

    avg_loss = np.nanmean([r['mean_loss'] for r in result[-10:]])

    return {'name': trial, 'result': result, 'avg_loss': avg_loss}


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_dict(dict_path):
    _, ext = osp.splitext(dict_path)
    with open(dict_path, 'r') as stream:
        if ext in ['.json']:
            yaml_dict = json.load(stream)
        elif ext in ['.yml', '.yaml']:
            yaml_dict = yaml.safe_load(stream)
    return yaml_dict


def get_tune_configs(log_dir):
    assert osp.isdir(log_dir), 'Log dir does not exists.'

    if osp.exists(osp.join(log_dir, 'tune.yaml')):
        tune_configs = load_dict(osp.join(log_dir, 'tune.yaml'))

        return get_configs(log_dir), tune_configs
    else:
        raise RuntimeError('No tune configs found')


def get_configs(log_dir):
    assert osp.isdir(log_dir), 'Log dir does not exists.'

    if osp.exists(osp.join(log_dir, 'conf.yaml')):
        exp_configs = load_dict(osp.join(log_dir, 'conf.yaml'))

        ds_path = exp_configs['data_params']['dm_args']['root_dir']
        if '%p' in ds_path:
            if 'ROOT_DIR' in os.environ:
                ds_path.replace('%p', os.environ['ROOT_DIR'])
            else:
                raise RuntimeError('For %%p use ROOT env variable')
        elif '%l' in ds_path:
            ds_path.replace('%l', log_dir)
        exp_configs['data_params']['dm_args']['root_dir'] = ds_path
        return exp_configs

    if osp.exists(osp.join(log_dir, 'params.json')):
        exp_configs = load_dict(osp.join(log_dir, 'params.json'))
        if 'runner_config' in exp_configs:
            return exp_configs['runner_config']

        # if osp.exists(log_dir) and clear:
        #     shutil.rmtree(log_dir)
        # os.mkdir(log_dir)
        return exp_configs

    raise RuntimeError('No configs found')


def print_metrics(scores):
    scores_print = ['%s: %.4f,' % (metric, value) for metric, value in scores.items()]
    logging.debug(' '.join(['Val loss:'] + scores_print))
    # tqdm.write(' '.join(['Val loss:'] + scores_print))


class TicToc:
    def __init__(self):
        self.start = None
        self.lap = None

        self.tic()

    def tic(self):
        self.start = time.time()
        self.lap = self.start

    def toc(self, message=''):
        now = time.time()
        m = 'Cum: %.3f\tLap: %.3f' % (now - self.start, now - self.lap)
        if message:
            m += f', {message}'
        logging.debug(m)
        self.lap = now

    def cum(self):
        now = time.time()
        m = 'Cum: %.3f\t' % (now - self.start)
        logging.debug(m)


class TimerStat:
    """A running stat for conveniently logging the duration of a code block.

    Note that this class is *not* thread-safe.

    Examples:
        Time a call to 'time.sleep'.

        >>> import time
        >>> sleep_timer = TimerStat()
        >>> with sleep_timer:
        ...     time.sleep(1)
        >>> round(sleep_timer.mean)
        1
    """
    def __init__(self, window_size=10):
        self._window_size = window_size
        self._samples = []
        self._units_processed = []
        self._start_time = None
        self._total_time = 0.0
        self.count = 0

    def __enter__(self):
        assert self._start_time is None, "concurrent updates not supported"
        self._start_time = time.time()

    def __exit__(self, type, value, tb):
        assert self._start_time is not None
        time_delta = time.time() - self._start_time
        self.push(time_delta)
        self._start_time = None

    def push(self, time_delta):
        self._samples.append(time_delta)
        if len(self._samples) > self._window_size:
            self._samples.pop(0)
        self.count += 1
        self._total_time += time_delta

    def push_units_processed(self, n):
        self._units_processed.append(n)
        if len(self._units_processed) > self._window_size:
            self._units_processed.pop(0)

    @property
    def mean(self):
        if self._samples:
            return np.mean(self._samples)
        return None

    @property
    def median(self):
        if self._samples:
            return np.median(self._samples)
        return None

    @property
    def sum(self):
        return np.sum(self._samples)

    @property
    def max(self):
        return np.max(self._samples)

    @property
    def first(self):
        return self._samples[0] if self._samples else None

    @property
    def last(self):
        return self._samples[-1] if self._samples else None

    @property
    def size(self):
        return len(self._samples)

    @property
    def mean_units_processed(self):
        return float(np.mean(self._units_processed))

    @property
    def mean_throughput(self):
        time_total = sum(self._samples)
        if not time_total:
            return 0.0
        return sum(self._units_processed) / time_total

    def reset(self):
        self._samples = []
        self._units_processed = []
        self._start_time = None
        self._total_time = 0.0
        self.count = 0
