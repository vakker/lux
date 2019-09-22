from os import path as osp

import pytest
from ray import tune

from pytorch_ray import PyTorchTrainable
from pytorch_ray.logger import pr_logger_creator

from .utils import create_runner, get_chkp_dir, trial_str_creator


def test_trainable0(trainable):
    assert trainable.epoch == 0
    assert trainable._runner.epoch == 0
    assert trainable._iteration == 0

    stats0 = trainable.train()
    assert trainable.epoch == 1
    assert trainable._runner.epoch == 1
    assert trainable._iteration == 1

    stats1 = trainable.train()
    assert trainable.epoch == 2
    assert trainable._runner.epoch == 2
    assert trainable._iteration == 2

    assert stats1['scalar/tng/loss'] < stats0['scalar/tng/loss']


def test_trainable_save_restore0(trainable):
    assert trainable.epoch == 0
    assert trainable._runner.epoch == 0
    assert trainable._iteration == 0

    stats0 = trainable.train()
    assert trainable.epoch == 1
    assert trainable._runner.epoch == 1
    assert trainable._iteration == 1

    chkp_path = trainable.save()
    assert get_chkp_dir(chkp_path) == 'checkpoint_1'
    trainable.restore(chkp_path)
    stats1 = trainable.val()
    assert trainable.epoch == 1
    assert trainable._runner.epoch == 1
    assert trainable._iteration == 1

    assert stats1['scalar/val/loss'] == stats0['scalar/val/loss']


def test_trainable_save_restore1(trainable_config):
    logdir = trainable_config['logdir']

    trainable0 = PyTorchTrainable(trainable_config)

    stats0 = trainable0.train()
    stats0 = trainable0.train()
    stats0 = trainable0.train()
    assert trainable0.epoch == 3
    assert trainable0._runner.epoch == 3
    assert trainable0._iteration == 3

    chkp_path = trainable0.save()
    assert get_chkp_dir(chkp_path) == 'checkpoint_3'

    trainable_config['restore_from'] = chkp_path
    trainable1 = PyTorchTrainable(trainable_config)
    stats1 = trainable1.val()
    assert trainable1.epoch == 3
    assert trainable1._runner.epoch == 3
    assert trainable1._iteration == 3

    assert stats1['scalar/val/loss'] == stats0['scalar/val/loss']


@pytest.mark.parametrize("val_freq", [3, 5])
def test_trainable_val_freq(trainable, val_freq):
    max_epochs = trainable.config['max_epochs']
    trainable.config['val_freq'] = val_freq

    for i in range(1, max_epochs + 1):
        stats = trainable.train()
        assert 'scalar/tng/loss' in stats
        if i == 1 or i % val_freq == 0 or i == max_epochs:
            assert 'scalar/val/loss' in stats
        else:
            assert 'scalar/val/loss' not in stats
