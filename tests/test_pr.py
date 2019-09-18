import pytest
from ray import tune

from pytorch_ray import PyTorchTrainable

from .utils import create_runner, trial_str_creator


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

    assert stats1['tng/loss'] < stats0['tng/loss']


def test_trainable_save_restore0(trainable):
    assert trainable.epoch == 0
    assert trainable._runner.epoch == 0
    assert trainable._iteration == 0

    stats0 = trainable.train()
    assert trainable.epoch == 1
    assert trainable._runner.epoch == 1
    assert trainable._iteration == 1

    chkp_path = trainable.save()
    trainable.restore(chkp_path)
    stats1 = trainable.val()
    assert trainable.epoch == 1
    assert trainable._runner.epoch == 1
    assert trainable._iteration == 1

    assert stats1['val/loss'] == stats0['val/loss']


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

    trainable_config['restore_from'] = chkp_path
    trainable1 = PyTorchTrainable(trainable_config)
    stats1 = trainable1.val()
    assert trainable1.epoch == 3
    assert trainable1._runner.epoch == 3
    assert trainable1._iteration == 3

    assert stats1['val/loss'] == stats0['val/loss']


@pytest.mark.parametrize("val_freq", [3, 5])
def test_trainable_val_freq(trainable, val_freq):
    max_epochs = trainable.config['max_epochs']
    trainable.config['val_freq'] = val_freq

    for i in range(1, max_epochs + 1):
        stats = trainable.train()
        assert 'tng/loss' in stats
        if i == 1 or i % val_freq == 0 or i == max_epochs:
            assert 'val/loss' in stats
        else:
            assert 'val/loss' not in stats


@pytest.mark.parametrize("num_gpus", [0, 1])
def test_tune_train(start_ray, trainable_config, num_gpus):

    trainable_config.update({
        "num_gpus": num_gpus,
        "num_cpus": 2,
    })

    trainable_config['runner_config']['hparams']['lr'] = tune.grid_search(
        [1e-5, 1e-4])
    trainable_config['runner_config']['hparams'][
        'momentum'] = tune.grid_search([0, 0.9])

    analysis = tune.run(
        PyTorchTrainable,
        num_samples=1,
        config=trainable_config,
        trial_name_creator=trial_str_creator,
        stop={"training_iteration": 5},
        local_dir='./logs',
        verbose=1)

    # checks loss decreasing for every trials
    for path, df in analysis.trial_dataframes.items():
        tng_loss1 = df.loc[0, "tng/loss"]
        tng_loss2 = df.loc[1, "tng/loss"]

        assert tng_loss2 < tng_loss1
