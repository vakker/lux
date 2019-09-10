import pytest
from ray import tune

from pytorch_ray import PyTorchTrainable

from .utils import create_runner, trial_str_creator


def test_sr(simple_module):
    assert simple_module.epoch == 0

    stats0 = simple_module.train()
    assert simple_module.epoch == 1

    stats1 = simple_module.train()
    assert simple_module.epoch == 2

    assert stats1['tng_loss'] < stats0['tng_loss']


@pytest.mark.parametrize("num_gpus", [0, 1])
def test_tune_train(start_ray, runner_config, num_gpus):

    config = {
        "runner_creator": tune.function(create_runner),
        "num_gpus": num_gpus,
        "num_cpus": 2,
    }
    config.update({'runner_config': runner_config})
    config['runner_config']['hparams']['lr'] = tune.grid_search(
        [1e-5, 1e-4, 1e-3, 1e-2])

    analysis = tune.run(
        PyTorchTrainable,
        num_samples=1,
        config=config,
        trial_name_creator=tune.function(trial_str_creator),
        stop={"training_iteration": 20},
        local_dir='./logs',
        verbose=1)

    # checks loss decreasing for every trials
    for path, df in analysis.trial_dataframes.items():
        tng_loss1 = df.loc[0, "tng_loss"]
        tng_loss2 = df.loc[1, "tng_loss"]

        assert tng_loss2 < tng_loss1
