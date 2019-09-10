import pytest
from ray import tune

from pytorch_ray import PyTorchTrainable

from .utils import create_runner


def test_sr(simple_module):
    assert simple_module.epoch == 0

    stats0 = simple_module.train()
    assert simple_module.epoch == 1

    stats1 = simple_module.train()
    assert simple_module.epoch == 2

    assert stats1['tng_loss'] < stats0['tng_loss']
    assert stats1['val_loss'] < stats0['val_loss']


@pytest.mark.parametrize("num_gpus", [0, 1])
def test_tune_train(start_ray, runner_config, num_gpus):  # noqa: F811

    config = {
        "runner_creator": tune.function(create_runner),
        "num_gpus": num_gpus,
        "num_cpus": 2,
    }
    config.update(runner_config)

    analysis = tune.run(
        PyTorchTrainable,
        num_samples=2,
        config=config,
        stop={"training_iteration": 2},
        verbose=1)

    # checks loss decreasing for every trials
    for path, df in analysis.trial_dataframes.items():
        tng_loss1 = df.loc[0, "tng_loss"]
        tng_loss2 = df.loc[1, "tng_loss"]
        val_loss1 = df.loc[0, "val_loss"]
        val_loss2 = df.loc[1, "val_loss"]

        assert tng_loss2 < tng_loss1
        assert val_loss2 < val_loss1
