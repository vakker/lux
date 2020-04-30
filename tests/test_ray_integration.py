import pytest
from ray import tune

from lux import LuxTrainable
from lux.logger import mlf_logger_creator, tb_logger_creator

from .utils import trial_str_creator


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

    analysis = tune.run(LuxTrainable,
                        loggers=[tb_logger_creator, mlf_logger_creator],
                        num_samples=1,
                        config=trainable_config,
                        trial_name_creator=trial_str_creator,
                        stop={"training_iteration": 5},
                        checkpoint_score_attr='min-scalar/val/loss',
                        keep_checkpoints_num=2,
                        local_dir='./logs',
                        verbose=1)

    # checks loss decreasing for every trials
    for path, df in analysis.trial_dataframes.items():
        tng_loss1 = df.loc[0, "scalar/tng/loss"]
        tng_loss2 = df.loc[1, "scalar/tng/loss"]

        assert tng_loss2 < tng_loss1
