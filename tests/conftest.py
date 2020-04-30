import tempfile

import numpy as np
import pytest
import ray

from lux import LuxTrainable

from .utils import create_runner

D_in = 200
D_out = 5


def get_trainable_config():
    runner_config = {
        'ds_params': {
            'a': np.random.random((D_in, D_out)).tolist(),
            'b': np.random.random((D_out)).tolist()
        },
        'loss': 'MSELoss',
        'optim': 'SGD',
        'hparams': {
            'D_in': D_in,
            'H': 100,
            'D_out': D_out,
            'lr': 1e-4,
            'momentum': 0.9
        }
    }

    config = {'logdir': tempfile.mkdtemp()}
    config['num_gpus'] = 1
    config['val_freq'] = 1
    config['max_epochs'] = 10

    config['runner_creator'] = create_runner
    config.update({'runner_config': runner_config})
    return config


@pytest.fixture(scope="module")
def start_ray():
    ray.init()


@pytest.fixture
def trainable_config():
    return get_trainable_config()


@pytest.fixture(scope="function", params=[
    'cpu',
    'gpu',
])
def trainable(request):
    config = get_trainable_config()
    config['num_gpus'] = 1 if request.param == 'gpu' else 0
    st = LuxTrainable(config)
    return st
