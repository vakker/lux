import numpy as np
import pytest
import ray

from pytorch_ray import PyTorchTrainable

from .utils import create_runner

D_in = 200
D_out = 5


def get_runner_config():
    return {
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


@pytest.fixture(scope="module")
def start_ray():
    ray.init()


@pytest.fixture
def runner_config():
    return get_runner_config()


@pytest.fixture(
    scope="function",
    params=[
        'runner-cpu',
        'trainable-cpu',
        'runner-gpu',
        'trainable-gpu',
    ])
def simple_module(request):
    s_runner_config = get_runner_config()
    config = {}
    if request.param.endswith('gpu'):
        config['num_gpus'] = 1
    else:
        config['num_gpus'] = 0

    if request.param.startswith('runner'):
        config.update(s_runner_config)
        sr = create_runner(config)
        return sr
    config['runner_creator'] = create_runner
    config.update({'runner_config': s_runner_config})
    st = PyTorchTrainable(config)
    return st
