import numpy as np
import pytest

import utils
from pytorch_ray import PyTorchTrainable

D_in = 200
D_out = 5
runner_config = {
    'model_params': {
        'D_in': D_in,
        'H': 100,
        'D_out': D_out
    },
    'ds_params': {
        'a': np.random.random((D_in, D_out)),
        'b': np.random.random((D_out))
    },
    'optim_params': {
        'lr': 1e-4
    },
    'loss': 'MSELoss',
    'optim': 'SGD'
}


@pytest.fixture(
    scope="function",
    params=['runner-cpu', 'trainable-cpu', 'runner-gpu', 'trainable-gpu'])
def simple_module(request):
    config = runner_config
    if request.param.endswith('gpu'):
        config['num_gpus'] = 1
    else:
        config['num_gpus'] = 0

    if request.param.startswith('runner'):
        sr = utils.create_runner(config)
        return sr
    config['runner_creator'] = utils.create_runner
    st = PyTorchTrainable(config)
    return st
