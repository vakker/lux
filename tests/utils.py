import numpy as np
import torch
from torch import nn

from pytorch_ray import PyTorchRunner


def create_runner(config):
    return SimpleRunner(config)


class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


class LinearDataset(torch.utils.data.Dataset):
    """y = x * a + b"""

    def __init__(self, a, b, size=1000):
        in_feats = a.shape[0]
        x = np.linspace(0, 1, size)
        x = np.tile(x, (in_feats, 1)).T.astype(np.float32)
        y = (np.dot(x, a) + b).astype(np.float32)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x[index, None], self.y[index, None]

    def __len__(self):
        return len(self.x)


class SimpleRunner(PyTorchRunner):
    def model_creator(self, config):
        model_params = config['model_params']
        return TwoLayerNet(model_params['D_in'], model_params['H'],
                           model_params['D_out'])

    def data_creator(self, config):
        ds_params = config['ds_params']
        tng_ds = LinearDataset(ds_params['a'], ds_params['b'], size=700)
        val_ds = LinearDataset(ds_params['a'], ds_params['b'], size=300)
        return tng_ds, val_ds

    def optimizer_creator(self, model, config):
        optim_params = config['optim_params']
        criterion = getattr(torch.nn, config['loss'])()
        optimizer_cls = getattr(torch.optim, config['optim'])
        optimizer = optimizer_cls(model.parameters(), **optim_params)
        return criterion, optimizer
