from os import path as osp

import numpy as np
import torch
from torch import nn

from lux import LuxRunner


def get_chkp_dir(chkp_path):
    return osp.basename(osp.dirname(chkp_path))


def trial_str_creator(trial):
    return "{}_{}".format(trial.trainable_name, trial.trial_id)


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
        np.random.seed(size)
        a = np.array(a)
        b = np.array(b)
        in_feats = a.shape[0]
        x = np.linspace(0, 1, size)
        x = np.tile(x, (in_feats, 1)).T
        y = (np.dot(x, a) + b)
        y = y + np.random.normal(size=y.shape)
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __getitem__(self, index):
        return self.x[index, None], self.y[index, None]

    def __len__(self):
        return len(self.x)


class SimpleRunner(LuxRunner):
    def model_creator(self, config):
        hparams = config['hparams']
        return TwoLayerNet(hparams['D_in'], hparams['H'], hparams['D_out'])

    def data_creator(self, config):
        ds_params = config['ds_params']
        tng_ds = LinearDataset(ds_params['a'], ds_params['b'], size=700)
        val_ds = LinearDataset(ds_params['a'], ds_params['b'], size=300)
        return tng_ds, val_ds

    def optimizer_creator(self, model, config):
        hparams = config['hparams']
        criterion = getattr(torch.nn, config['loss'])()
        optimizer_cls = getattr(torch.optim, config['optim'])
        optimizer = optimizer_cls(model.parameters(), hparams['lr'])
        return criterion, optimizer

    def tng_step(self, samples):
        inputs = samples[0]
        targets = samples[1]

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        return loss, {'loss': loss}

    def val_step(self, samples):
        inputs = samples[0]
        targets = samples[1]

        outputs = self.model(inputs)
        errors = targets - outputs
        loss = self.criterion(outputs, targets)
        return loss, {'loss': loss, 'errors': errors}

    def post_tng_step(self, batch_outputs):
        scalar = {
            'tng/loss': np.mean([ba['loss'] for ba in batch_outputs]),
        }
        return {'scalar': scalar}

    def post_val_step(self, batch_outputs):
        scalar = {
            'val/loss': np.mean([ba['loss'] for ba in batch_outputs]),
        }

        histogram = {'val/errors': np.concatenate([ba['errors'] for ba in batch_outputs], axis=0)}
        return {'scalar': scalar, 'histogram': histogram}
