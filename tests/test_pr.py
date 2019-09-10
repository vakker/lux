def test_sr(simple_module):
    assert simple_module.epoch == 0

    stats0 = simple_module.train()
    assert simple_module.epoch == 1

    stats1 = simple_module.train()
    assert simple_module.epoch == 2

    assert stats1['tng_loss'] < stats0['tng_loss']
    assert stats1['val_loss'] < stats0['val_loss']
