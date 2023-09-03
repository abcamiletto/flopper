import pytest

from flopper import count_flops


def test_kwargs(model, input_batch):
    flops = count_flops(model, input_batch).total()
    flops_kwargs = count_flops(model, input_batch, early_return=True).total()
    assert flops_kwargs < flops
    
def test_table(model, input_batch):
    flops = count_flops(model, input_batch)
    table = flops.get_table()
    assert table is not None