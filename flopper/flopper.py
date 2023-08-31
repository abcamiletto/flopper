from functools import partial

import fvcore
import numpy as np
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import get_shape

from .custom_ops import FLOPPER_OPS


def count_flops(model, *args, silent=False, **kwargs):
    """Count the number of FLOPs in a model.

    Args:
        model: The model to count FLOPs for.
        args: Arguments to pass to the model.
        kwargs: Keyword arguments to pass to the model.
    """
    if kwargs is not None:
        model.forward = partial(model.forward, **kwargs)

    flops = FlopCountAnalysis(model, args, kwargs).set_op_handle(**FLOPPER_OPS)

    if not silent:
        gflops = flops.total() / 1e9
        print(f"FLOPs: {gflops:.2f} GFLOPs")
