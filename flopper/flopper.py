from functools import partial
from typing import Any, Dict, List

import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

from .custom_ops import FLOPPER_OPS


def count_flops(model: nn.Module, *args, silent: bool = False, **kwargs) -> int:
    """Count the number of FLOPs in a model.

    Args:
        model: The model to count FLOPs for.
        args: Arguments to pass to the model.
        silent: Whether to print the number of FLOPs or only return it.
        kwargs: Keyword arguments to pass to the model.

    Returns:
        The number of FLOPs in the model.

    Usage:
        >>> import torch
        >>> from flopper import count_flops
        >>> model = torch.nn.Linear(10, 10)
        >>> count_flops(model, torch.randn(1, 10))
        FLOPs of the model Linear: 0.00 GFLOPs
    """
    if kwargs is not None:
        model.forward = partial(model.forward, **kwargs)  # To be tested
        raise NotImplementedError("Keyword arguments are not supported yet.")

    flops = FlopCountAnalysis(model, args).set_op_handle(**FLOPPER_OPS)

    if not silent:
        gflops = flops.total() / 1e9
        print(f"FLOPs of the model {model.__class__.__name__}: {gflops:.2f} GFLOPs")

    return flops.total()
