from functools import partial

import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_table

from .operations import FLOPPER_OPS


def count_flops(model: nn.Module, *args, silent: bool = False, custom_ops=None, **kwargs) -> int:
    """Count the number of FLOPs in a model.

    Args:
        model: The model to count FLOPs for.
        args: Arguments to pass to the model.
        silent: Whether to print the number of FLOPs or only return it.
        custom_ops: Custom operations to add to the FLOP count.
        kwargs: Keyword arguments to pass to the model.

    Returns:
        The number of FLOPs in the model.

    Usage:
        >>> import torch
        >>> from flopper import count_flops
        >>> model = torch.nn.Linear(10, 10)
        >>> flops = count_flops(model, torch.randn(1, 10))
        FLOPs of the model Linear: X.XX GFLOPs

    Get more specific information about the FLOPs with the following methods:
        >>> flops.by_operator()
        >>> flops.by_module()
        >>> flops.by_module_and_operator()
        >>> flops.get_table()
    """
    if kwargs is not None:
        model.forward = partial(model.forward, **kwargs)

    if custom_ops is not None:
        FLOPPER_OPS.update(custom_ops)

    flops = FlopCountAnalysis(model, args).set_op_handle(**FLOPPER_OPS)

    if not silent:
        flops_str = smart_format(flops.total())
        name = model.__class__.__name__
        print(f"FLOPs of the model {name}: {flops_str}FLOPs")

    flops.get_table = lambda: flop_count_table(flops)

    return flops


def smart_format(num, decimals=2):
    if num > 1e12:
        return f"{num / 1e12:.{decimals}f} T"
    elif num > 1e9:
        return f"{num / 1e9:.{decimals}f} G"
    elif num > 1e6:
        return f"{num / 1e6:.{decimals}f} M"
    elif num > 1e3:
        return f"{num / 1e3:.{decimals}f} K"
    else:
        return f"{num:.{decimals}f }"
