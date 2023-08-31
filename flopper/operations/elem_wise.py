from typing import Any, List

import numpy as np
from fvcore.nn.jit_handles import get_shape


def add_flop_jit(inputs: List[Any], outputs: List[Any]):
    return np.prod(get_shape(outputs[0]))


def mul_flop_jit(inputs: List[Any], outputs: List[Any]):
    return np.prod(get_shape(outputs[0]))


def div_flop_jit(inputs: List[Any], outputs: List[Any]):
    return np.prod(get_shape(outputs[0]))


ELEM_WISE_OPS = {
    "aten::mul": mul_flop_jit,
    "aten::mul_": mul_flop_jit,
    "aten::add": add_flop_jit,
    "aten::add_": add_flop_jit,
    "aten::div": div_flop_jit,
    "aten::div_": div_flop_jit,
    "aten::sub": add_flop_jit,
    "aten::sub_": add_flop_jit,
    "aten::rsub": add_flop_jit,
    "aten::rsub_": add_flop_jit,
}
