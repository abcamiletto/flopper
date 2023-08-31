from typing import Any, List

import numpy as np
from fvcore.nn.jit_handles import get_shape


def add_flop_jit(inputs: List[Any], outputs: List[Any]):
    return np.prod(get_shape(outputs[0]))


def mul_flop_jit(inputs: List[Any], outputs: List[Any]):
    return np.prod(get_shape(outputs[0]))


def div_flop_jit(inputs: List[Any], outputs: List[Any]):
    return np.prod(get_shape(outputs[0]))


def linalg_solve_flop_jit(inputs: List[Any], outputs: List[Any]):
    A_shape = get_shape(inputs[0])
    B_shape = get_shape(inputs[1])
    n = A_shape[0]
    m = B_shape[1] if len(B_shape) > 1 else 1
    return int((2 / 3) * n**3) + (n**2 * m)


FLOPPER_OPS = {
    "aten::mul": mul_flop_jit,
    "aten::add": add_flop_jit,
    "aten::div": div_flop_jit,
    "aten::linalg_solve": linalg_solve_flop_jit,
}
