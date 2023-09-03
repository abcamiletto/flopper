import numpy as np
from fvcore.nn.jit_handles import get_shape


def mean_flop_jit(inputs, outputs):
    input_shape = get_shape(inputs[0])
    return np.prod(input_shape)


def sum_flop_jit(inputs, outputs):
    input_shape = get_shape(inputs[0])
    return np.prod(input_shape)


COMMON_OPS = {
    "aten::mean": mean_flop_jit,
    "aten::sum": sum_flop_jit,
}
