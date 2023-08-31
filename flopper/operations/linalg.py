from typing import Any, List

import numpy as np
from fvcore.nn.jit_handles import get_shape


def linalg_solve_flop_jit(inputs: List[Any], outputs: List[Any]):
    A_shape = get_shape(inputs[0])
    B_shape = get_shape(inputs[1])

    n = A_shape[-2]
    m = B_shape[-1]

    base_flops = int((2 / 3) * n**3) + (n**2 * m)

    # Calculate broadcasted dimensions
    broadcast_dims = [max(a, b) for a, b in zip(A_shape[:-2], B_shape[:-2])]
    total_instances = np.prod(broadcast_dims)

    return base_flops * total_instances


LINALG_OPS = {
    "aten::linalg_solve": linalg_solve_flop_jit,
}
