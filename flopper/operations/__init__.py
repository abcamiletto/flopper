from .common import COMMON_OPS
from .elem_wise import ELEM_WISE_OPS
from .linalg import LINALG_OPS
from .zero_flops import ZERO_FLOP_OPS

FLOPPER_OPS = {**ELEM_WISE_OPS, **LINALG_OPS, **ZERO_FLOP_OPS, **COMMON_OPS}
