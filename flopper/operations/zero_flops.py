def zero_fn(inputs, outputs):
    return 0


ZERO_FLOP_OPS = {
    "aten::zeros": zero_fn,
    "aten::zeros_like": zero_fn,
    "aten::eye": zero_fn,
    "aten::ones": zero_fn,
    "aten::ones_like": zero_fn,
    "aten::ceil": zero_fn,
    "aten::floor": zero_fn,
    "aten::round": zero_fn,
    "aten::lt": zero_fn,
    "aten::gt": zero_fn,
    "aten::le": zero_fn,
    "aten::ge": zero_fn,
    "aten::eq": zero_fn,
    "aten::ne": zero_fn,
    "aten::all": zero_fn,
}
