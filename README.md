# Flopper - A FLOP counter for PyTorch

An FLOP counter based on fvcore with a more extensive support for any (we're trying) PyTorch modules.
This tool is a lightweight wrapper around [fvcore](https://github.com/facebookresearch/fvcore) flop counter, which does all the work under the hood. We provide an easy to use API to count the number of FLOPs of any PyTorch model.

It's going to be a bit slower than fvcore, but more accurate.

## Installation

```bash
pip install flopper
```

## Usage

The simplest way to use flopper is to use the `count_flops` function. It takes a model and an input batch as input and prints the total number of FLOPs.

```python
from flopper import count_flops

model = YourRandomModel()
batch = torch.randn(1, 3, 224, 224)

flops = count_flops(model, batch) # This will print the total number of FLOPs

n_flops = flops.total()
```

To get more detailed information, you can do the following:

```python
print(flops.by_operator())
print(flops.by_module())
print(flops.by_module_and_operator())
print(flops.get_table())
```

Out API supports also the usage of keyword arguments in the model's forward function. Let's look at an example:

```python
input_1, input_2 = ...
mode = "advanced"

flops = count_flops(model, input_1, input_2, mode=mode)
```

## Adding support for custom new modules

If you want to add support for a new module, you can do so by creating a dictionary with the following structure:

```python
import numpy as np
from fvcore.nn.jit_handles import get_shape
from flopper import count_flops

model = YourRandomModel()
batch = torch.randn(1, 3, 224, 224)

def mean_flop_jit(inputs, outputs):
    input_shape = get_shape(inputs[0])
    return np.prod(input_shape)

custom_ops = {"aten::mean": mean_flop_jit}
flops = count_flops(model, batch, custom_ops=custom_ops)
```
