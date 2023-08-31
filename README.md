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

flops = count_flops(model, batch)
```

To get more detailed information, you can do the following:

```python
print(flops.by_operator())
print(flops.by_module())
print(flops.by_module_and_operator())
print(flops.get_table())
```

## Adding support for custom new modules

Work in progress
