# Flopper - A FLOP counter for PyTorch

An enhanced FLOP counter based on fvcore with additional support to custom modules and simplified API
This tool is a lightweight wrapper around [fvcore](https://github.com/facebookresearch/fvcore) flop counter, which is a PyTorch based FLOP counter. It supports counting FLOPs for additional modules and provides a simplified API, which makes it easier to use.

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
