# Flopper - A FLOP counter for PyTorch

An enhanced FLOP counter based on fvcore with additional support to custom modules and simplified API
This tool is a lightweight wrapper around [fvcore](https://github.com/facebookresearch/fvcore) flop counter, which is a PyTorch based FLOP counter. It supports counting FLOPs for custom modules and provides a simplified API.

## Installation

```bash
pip install flopper
```

## Usage

```python
from flopper import count_flops

model = YourRandomModel()
batch = torch.randn(1, 3, 224, 224)

flops = count_flops(model, batch)
```
