import pytest
import torch
import torch.nn as nn
from torch import Tensor


class KwargsNet(nn.Module):
    """A network to test kwargs in flopper."""

    def __init__(self) -> None:
        super().__init__()
        input_dim, conv_dim, linear_dim, out_dim = 16, 8, 4, 1

        self.conv = nn.Conv2d(input_dim, conv_dim, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((out_dim, out_dim))
        self.linear1 = nn.Linear(conv_dim, linear_dim)
        self.linear2 = nn.Linear(linear_dim, 1)

    def forward(self, x: Tensor, early_return: bool = False) -> Tensor:
        x = self.conv(x)

        if early_return:
            return x.mean()

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


@pytest.fixture
def model():
    return KwargsNet()


@pytest.fixture
def input_batch():
    return torch.randn(4, 16, 32, 32)
