__all__ = [
    "ResNet9"
]

from typing import Optional

from torch import Tensor
import torch.nn as nn

from .. import _config as config  # noqa
from .types import BatchNorm, GhostBatchNorm  # noqa


def conv_block(in_channels, out_channels, pool=False, p_dropout: float = 0):
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                       padding=(1, 1), bias=False)
    layers = [
        conv2d,
        GhostBatchNorm(out_channels, momentum=0.1, num_splits=config.BATCH_SIZE // 32),
    ]
    if p_dropout > 0:
        layers.append(nn.Dropout2d(p=p_dropout))
    if pool:
        layers.append(nn.MaxPool2d(2))
    layers.append(nn.CELU(alpha=0.075, inplace=True))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, dim: int, p_dropout: float = 0):
        super().__init__()
        self.res1 = nn.Sequential(
            conv_block(dim, dim, p_dropout=p_dropout),
            conv_block(dim, dim, p_dropout=p_dropout),
        )

    def forward(self, xb):
        return self.res1(xb) + xb


class ResNet9(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, p_dropout: float = 0,
                 scale: Optional[float] = None):
        assert in_channels > 0, "Channel count must be positive"

        super().__init__()
        self._n_classes = n_classes
        self._scale = scale
        div = 2

        self.conv1 = conv_block(in_channels, 128 // div)
        self.conv2 = conv_block(128 // div, 256 // div, pool=True, p_dropout=p_dropout)
        self.res1 = ResBlock(256 // div)

        self.conv3 = conv_block(256 // div, 512 // div, pool=True, p_dropout=p_dropout)
        self.conv4 = conv_block(512 // div, 1024 // div, pool=True, p_dropout=p_dropout)
        self.res2 = ResBlock(1024 // div, p_dropout=p_dropout)

        self.max_pool = nn.MaxPool2d(4)
        self.flatten = nn.Flatten()

        # Only use to get _blocks only parameters
        self._blocks = nn.Sequential(self.conv1,
                                     self.conv2,
                                     self.res1,
                                     self.conv3,
                                     self.conv4,
                                     self.res2,
                                     self.max_pool,
                                     )

        self.linear = None
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        r""" Initializes linear layer in the first pass """
        assert x.shape[1] == 3, "Expect three channel images"
        out = self._blocks.forward(x)
        out = self.flatten(out)
        if self.linear is None:
            in_dim = out.shape[1]
            self.linear = nn.Linear(in_features=in_dim, out_features=self._n_classes,
                                    bias=False)
        out = self.linear(out)
        if self._scale is not None:
            out *= self._scale
        return out
