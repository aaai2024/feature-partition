__all__ = [
    "Model",
]

from typing import Optional

from torch import Tensor
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F

from .types import ViewTo1D


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(BasicBlock, self).__init__()
        padding = int((kernel_size-1)/2)
        self.layers = nn.Sequential()
        conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                         padding=padding, bias=False)
        self.layers.add_module('Conv', conv)
        self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))
        self.layers.add_module('ReLU', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)


class Model(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 10,
                 num_stages: Optional[int] = None):
        super().__init__()
        self._linear = None
        self._n_classes = n_classes

        if num_stages is None:
            num_stages = 4

        nchannels = 192
        nchannels2 = 160
        nchannels3 = 96

        # noinspection PyListCreation
        blocks = []

        # 1st block
        blocks.append(nn.Sequential())
        blocks[-1].add_module('Block1_ConvB1', BasicBlock(in_channels, nchannels, 5))
        blocks[-1].add_module('Block1_ConvB2', BasicBlock(nchannels,  nchannels2, 1))
        blocks[-1].add_module('Block1_ConvB3', BasicBlock(nchannels2, nchannels3, 1))
        if num_stages >= 2:
            avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            blocks[-1].add_module('Block1_MaxPool', avg_pool)

        if num_stages >= 2:
            # 2nd block
            blocks.append(nn.Sequential())
            blocks[-1].add_module('Block2_ConvB1',  BasicBlock(nchannels3, nchannels, 5))
            blocks[-1].add_module('Block2_ConvB2',  BasicBlock(nchannels,  nchannels, 1))
            blocks[-1].add_module('Block2_ConvB3',  BasicBlock(nchannels,  nchannels, 1))
            if num_stages > 2:
                avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
                blocks[-1].add_module('Block2_AvgPool', avg_pool)

        for i in range(3, num_stages + 1):
            blocks.append(nn.Sequential())
            blocks[-1].add_module(f'Block{i}_ConvB1',  BasicBlock(nchannels, nchannels, 3))
            blocks[-1].add_module(f'Block{i}_ConvB2',  BasicBlock(nchannels, nchannels, 1))
            blocks[-1].add_module(f'Block{i}_ConvB3',  BasicBlock(nchannels, nchannels, 1))
            if i == 3 and num_stages > 3:
                avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
                blocks[-1].add_module(f'Block{i}_AvgPool', avg_pool)

        # global average pooling and classifier
        blocks.append(nn.Sequential())
        blocks[-1].add_module('GlobalAveragePooling',  GlobalAveragePooling())

        # noinspection PyTypeChecker
        blocks.append(ViewTo1D())

        self._blocks = nn.Sequential(*blocks)
        self.eval()

    def forward(self, xs: Tensor) -> Tensor:
        out = xs

        out = self._blocks.forward(out)

        # Automatically define linear size
        if self._linear is None:
            self._linear = nn.Linear(out.shape[1], self._n_classes)
        out = self._linear.forward(out)
        return out
