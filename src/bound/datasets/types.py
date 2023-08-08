__all__ = [
    "BatchNorm",
    "GhostBatchNorm",
    "SplitDataset",
    "ViewModule",
    "ViewTo1D",
]

import collections
from enum import Enum
from typing import List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F  # noqa

DatasetParams = collections.namedtuple("DatasetParams", "name dim n_classes")
CIFAR_DIM = [3, 32, 32]
MNIST_DIM = [1, 28, 28]
WEATHER_DIM = []


# noinspection PyPep8Naming
class SplitDataset(Enum):
    r""" Valid datasets for testing """
    AMES = DatasetParams("Ames-Housing", [352], -1)

    CIFAR10 = DatasetParams("CIFAR10", CIFAR_DIM, 10)

    MNIST = DatasetParams("MNIST", MNIST_DIM, 10)

    WEATHER = DatasetParams("Shifts-Weather", WEATHER_DIM, -1)

    def is_cifar(self) -> bool:
        r""" Returns \p True if dataset is a CIFAR dataset """
        cifar_ds = [
            self.CIFAR10,
        ]
        return self in cifar_ds

    def is_mnist(self) -> bool:
        r""" Returns \p True if dataset is a MNIST or MNIST-variant dataset """
        mnist_ds = [
            self.MNIST,
        ]
        return self in mnist_ds

    def is_weather(self) -> bool:
        weather_ds = [
            self.WEATHER
        ]
        return self in weather_ds

    def is_ames(self) -> bool:
        return self == self.AMES

    def get_n_classes(self) -> int:
        r""" Return the number of classes for the dataset """
        n_classes = self.value.n_classes
        assert n_classes > 1, "At least two classes expected"
        return n_classes

    def is_tabular(self) -> bool:
        r""" Return \p True if using a tabular dataset """
        tabular_ds = [
            self.AMES,
            self.WEATHER,
        ]
        return self in tabular_ds

    def is_expm1_scale(self) -> bool:
        r""" Returns \p True if the dataset requires scaling y by expm1 """
        expm1_ds = [
            self.AMES,
        ]
        return self in expm1_ds


class ViewModule(nn.Module):
    r""" General view layer to flatten to any output dimension """
    def __init__(self, d_out: List[int]):
        super().__init__()
        self._d_out = tuple(d_out)

    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        # noinspection PyUnresolvedReferences
        return x.reshape((x.shape[0], *self._d_out))


class ViewTo1D(ViewModule):
    r""" View layer simplifying to specifically a single dimension """
    def __init__(self):
        super().__init__([-1])


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features*self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features*self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C*self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)
