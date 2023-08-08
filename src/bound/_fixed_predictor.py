__all__ = [
    "MeanRegressor",
    "MedianRegressor",
]

import abc
from typing import NoReturn

import numpy as np
from sklearn.base import RegressorMixin

import torch


class _BaseFixedRegressor(RegressorMixin, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self._fixed_val = None

    def fit(self, X, y) -> NoReturn:
        r""" Calculates a fixed statistic """
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        self._fixed_val = self._calc_fixed(y=y)

    @abc.abstractmethod
    def _calc_fixed(self, y) -> NoReturn:
        r""" Calculates a fixed value to return """

    def predict(self, x) -> NoReturn:
        r""" Calculates a fixed value """
        n_ele = x.shape[0]
        vals = torch.full([n_ele], self._fixed_val)
        return vals.numpy()


class MeanRegressor(_BaseFixedRegressor):
    r""" Just predicts the mean """
    def _calc_fixed(self, y) -> NoReturn:
        return y.mean().item()


class MedianRegressor(_BaseFixedRegressor):
    r""" Just predicts the median """
    def _calc_fixed(self, y) -> NoReturn:
        return torch.quantile(y, 0.5).item()
