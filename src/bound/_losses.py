__all__ = [
    "Loss",
]

from typing import Callable, Optional

import torch
from torch import Tensor
import torch.nn as nn

from . import _config as config

_loss_module = None


def _loss(inputs: Tensor, targets: Tensor) -> Tensor:
    r""" Cross-entropy loss that takes two arguments instead of default one """
    global _loss_module
    if _loss_module is None:
        if config.IS_CLASSIFICATION:
            label_smooth = 0.2 if config.DATASET.is_cifar() else 0
            _loss_module = nn.CrossEntropyLoss(label_smoothing=label_smooth)
        else:
            raise ValueError("Unknown dataset.  Cannot set cross entropy loss weights.")

    return _loss_module.forward(inputs, targets)


def _valid_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    r""" Calculate validation loss as 1 - 0-1 loss """
    output = inputs.argmax(dim=1)
    assert output.shape == targets.shape, "Size mismatch between inputs and targets"
    mask = output == targets
    assert mask.numel() == output.numel(), "Size mismatch after mask check"
    acc = mask.float().mean()
    return 1 - acc


class Loss:
    def __init__(self, train_loss: Optional[Callable] = None,
                 valid_loss: Optional[Callable] = None, name_suffix: str = ""):
        if train_loss is None:
            assert valid_loss is None, "Validation loss cannot be specified without training loss"
            train_loss = _loss

        if valid_loss is None:
            valid_loss = _valid_loss

        self.tr_loss = train_loss
        self.val_loss = valid_loss

        self._name_suffix = name_suffix

    def name(self) -> str:
        r""" Name of the risk estimator """
        flds = [f"L", self._name_suffix]
        return "-".join(flds)

    @staticmethod
    def _loss(dec_scores: Tensor, lbls: Tensor, f_loss: Callable, **kwargs) -> Tensor:
        r""" Straight forward PN loss -- No weighting by prior & label """
        is_binary = config.DATASET.is_cifar() or config.DATASET.is_mnist()
        assert is_binary or len(dec_scores.shape) == 2, "Bizarre input shape"
        assert dec_scores.shape[0] == lbls.shape[0], "Vector shape loss mismatch"

        lst_loss = f_loss(dec_scores, lbls)
        return lst_loss.mean()

    def calc_train_loss(self, dec_scores: Tensor, labels: Tensor, **kwargs) -> Tensor:
        r""" Calculates the risk using the TRAINING specific loss function """
        return self._loss(dec_scores=dec_scores, lbls=labels, f_loss=self.tr_loss, **kwargs)

    def calc_validation_loss(self, dec_scores: Tensor, labels: Tensor, **kwargs) -> Tensor:
        r""" Calculates the risk using the VALIDATION specific loss function """
        return self._loss(dec_scores=dec_scores, lbls=labels, f_loss=self.val_loss, **kwargs)

    @staticmethod
    def has_any(mask: Tensor) -> bool:
        r""" Checks if the mask has any set to \p True """
        assert mask.dtype == torch.bool, "Mask should be a Boolean Tensor"
        return bool(mask.any().item())
