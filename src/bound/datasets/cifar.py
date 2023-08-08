__all__ = [
    "build_model",
    "construct_tfms",
    "load_data",
]

import dataclasses
import dill as pk
from pathlib import Path
from typing import NoReturn, Tuple

import numpy as np

import torch
from torch import LongTensor, Tensor
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from . import _cifar_cnn as cifar10_resnet  # noqa
from . import utils
from .. import _config as config
from ..types import TensorGroup
from .. import utils as parent_utils

CIFAR10_NORMALIZE_FACTOR = 255
CIFAR_PADDING = 4


def build_model(x: Tensor) -> nn.Module:
    r""" Construct the model used for MNIST training """
    n_classes = config.DATASET.get_n_classes()
    model = cifar10_resnet.ResNet9(in_channels=x.shape[1], n_classes=n_classes, scale=1 / 8)
    model.eval()
    return model


def download_data(cifar_dir: Path) -> Tuple[Path, Path]:
    r"""
    Downloads the CIFAR10 dataset then returns the path to the training and test tensors
    respectively.
    """
    tfms = [torchvision.transforms.ToTensor()]

    tensors_dir = cifar_dir / "processed"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for is_training in [True, False]:
        # Path to write the processed tensor
        file_path = tensors_dir / f"{'training' if is_training else 'test'}.pt"
        if file_path.exists():
            paths.append(file_path)
            continue

        # noinspection PyTypeChecker
        ds = torchvision.datasets.cifar.CIFAR10(cifar_dir,
                                                transform=torchvision.transforms.Compose(tfms),
                                                train=is_training, download=True)

        # Write the pickle fields
        x = ds.data
        x = np.transpose(x, (0, 3, 1, 2))
        x = torch.from_numpy(x)
        y = torch.LongTensor(ds.targets)
        with open(str(file_path), "wb+") as f_out:
            torch.save((x, y), f_out)
        paths.append(file_path)
    # Path to the train and test set tensor files respectively
    return paths[0], paths[1]


def _build_y(lbls: LongTensor) -> LongTensor:
    r"""
    Build the \p TensorGroup \p y vector depending on the setup. For classification, \p lbls
    is split into vehicles (negative class) and animals (positive class).
    """
    if len(lbls.shape) > 1:
        lbls = lbls.squeeze(dim=1)
    assert len(lbls.shape) == 1, "Unexpected shape of the y tensor"
    return lbls


def _format_x(x: Tensor) -> Tensor:
    assert x.dtype == torch.uint8, "Unexpected type of the data"
    return x.float() / CIFAR10_NORMALIZE_FACTOR


def load_data(cifar_dir: Path) -> TensorGroup:
    r""" Loads the CIFAR10 dataset """
    tg_pkl_path = parent_utils.construct_filename("bk-tg", out_dir=cifar_dir,
                                                  file_ext="pk", add_ds_to_path=False,
                                                  add_label_fields=False)

    if not tg_pkl_path.exists():
        tr_path, te_path = download_data(cifar_dir)
        tg = TensorGroup()

        with open(tr_path, "rb") as f_in:
            tg.tr_x, tg.tr_lbls = torch.load(f_in)
        tg.tr_x = _format_x(x=tg.tr_x)
        tg.tr_y = _build_y(lbls=tg.tr_lbls)

        # Test data requires no splits
        with open(te_path, "rb") as f_in:
            tg.test_x, tg.test_lbls = torch.load(f_in)
        tg.test_x = _format_x(x=tg.test_x)
        tg.test_y = _build_y(lbls=tg.test_lbls)

        tg.calc_tr_hash()
        tg.build_ids()

        with open(tg_pkl_path, "wb+") as f_out:
            pk.dump(tg, f_out)
        # wandb_utils.upload_data(tg=tg, labels=LABELS)

    with open(tg_pkl_path, "rb") as f_in:
        tg = pk.load(f_in)  # type: TensorGroup

    config.set_num_classes(num_classes=config.DATASET.get_n_classes())
    _normalize_x(tg=tg)
    utils.print_stats(tg=tg)

    return tg


def _normalize_x(tg: TensorGroup) -> NoReturn:
    r""" Normalize x according to the mean and variance"""
    std, mean = torch.std_mean(tg.tr_x, dim=(0, 2, 3), keepdim=True)
    for f in dataclasses.fields(tg):
        f_name = f.name
        if not f_name.endswith("_x"):
            continue
        val = tg.__getattribute__(f_name)
        val -= mean
        val /= std


def construct_tfms(x: Tensor):
    r""" Tuple of train and test transforms respectively """
    erase_transform = transforms.RandomErasing(p=1., scale=(1/32, 1/32), ratio=(1., 1.),
                                               value=0.0)

    tfms_tr = transforms.Compose([
        transforms.RandomCrop(size=x.shape[-1], padding=CIFAR_PADDING, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(p=0.5),
        erase_transform,
    ])

    tfms_test = transforms.Compose([])

    return tfms_tr, tfms_test
