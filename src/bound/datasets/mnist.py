__all__ = [
    "build_model",
    "construct_tfms",
    "load_data",
]

import dill as pk
import logging
from pathlib import Path
from typing import Optional

import torch
from torch import LongTensor, Tensor
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms

from . import _mnist_cnn as mnist_cnn
from . import utils
from .. import _config as config
from ..types import TensorGroup
from .. import utils as parent_utils

MNIST_NORMALIZE_FACTOR = 255
MNIST_FF_HIDDEN_DIM = 512
MNIST_MIN = 0
MNIST_MAX = 1


def build_model(x: Tensor, num_stages: Optional[int]) -> nn.Module:
    r""" Construct the model used for MNIST training """
    model = mnist_cnn.Model(in_channels=1, num_stages=num_stages)
    model.forward(xs=x)
    return model


def download_mnist_dataset(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    flds = []
    for train, ds_name in ((True, "training"), (False, "test")):
        ds = torchvision.datasets.MNIST(root=str(dest.parent), download=True, train=train)
        logging.debug(f"Downloaded dataset {ds_name} ({ds_name})")

        # Export the dataset information
        flds.append((f"{ds_name}.pt", ds.data, ds.targets))

    dest /= "processed"
    dest.mkdir(parents=True, exist_ok=True)
    for base_name, x, y in flds:
        full_name = dest / base_name
        if full_name.exists():
            continue
        torch.save((x, y), full_name)
    return dest


def _normalize_inputs(x: Tensor) -> Tensor:
    r""" Normalize the tensor inputs """
    assert x.dtype == torch.uint8, "Unexpected type of the data"
    x = x.unsqueeze(dim=1)
    if config.N_DISJOINT_MODELS in (25, 45):
        x = torch.transpose(x, 3, 2).clone()
        x = torch.reshape(x, x.shape).clone()
    return utils.structure_2d_x(x=x, prenorm_max=MNIST_NORMALIZE_FACTOR,
                                normalize_factor=MNIST_NORMALIZE_FACTOR)


def _build_ids(y: LongTensor, offset: int = 0) -> LongTensor:
    r""" Each training and test instance is assigned a unique ID number """
    ids = torch.arange(y.numel(), dtype=torch.long).long()
    if offset != 0:
        ids += offset
    return ids


def _build_y(lbls: LongTensor) -> LongTensor:
    r"""
    Build the \p TensorGroup \p y vector depending on the setup. For classification, \p lbls
    is split into evens being the negative class and odds being the positive class.
    """
    # if config.IS_ROTATE:
    #     y = lbls
    # elif config.IS_CLASSIFICATION:
    #     # Separate into odd and even
    #     y = torch.full_like(lbls, utils.NEG_LABEL).long()
    #     y[lbls % 2 == 1] = utils.POS_LABEL
    # else:
    #     raise ValueError("Unknown how to build y vector for this experiment type")
    if len(lbls.shape) > 1:
        lbls = lbls.squeeze(dim=1)
    assert len(lbls.shape) == 1, "Unexpected shape of the y tensor"
    return lbls


def load_data(base_dir: Path) -> TensorGroup:
    r""" Loads the MNIST dataset """
    # Shave off suffix on the dataset name
    ds_name = config.DATASET.value.name
    suffix = "-FLAT"
    if ds_name.upper().endswith(suffix):
        ds_name = ds_name[:-len(suffix)]

    mnist_dir = base_dir / ds_name

    tg_pkl_path = parent_utils.construct_filename("bk-tg", out_dir=mnist_dir,
                                                  file_ext="pk", add_ds_to_path=False,
                                                  add_label_fields=False)

    if not tg_pkl_path.exists():
        tg = TensorGroup()

        tensors_dir = download_mnist_dataset(mnist_dir)
        tr_path, te_path = utils.get_paths(base_dir=mnist_dir, data_dir=tensors_dir)

        with open(tr_path, "rb") as f_in:
            tr_x_raw, tg.tr_lbls = torch.load(f_in)
        tg.tr_x = _normalize_inputs(x=tr_x_raw)
        tg.tr_y = _build_y(lbls=tg.tr_lbls)

        # Test data requires no splits
        with open(te_path, "rb") as f_in:
            test_x, tg.test_lbls = torch.load(f_in)
        tg.test_x = _normalize_inputs(x=test_x)
        tg.test_y = _build_y(lbls=tg.test_lbls)

        tg.calc_tr_hash()
        tg.build_ids()

        with open(tg_pkl_path, "wb+") as f_out:
            pk.dump(tg, f_out)
        # wandb_utils.upload_data(tg=tg)

    with open(tg_pkl_path, "rb") as f_in:
        tg = pk.load(f_in)  # type: TensorGroup

    config.set_num_classes(num_classes=10)
    utils.print_stats(tg=tg)

    return tg


def construct_tfms(x: Tensor):
    r""" Tuple of train and test transforms respectively """
    # Configure train transform list. Do not use random rotation in the rotation experiments
    # As that would affect the baseline accuracy
    tr_tfms_lst = [
    ]
    tfms_tr = transforms.Compose(tr_tfms_lst)

    tfms_test = transforms.Compose([
    ])

    return tfms_tr, tfms_test
