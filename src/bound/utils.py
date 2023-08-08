__all__ = [
    "LOG_LEVEL",
    "NUM_WORKERS",
    "TORCH_DEVICE",
    "TrainTimer",
    "construct_filename",
    "get_tfms",
    "log_seeds",
    "rotate_tensor",
    "set_random_seeds",
    "set_debug_mode",
]

import dill as pk
import io
import logging
from pathlib import Path
import random
import sys
import time
from typing import NoReturn, Optional

import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torchvision.transforms.functional

from .import _config as config
from . import dirs
from .datasets import cifar
from .datasets import mnist
from .datasets import tabular
from .types import TensorGroup


TORCH_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

LOG_LEVEL = logging.DEBUG


# Intelligently select number of workers
gettrace = getattr(sys, 'gettrace', None)
# NUM_WORKERS = 0
if gettrace is None:
    NUM_WORKERS = 2
elif gettrace():
    print("Debugger detected.  Using only a single worker")
    NUM_WORKERS = 0
else:
    print("Debugger does not appear to be enabled.  Defaulting to parallel workers")
    NUM_WORKERS = 2


def set_debug_mode(seed: int = 42) -> NoReturn:
    logging.warning("Debug mode enabled")
    config.enable_debug_mode()

    torch.manual_seed(seed)


def set_random_seeds() -> NoReturn:
    r"""
    Sets random seeds to avoid non-determinism
    :See: https://pytorch.org/docs/stable/notes/randomness.html
    """
    seed = torch.initial_seed()
    torch.manual_seed(seed)
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.deterministic = True
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.benchmark = False

    seed &= 2 ** 32 - 1  # Ensure a valid seed for the learner
    random.seed(seed)
    np.random.seed(seed)

    log_seeds()


def log_seeds():
    r""" Log the seed information """
    logging.debug("Torch Random Seed: %d", torch.initial_seed())


def configure_dataset_args() -> TensorGroup:
    r""" Manages generating the source data (if not already serialized to disk """
    logging.debug("Reseed training set for consistent dataset creation given the seed")
    set_random_seeds()

    if config.DATASET.is_cifar():
        cifar_dir = dirs.DATA_DIR / config.DATASET.name
        tg = cifar.load_data(cifar_dir)
    elif config.DATASET.is_mnist():
        tg = mnist.load_data(dirs.DATA_DIR)
    elif config.DATASET.is_tabular():
        tabular_dir = dirs.DATA_DIR / config.DATASET.value.name.lower()
        tg = tabular.load_data(tabular_dir)
    else:
        raise ValueError(f"Dataset generation not supported for {config.DATASET.name}")

    _calc_num_feats(x=tg.tr_x)
    return tg


def _calc_num_feats(x: Tensor) -> NoReturn:
    r"""
    Calculate the data dimension as considered by the program. This may differ from the true
    data dimension of the \p x Tensor
    :param x: Data tensor
    """
    if len(x.shape) == 2:
        data_dim = x.shape[1]
    elif len(x.shape) in (3, 4):
        x_shape = x.shape[2:]
        data_dim = np.prod(x_shape)
    else:
        raise ValueError("Unknown how to calculate the number of dimensions")

    # Verify the data dimension at least matches the number of expected splits
    n_split = config.N_DISJOINT_MODELS
    assert data_dim >= n_split, "Data dimension less than number of models"
    if config.PATCH_FEAT_SPLIT:
        data_dim = n_split
    config.set_num_feats(n_feats=data_dim)  # noqa
    logging.info(f"Dataset Dimension: {data_dim}")


def get_new_model(x: Optional[Tensor], opt_params: Optional[dict]) -> nn.Module:
    if config.DATASET.is_cifar():
        assert x is not None, "X must be specified"
        net = cifar.build_model(x)
    elif config.DATASET.is_mnist():
        assert x is not None, "X must be specified"
        num_stages = opt_params.get("n_layers", None)
        net = mnist.build_model(x, num_stages=num_stages)
    else:
        raise ValueError(f"Model creation not supported for {config.DATASET.name}")
    return net


def get_tfms(x: Tensor):
    r""" Tuple and training and test transforms respectively """
    if config.DATASET.is_cifar():
        tfms = cifar.construct_tfms(x=x)
    elif config.DATASET.is_mnist():
        tfms = mnist.construct_tfms(x=x)
    else:
        raise ValueError(f"Transform selected not supported for {config.DATASET.name}")
    return tfms


def construct_filename(prefix: str, out_dir: Path, file_ext: str, model_num: Optional[int] = None,
                       add_timestamp: bool = False, add_ds_to_path: bool = True,
                       add_label_fields: bool = True) -> Path:
    r""" Standardize naming scheme for the filename """
    fields = [
        prefix,
        config.DATASET.name.lower().replace("_", "-"),
    ]
    if config.IS_CLASSIFICATION:
        fields.append("class")
    elif config.IS_REGRESSION:
        fields.append("reg")
    else:
        raise ValueError("Unknown learning task")

    if config.PARTITION_TRAIN and add_label_fields:
        fields.append("part")

    if model_num is not None:
        fields.append(f"m-id={model_num:04d}")
    if config.RANDOM_FEAT_SPLIT:
        fields.append("rnd")
    if config.PATCH_FEAT_SPLIT:
        fields.append("2d")
    if config.WALKING_FEAT_SPLIT:
        fields.append("walk")
    if config.DEBUG:
        fields.append("dbg")
    fields.append(f"n-mod={config.N_DISJOINT_MODELS}")
    if config.is_ssl() and add_label_fields:
        fields.append(f"ssl={config.SSL_DEGREE:03d}")

    if add_timestamp:
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
        fields.append(time_str)

    if file_ext[0] != ".":
        # Add period before extension if not already specified
        file_ext = "." + file_ext
    fields[-1] += file_ext

    # Add the dataset name to better organize files
    if add_ds_to_path:
        out_dir /= config.DATASET.name.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "_".join(fields)


def rotate_tensor(xs: Tensor, angle: Tensor) -> Tensor:
    r""" Rotate the image (counter-clockwise) by the specified angle """
    assert xs.shape[0] == angle.numel(), "Mismatch in length of rotation angle tensor"
    assert len(angle.shape) == 1, "Angle tensor has a bizarre shape"

    rotate_xs = []
    for i in range(xs.shape[0]):
        tmp_xs = torchvision.transforms.functional.rotate(img=xs[i:i + 1],
                                                          angle=angle[i].item())
        rotate_xs.append(tmp_xs)
    return torch.cat(rotate_xs, dim=0)


class TrainTimer:
    r""" Used for tracking the training time """
    def __init__(self, model_name: str, model_id: int):
        self._model_name = model_name
        self._model_id = model_id
        self._start_time = None

    def __enter__(self):
        self._start_time = time.time()

    def __exit__(self, type, value, traceback):
        elapsed = time.time() - self._start_time
        flds = [self._model_name, f"Model # {self._model_id}",
                "Training Time:", f"{elapsed:.6f}"]
        logging.info(" ".join(flds))


class CpuUnpickler(pk.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
