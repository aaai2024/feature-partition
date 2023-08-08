__all__ = [
    "TEST_X_KEY",
    "TEST_Y_KEY",
    "TR_X_KEY",
    "TR_Y_KEY",
    "ToFloatAndNormalize",
    "VAL_X_KEY",
    "VAL_Y_KEY",
    "download_from_google_drive",
    "get_paths",
    "in1d",
    "make_normalize_transform",
    "print_stats",
    "scale_expm1_y",
    "structure_2d_x",
]

import collections
import logging
from pathlib import Path
import tarfile
from typing import NoReturn, Optional, Tuple, Union

import gdown

import torch
from torch import BoolTensor, Tensor
import torch.nn as nn
import torchvision.transforms

from .. import _config as config
from ..types import TensorGroup


TR_X_KEY = "tr_x"
TR_Y_KEY = "tr_y"
VAL_X_KEY = "val_x"
VAL_Y_KEY = "val_y"
TEST_X_KEY = "test_x"
TEST_Y_KEY = "test_y"


def get_paths(base_dir: Path, data_dir: Path) -> Tuple[Path, Path]:
    r""" Reduce the training and test sets based on a fixed divider of the ordering """
    # Location to store the pruned data
    prune_dir = base_dir / "pruned"
    prune_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    # div = int(round(1 / config.VALIDATION_SPLIT_RATIO))
    for is_train in [True, False]:
        # Load the complete source data
        base_fname = "training" if is_train else "test"
        # Support two different file extensions
        for file_ext in (".pth", ".pt"):
            path = data_dir / (base_fname + file_ext)
            if not path.exists():
                continue
            paths.append(path)
            break
        else:
            raise ValueError("Unable to find processed tensor")

    return paths[0], paths[1]


def in1d(ar1, ar2) -> BoolTensor:
    r""" Returns \p True if each element in \p ar1 is in \p ar2 """
    mask = ar2.new_zeros((max(ar1.max(), ar2.max()) + 1,), dtype=torch.bool)
    mask[ar2.unique()] = True
    return mask[ar1]


def make_normalize_transform(x: Tensor):
    r""" Create a normalize transform based on the channel statistics of the \p x \p Tensor """
    # Normalization specific to the submodel's training set
    std, mean = torch.std_mean(x, dim=(0, 2, 3))
    return torchvision.transforms.Normalize(mean, std)


def download_from_google_drive(dest: Path, gd_url: str, file_name: str,
                               decompress: bool = False) -> NoReturn:
    r"""
    Downloads the source data from Google Drive

    :param dest: Folder to which the dataset is downloaded
    :param gd_url: Google drive file url
    :param file_name: Filename to store the downloaded file
    :param decompress: If \p True (and \p file_name has extension ".tar.gz"), unzips the downloaded
                       zip file
    """
    full_path = dest / file_name
    if full_path.exists():
        logging.info(f"File \"{full_path}\" exists.  Skipping download")
        return

    # Define the output files
    dest.mkdir(exist_ok=True, parents=True)
    gdown.download(url=gd_url, output=str(full_path), quiet=config.QUIET)
    if file_name.endswith(".tar.gz"):
        if decompress:
            with tarfile.open(str(full_path), "r") as tar:
                tar.extractall(path=str(dest))
    else:
        assert not decompress, "Cannot decompress a non tar.gz file"


class ToFloatAndNormalize(nn.Module):
    def __init__(self, normalize_factor: float):
        super().__init__()
        self._factor = normalize_factor

    def forward(self, x: Tensor) -> Tensor:
        out = x.float()
        out.div_(self._factor)
        return out


def print_stats(tg: TensorGroup) -> NoReturn:
    r""" Prints a simple perturbation histogram to understand the extent of each perturbation """
    # Find a consistent min and max range
    y_vals = torch.cat([tg.tr_y, tg.test_y], dim=0)
    if config.DATASET.is_expm1_scale():
        y_vals = scale_expm1_y(y=y_vals)
    min_y, max_y = y_vals.min().item(), y_vals.max().item()
    # Print statistics on the y values
    for ds_prefix, ds_name in [("tr", "Train"), ("test", "Test")]:
        y = tg.__getattribute__(f"{ds_prefix}_y")
        logging.info(f"{config.DATASET.value.name} {ds_name} Dataset Size: {y.numel()}")

        if config.DATASET.is_expm1_scale():
            y = scale_expm1_y(y=y)
        if not config.IS_CLASSIFICATION:
            _print_y_stats(y=y, ds_name=ds_name, min_y=min_y, max_y=max_y)
        else:
            _log_prior(ds_name=ds_name, y=y)


def _log_prior(ds_name: str, y: Tensor) -> NoReturn:
    r""" Calculate the prior """
    assert config.IS_CLASSIFICATION, "Calculating prior but not running classification"
    # n_pos, n_ele = (y == POS_LABEL).sum(), y.numel()
    counter = collections.Counter(y.tolist())
    keys = sorted(counter.keys())
    for key in keys:
        n_key = counter[key]
        rate = n_key / y.numel()
        logging.info(f"{ds_name} {key} Prior: {n_key} / {y.numel()} ({rate:.2%})")


def _print_y_stats(y: Tensor, ds_name: str, min_y: float, max_y: float,
                   n_bins: int = 20) -> NoReturn:
    r""" Print stats on the Y values """
    # Allow for small floating point errors
    histc = torch.histc(y, min=min_y, max=max_y, bins=n_bins).long()
    assert y.numel() == torch.sum(histc).item(), "Elements lost in histogram"

    # Standard header when printing the Y stats
    head_flds = [config.DATASET.value.name, ds_name, "Y"]
    header = " ".join(head_flds)

    # Print the histogram
    lin = torch.linspace(min_y, max_y, steps=n_bins + 1)
    for i_bin in range(1, n_bins + 1):
        range_str = f"[{lin[i_bin - 1]:.3f},{lin[i_bin]:.3f})"
        bin_cnt = histc[i_bin - 1].item()
        rate = bin_cnt / y.numel()
        logging.info(f"{header} Bin #{i_bin:02d} {range_str}: {bin_cnt} ({rate:.2%})")
    # Exclude anything that exceeds min_val or max_val
    if histc.numel() > n_bins:
        logging.info(f"{header} Other Excluded: {histc[-1].item()}")

    # Print perturbation stats
    quantiles = torch.tensor([0., 0.25, 0.5, 0.75, 1.])
    names = ["Min", "25%-Quartile", "Median", "75%-Quartile", "Max"]
    quant_vals = torch.quantile(y, q=quantiles)
    for name, val in zip(names, quant_vals.tolist()):
        logging.info(f"{header} Perturb {name}: {val:.3f}")
    # Interquartile range
    val = quant_vals[-2] - quant_vals[1]
    logging.info(f"{header} IQR: {val.item():.3f}")

    std, mean = torch.std_mean(y, unbiased=True)
    for val, val_name in zip((mean, std), ("Mean", "Stdev")):
        logging.info(f"{header} {val_name}: {val.item():.3f}")


def structure_2d_x(x: Tensor, prenorm_max: Optional[Union[float, int]] = None,
                   normalize_factor: Optional[Union[float, int]] = None) -> Tensor:
    r""" Structures 2D data (e.g., CIFAR10, MNIST) for compatibility with the code """
    # If the model is torch, will be using a convolutional network so keep as 2D
    if config.ALT_TYPE.is_torch():
        pass
    # If non-torch, flatten the pixel dimension
    else:
        x = x.reshape([x.shape[0], x.shape[1], -1])

    if normalize_factor is not None:
        x = x.float() / normalize_factor
    assert x.dtype == torch.float, "X expected to be a float"
    return x


def scale_expm1_y(y: Tensor) -> Tensor:
    r""" Large disparity in price values so housing prices are scaled. """
    return torch.expm1(y)
