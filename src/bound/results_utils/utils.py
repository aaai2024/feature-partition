__all__ = [
    "build_bound_str",
    "log_robustness",
]

import logging
from typing import List, NoReturn, Union

import torch
from torch import BoolTensor, LongTensor, Tensor

from .. import _config as config

PLOT_QUANTILE = 0.05
DEFAULT_N_BINS = 20


def log_robustness(bounds: List[LongTensor], bounds_desc: List[str]) -> NoReturn:
    r""" Log the certification ratio for the predictions """
    assert len(bounds) == len(bounds_desc), "Mismatch in the bound shapes"
    for bound in bounds:
        if len(bound.shape) > 1:
            bound.squeeze_(dim=1)
    shape = bounds[0].shape
    assert all(bound.shape == shape for bound in bounds), "Mismatch bound shapes"

    max_bound = max(bound.max().item() for bound in bounds)
    for i in range(0, max_bound + 1):
        bound_masks = [bound >= i for bound in bounds]
        _print_cert_res(header="Cert.", bound_cnt=i, bound_masks=bound_masks,
                        masks_desc=bounds_desc)


def _print_cert_res(header: str, bound_cnt: int, bound_masks: List[BoolTensor],
                    masks_desc: List[str]) -> NoReturn:
    r"""
    Standardizes printing the certification results
    :return: Prints the certified ratio
    """
    tot_count = bound_masks[0].numel()
    assert all(tot_count == mask.numel() for mask in bound_masks), "Masks have different sizes"

    logging.info(f"{header} Robustness: {bound_cnt}")
    for mask, desc in zip(bound_masks, masks_desc):
        cert_count = torch.sum(mask).item()

        ratio = cert_count / tot_count
        logging.info(f"{desc} {header} Acc.: {ratio:.2%} ({cert_count} / {tot_count})")


def build_bound_str(bound_dist: Union[float, int, str]) -> str:
    r""" Construct the bound string from the distance """
    assert isinstance(bound_dist, (float, int, str)), f"Type is {bound_dist.__class__.__name__}"
    bound_str = str(bound_dist)
    if config.IS_BOUND_PERCENT:
        bound_str += "%"
    return bound_str


def calc_bound_val(dist: float, y: Tensor) -> Union[float, Tensor]:
    r""" Standardizes the calculation of the bound value """
    if config.IS_BOUND_PERCENT:
        return dist / 100 * y
    return torch.full_like(y, dist)
