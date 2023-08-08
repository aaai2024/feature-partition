__all__ = [
    "calc",
]

import dataclasses
import logging
from typing import NoReturn, Tuple

import torch
from torch import LongTensor, Tensor

from . import utils
from .. import _config as config
from ..datasets import utils as ds_utils
from .. import learner_ensemble
from ..types import TensorGroup

# Base pickling only supports pickle up to 4GB.  Use pickle protocol 4 for larger files. See:
# https://stackoverflow.com/questions/29704139/pickle-in-python3-doesnt-work-for-large-data-saving
PICKLE_PROTOCOL = 4


@dataclasses.dataclass
class RegressionResults:
    name: str
    x: Tensor
    y: Tensor
    ids: LongTensor
    full_yhat: Tensor = torch.zeros(0)
    yhat: Tensor = torch.zeros(0)

    def is_empty(self) -> bool:
        r""" Returns \p True if the dataset actually contains some results """
        return self.y.numel() == 0


def calc(model: learner_ensemble.DisjointEnsemble, tg: TensorGroup) -> NoReturn:
    r"""
    Calculates and logs the results for the rotation results

    :param model: Ensemble learner
    :param tg: \p TensorGroup of the results
    """
    msg = "Calculating regression results for the test dataset"
    logging.info(f"Starting: {msg}")

    test_x, test_y, test_ids = _get_test_x_y(tg=tg)
    res = RegressionResults(name="Test", x=test_x, y=test_y, ids=test_ids)
    if config.DATASET.is_expm1_scale():
        res.y = ds_utils.scale_expm1_y(y=res.y)

    # Calculate the prediction results
    with torch.no_grad():
        res.full_yhat = model.forward_wide(x=test_x).cpu()
    if config.DATASET.is_expm1_scale():
        res.full_yhat = ds_utils.scale_expm1_y(res.full_yhat)
    # Get the final prediction
    res.yhat = model.calc_prediction(res.full_yhat)

    logging.info(f"COMPLETED: {msg}")

    assert config.BOUND_DIST, "No bound distances specified"
    for bound_dist in config.BOUND_DIST:
        _log_ds_bounds(model=model, ds_info=res, bound_dist=bound_dist)


def _get_test_x_y(tg: TensorGroup) -> Tuple[Tensor, Tensor, LongTensor]:
    r""" Select the u.a.r. (without replacement) test set to consider. """
    return tg.test_x, tg.test_y, tg.test_ids


def _log_ds_bounds(model: learner_ensemble.DisjointEnsemble, ds_info: RegressionResults,
                   bound_dist: float) -> dict:
    r""" Log the robustness bound """
    assert bound_dist > 0, "Bound distance is assumed positive"
    bound_val = utils.calc_bound_val(dist=bound_dist, y=ds_info.y)

    # Only consider predictions inside the error bounds
    err = ds_info.yhat - ds_info.y
    is_correct = err.abs() <= bound_val
    assert is_correct.shape == ds_info.y.shape, "Unexpected shape for the correctness mask"

    # Find prediction error on a submodel basis
    full_err = ds_info.full_yhat - ds_info.y.unsqueeze(dim=1)
    all_cert = []
    bound_val.unsqueeze_(dim=1)
    for is_upper in [True, False]:
        if is_upper:
            binarized = full_err <= bound_val
        else:
            binarized = -full_err <= bound_val
        assert binarized.shape == full_err.shape, "Unexpected shape of the binarized vector"
        binarized = binarized.long()  # noqa

        # y is all ones since the binarized representation expects the compare above to
        # always be true
        y = torch.ones(binarized.shape[:1], dtype=torch.long).long()

        bound_res = model.calc_classification_bound(full_yhat=binarized, y_lbl=y, n_cls=2)
        bound_res.unsqueeze_(dim=1)
        all_cert.append(bound_res)
    # Bound is the minimum between the upper and lower bounds
    all_cert = torch.cat(all_cert, dim=1)
    combined_cert, _ = all_cert.min(dim=1)

    cert_ratio = utils.log_robustness(bounds=[combined_cert], bounds_desc=["Plurality Voting"])

    return cert_ratio
