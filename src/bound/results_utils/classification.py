__all__ = [
    "calc",
]

import dataclasses
import logging
from typing import NoReturn, Tuple

import torch
from torch import BoolTensor, LongTensor, Tensor

from . import utils
from .. import _config as config
from .. import learner_ensemble
from ..types import TensorGroup


@dataclasses.dataclass
class ClassificationResults:
    name: str
    x: Tensor
    y: LongTensor
    lbls: LongTensor
    ids: LongTensor
    is_correct: BoolTensor = torch.zeros(0)

    full_wide: Tensor = torch.zeros(0)


def calc(model: learner_ensemble.DisjointEnsemble, tg: TensorGroup) -> NoReturn:
    r"""
    Calculates and writes to disk the model's results when performing classification.

    :param model: Ensemble learner
    :param tg: \p TensorGroup of the results
    """
    base_msg = "Calculating classification results for the %s dataset"
    test_x, test_y, test_lbls, test_ids = _get_test_x_y(tg=tg)
    ds_info = ClassificationResults(name="Test", x=test_x, y=test_y, lbls=test_lbls, ids=test_ids)
    msg = base_msg % ds_info.name
    logging.info(f"Starting: {msg}")

    with torch.no_grad():
        ds_info.full_wide = model.forward_wide(x=ds_info.x).cpu()

    logging.info(f"COMPLETED: {msg}")

    # Logs the robustness bounds
    _log_ds_bounds(model=model, ds_info=ds_info)


def _get_test_x_y(tg: TensorGroup) -> Tuple[Tensor, LongTensor, LongTensor, LongTensor]:
    r""" Select the u.a.r. (without replacement) test set to consider. """
    return tg.test_x, tg.test_y, tg.test_lbls, tg.test_ids


def _log_ds_bounds(model: learner_ensemble.DisjointEnsemble,
                   ds_info: ClassificationResults) -> NoReturn:
    r""" Log the certified robustness bounds """
    # Sort all the labels
    all_lbls, _ = torch.sort(torch.unique(ds_info.lbls), dim=0)
    n_cls = torch.max(all_lbls).item() + 1

    # Calculate the robustness bounds
    full_yhat = torch.argmax(ds_info.full_wide, dim=2)
    plurality = model.calc_classification_bound(full_yhat=full_yhat, n_cls=n_cls,
                                                y_lbl=ds_info.lbls)

    for k in config.TOP_K_VALS:
        bound_res = model.calc_topk_bound(k=k, full_yhat=full_yhat, n_cls=n_cls, y=ds_info.y)
        utils.log_robustness(bounds=[bound_res], bounds_desc=[f"Plurality Top-{k}"])

    runoff = _get_runoff_robustness(model=model, ds_info=ds_info)
    # As explained above, bound distance is 0 which represents the example is correctly labeled.
    utils.log_robustness(bounds=[runoff, plurality], bounds_desc=["Run-Off", "Plurality"])


def _get_runoff_robustness(model: learner_ensemble.DisjointEnsemble,
                           ds_info: ClassificationResults) -> LongTensor:
    r""" Calculate the run-off certified robustness """
    bound, pred = model.calc_runoff_bound(full_yhat=ds_info.full_wide)

    is_correct = pred == ds_info.y
    assert is_correct.shape == pred.shape

    # Log the run-off accuracy
    acc = is_correct.sum().float() / is_correct.numel()
    logging.info(f"Run-Off Accuracy: {acc:.2%}")

    bound_correct = bound.clone()
    bound_correct[~is_correct] = -1
    return bound_correct
