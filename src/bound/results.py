__all__ = [
    "calc",
]

import logging
import time
from typing import NoReturn

from . import _config as config
from .learner_ensemble import DisjointEnsemble
from . import results_utils
from .types import TensorGroup


def calc(model: DisjointEnsemble, tg: TensorGroup) -> NoReturn:
    start = time.time()
    n_test = tg.test_y.numel()

    if config.IS_CLASSIFICATION:
        res = results_utils.classification.calc(model=model, tg=tg)
    elif config.IS_REGRESSION:
        res = results_utils.regression.calc(model=model, tg=tg)
    else:
        raise ValueError("Unknown how to calculate results for this experiment type")

    tot_time = time.time() - start
    avg_time = tot_time / n_test
    logging.info(f"Total Certification Time: {tot_time:.3f}s")
    logging.info(f"Average Time to Certify One Prediction: {avg_time:.3E}")

    return res
