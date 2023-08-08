__all__ = [
    "calc",
]

import collections
from typing import List, NoReturn, Tuple

import numpy as np

import torch
from torch import BoolTensor, LongTensor, Tensor

from .. import _config as config


def calc(all_votes: Tensor) -> Tuple[LongTensor, LongTensor]:
    r"""
    Calculates the top-1 run-off robustness

    :param all_votes:
    :return: Top-1 run-off robustness
    """
    dp = _build_2v1_dp_matrix()
    all_cls = list(range(config.NUM_CLASSES))
    all_bounds, all_preds = [], []
    top1_votes, _ = torch.argmax(all_votes, dim=-1).sort(dim=-1)
    for i in range(all_votes.shape[0]):
        votes = all_votes[i:i + 1]

        top1_counter = collections.Counter(all_cls + top1_votes[i].tolist())
        pred, sec = _get_pred(votes=votes, top1_counter=top1_counter)
        all_preds.append(pred)

        cert_r1 = _calc_certr1(top1_counter=top1_counter, pred=pred, dp=dp)
        cert_r2 = _calc_certr2(votes=votes, top1_counter=top1_counter, pred=pred, sec=sec)

        # Certificate is the minimum of the two certificates
        cert = min(cert_r1, cert_r2) - 1
        all_bounds.append(cert)

    all_bounds = torch.Tensor(all_bounds).long()
    all_preds = torch.Tensor(all_preds).long()
    # Calculate the robustness bounds
    return all_bounds, all_preds


def _check_votes_shape(votes: Tensor) -> NoReturn:
    assert votes.shape[0] == 1, "Expect one instance at a time"
    assert votes.shape[1] == config.get_n_model(), "Unexpected model count"


def _get_pred(votes: Tensor, top1_counter: collections.Counter) -> Tuple[int, int]:
    _check_votes_shape(votes=votes)

    # Get the top 2 labels
    most_common = top1_counter.most_common(n=2)
    # Default assignment for first and second
    first, second = most_common[0][0], most_common[1][0]
    # Check the first and second labels
    n_first, n_sec = _get_second_round_votes(votes=votes, c1=first, c2=second)
    if n_first < n_sec or (n_first == n_sec and second < first):
        first, second = second, first
    return first, second


def _get_second_round_votes(votes: Tensor, c1: int, c2: int) -> Tuple[int, int]:
    r"""
    Calculate the second votes for two labels \p c1 and \p c2
    :param c1: Label 1
    :param c2: Label 2
    :return: Number of votes for c1 and c2 respectively
    """
    _check_votes_shape(votes=votes)

    # Get the vector of votes
    c1_vec = votes[:, :, c1].squeeze()
    c2_vec = votes[:, :, c2].squeeze()
    assert c1_vec.numel() == config.get_n_model(), "Unexpected number of predictions"
    assert len(c1_vec.shape) == 1, "Prediction not a vector"
    assert c1_vec.shape == c2_vec.shape, "Mismatch in shape"

    # Compare the predictions
    c1_compare, c2_compare = c1_vec > c2_vec, c2_vec > c1_vec  # type: BoolTensor  # noqa
    assert c1_compare.shape == c2_compare.shape == c1_vec.shape, "Bizarre shape for compare"
    # Calculate the bound
    n_c1, n_c2 = c1_compare.sum().item(), c2_compare.sum().item()
    return n_c1, n_c2


def _build_2v1_dp_matrix() -> List[List[int]]:
    r""" Calculate the DP matrix for run-off elections """
    n_model = config.get_n_model()
    dp = [(n_model + 2) * [0] for _ in range(n_model + 2)]
    for i in range(len(dp)):
        for j in range(len(dp[0])):
            if min(i, j) < 2:
                dp[i][j] = (max(i, j) + 1) // 2  # Simulate the ceiling function by adding 1
                continue
            dp[i][j] = 1 + min(dp[i - 2][j - 1], dp[i - 1][j - 2])
    return dp


def _calc_certr1(top1_counter: collections.Counter, pred: int, dp: List[List[int]]) -> int:
    r""" Calculate the 2v1 robustness """
    cert_r1 = np.inf
    valid_cls = set(range(config.NUM_CLASSES)) - {pred}
    assert len(valid_cls) == config.NUM_CLASSES - 1, "Bizarre class set"

    for c1 in range(config.NUM_CLASSES):
        if c1 == pred:
            continue
        for c2 in range(c1 + 1, config.NUM_CLASSES):
            if c2 == pred:
                continue
            gap1 = _calc_round1_gap(top1_counter=top1_counter, c=pred, c_alt=c1)
            gap2 = _calc_round1_gap(top1_counter=top1_counter, c=pred, c_alt=c2)
            certv2 = dp[gap1][gap2]
            cert_r1 = min(cert_r1, certv2)
    return cert_r1


def _calc_round1_gap(top1_counter: collections.Counter, c: int, c_alt: int) -> int:
    r""" Calculate gap """
    n_c, n_c_alt = top1_counter[c], top1_counter[c_alt]
    return max(0, n_c - n_c_alt + (1 if c_alt > c else 0))


def _calc_certr2(votes: Tensor, top1_counter: collections.Counter, pred: int, sec: int) -> int:
    r""" Calculate the 1v1 robustness """
    assert pred != sec, "Prediction and second place label cannot be the same"
    valid_cls = set(range(config.NUM_CLASSES)) - {pred}
    assert len(valid_cls) == config.NUM_CLASSES - 1, "Bizarre class list"

    def _gap_to_cert(_gap: int) -> int:
        return (max(0, _gap) + 1) // 2

    cert_r2 = np.inf
    for c_alt in valid_cls:
        # Case 1: c_alt beating sec in round 1
        gap_sec_alt = _calc_round1_gap(top1_counter=top1_counter, c=sec, c_alt=c_alt)
        cert_r2_c_1 = _gap_to_cert(gap_sec_alt)

        # Case 2: c_alt beating c_pred in round 2
        gap_r2 = _calc_round2_gap(votes=votes, c=pred, c_alt=c_alt)
        cert_r2_c_2 = _gap_to_cert(gap_r2)

        # Take the maximum over the two class bounds
        cert_r2_c = max(cert_r2_c_1, cert_r2_c_2)
        cert_r2 = min(cert_r2, cert_r2_c)
    return cert_r2


def _calc_round2_gap(votes: Tensor, c: int, c_alt: int) -> int:
    r""" Calculate gap """
    n_c, n_c_alt = _get_second_round_votes(votes=votes, c1=c, c2=c_alt)
    # assert n_c + n_c_alt == config.get_n_model(), "Mismatch model count"
    return max(0, n_c - n_c_alt + (1 if c_alt > c else 0))
