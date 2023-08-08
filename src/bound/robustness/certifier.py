__all__ = [
    "calc_classification_bound",
    "calc_topk_bound",
]

import collections
import dataclasses
import heapq
from typing import List, NoReturn, Tuple

import torch
from torch import LongTensor, Tensor


@dataclasses.dataclass
class HeapInstance:
    lbl: int
    vote_cnt: int

    def __lt__(self, other):
        return (self.vote_cnt < other.vote_cnt
                or self.vote_cnt == other.vote_cnt and self.lbl > other.lbl)


def calc_classification_bound(n_cls: int, full_yhat: Tensor, y: LongTensor) -> LongTensor:
    r"""
    Calculates the top-1 bound
    """
    bound = []
    assert n_cls > 2 or full_yhat.shape[1] & 1 == 1, \
        "For binary classification, expect odd num votes"
    assert y.numel() == y.shape[0], f"y expected to be a vector but has shape {y.shape}"

    for i in range(full_yhat.shape[0]):
        # top2 = _get_top2_votes_counts(n_cls=n_cls, y_hat=full_yhat[i])
        top2 = _get_topk_votes_counts(n_cls=n_cls, k=1, y_hat=full_yhat[i].tolist())
        top2 = top2[::-1]  # Function above returns them in ascending not descending order
        y_hat, plural_cnt = top2[0]
        y_lbl = y[i].item()
        if y_hat != y_lbl:
            ex_bound = -1
        else:
            second_lbl, second_cnt = top2[1]
            # Calculate the bound
            diff = plural_cnt - second_cnt
            if y_hat > second_lbl:
                diff -= 1
            ex_bound = diff // 2
        bound.append(ex_bound)

        top1_bound = _calc_topk_bound(k=1, n_cls=n_cls, y_hat=full_yhat[i].tolist(),
                                      y_lbl=y_lbl)
        assert top1_bound == ex_bound, "Bound mismatch"
    # Construct the final bound
    bound = torch.LongTensor(bound)
    return bound


def calc_topk_bound(k: int, n_cls: int, full_yhat: Tensor, y: LongTensor) -> LongTensor:
    r"""
    Modifies the script to calculate the top-k bound
    :param k:
    :param n_cls:
    :param full_yhat:
    :param y:
    :return:
    """
    assert k < n_cls, "Does not make sense to have k greater than or equal to n_cls"
    assert y.numel() == full_yhat.shape[0], "Y vector length mismatch"

    bound = []
    for i in range(y.numel()):
        r = _calc_topk_bound(k=k, n_cls=n_cls, y_hat=full_yhat[i].tolist(), y_lbl=y[i].item())
        bound.append(r)
    # Construct the final bound
    bound = torch.LongTensor(bound)
    return bound


def _calc_topk_bound(k: int, n_cls: int, y_hat: List[int], y_lbl: int) -> int:
    r"""
    Calculates a single top-k bound for one prediction
    :param k:
    :param n_cls:
    :param y_hat: List of predictions by the ensemble
    :param y_lbl:
    :return:

    >>> _calc_topk_bound(k=2, n_cls=10, y_hat=[4] + 3 * [8] + 6 * [3], y_lbl=3)
    3
    >>> _calc_topk_bound(k=2, n_cls=10, y_hat=[2, 4] + 7 * [5], y_lbl=5)
    3
    >>> _calc_topk_bound(k=2, n_cls=10, y_hat=[4, 6] + 7 * [5], y_lbl=5)
    4
    >>> _calc_topk_bound(k=2, n_cls=10, y_hat=[4, 6] + 8 * [5], y_lbl=5)
    4
    >>> _calc_topk_bound(k=2, n_cls=10, y_hat=[2, 4] + 8 * [5], y_lbl=5)
    4
    >>> _calc_topk_bound(k=3, n_cls=4, y_hat=10 * [0], y_lbl=0)
    8
    >>> _calc_topk_bound(k=3, n_cls=4, y_hat=10 * [3], y_lbl=3)
    7
    >>> _calc_topk_bound(k=3, n_cls=4, y_hat=12 * [0], y_lbl=0)
    9
    >>> _calc_topk_bound(k=3, n_cls=4, y_hat=12 * [3], y_lbl=3)
    8
    >>> _calc_topk_bound(k=1, n_cls=2, y_hat=10 * [0], y_lbl=0)
    5
    >>> _calc_topk_bound(k=1, n_cls=2, y_hat=11 * [0], y_lbl=0)
    5
    >>> _calc_topk_bound(k=1, n_cls=2, y_hat=10 * [1], y_lbl=0)
    -1
    >>> _calc_topk_bound(k=1, n_cls=2, y_hat=5 * [0] + 5 * [1], y_lbl=0)
    0
    >>> _calc_topk_bound(k=1, n_cls=2, y_hat=6 * [0] + 5 * [1], y_lbl=0)
    0
    >>> _calc_topk_bound(k=1, n_cls=2, y_hat=5 * [0] + 5 * [1], y_lbl=1)
    -1
    >>> _calc_topk_bound(k=1, n_cls=2, y_hat=5 * [0] + 6 * [1], y_lbl=1)
    0
    >>> _calc_topk_bound(k=1, n_cls=2, y_hat=10 * [1], y_lbl=1)
    4
    >>> _calc_topk_bound(k=1, n_cls=10, y_hat=10 * [1], y_lbl=0)
    -1
    >>> _calc_topk_bound(k=2, n_cls=10, y_hat=10 * [1], y_lbl=0)
    0
    >>> _calc_topk_bound(k=2, n_cls=10, y_hat=10 * [1], y_lbl=2)
    -1
    >>> _calc_topk_bound(k=3, n_cls=10, y_hat=10 * [1], y_lbl=0)
    1
    >>> _calc_topk_bound(k=3, n_cls=10, y_hat=[0] + 10 * [1], y_lbl=0)
    1
    >>> _calc_topk_bound(k=4, n_cls=10, y_hat=[0] + 10 * [1], y_lbl=0)
    2
    >>> _calc_topk_bound(k=20, n_cls=100, y_hat=10 * [0] + 200 * [1], y_lbl=0)
    18
    >>> _calc_topk_bound(k=3, n_cls=10, y_hat=[4, 7, 7] + 22 * [2], y_lbl=7)
    0
    >>> _calc_topk_bound(k=3, n_cls=10, y_hat=10 * [1], y_lbl=2)
    0
    """
    assert min(y_hat) >= 0, "Expect labels in range [0, n_cls)"
    assert max(y_hat) < n_cls, "Expect labels in range [0, n_cls)"
    assert len(y_hat) == 1 or (k < n_cls and k < len(y_hat)), \
        "k does not match the expected dimensions or # classes"

    # Get the votes counts for the (k + 1) highest ranked labels
    top_vote_cnt = _get_topk_votes_counts(k=k, n_cls=n_cls, y_hat=y_hat)
    y_idx = -1
    # Identify all vote counts in the top k+1 which are less than the true label's count
    for j, (j_lbl, _) in enumerate(top_vote_cnt):
        if j_lbl != y_lbl:
            continue
        y_idx = j
        break
    # Predicted label not in the top-k
    if y_idx <= 0:
        return -1
    # Simple check of top k accuracy
    if len(y_hat) == 1:
        return 0

    start_y_votes = cur_y_votes = top_vote_cnt[y_idx][1]
    filt_vote_cnt = top_vote_cnt[:y_idx]

    # Deterministic case where y has zero votes. Bound is the number of labels in the top-k
    # below it
    if start_y_votes == 0:
        r = y_idx - 1
        _verify_topk_bound(vote_cnts=top_vote_cnt, y_cnt=0, y_lbl=y_lbl, r=r)
        return r

    # If the true label runs out of votes before exiting the top-k, may need extra votes taken
    # from other labels to exit the top-k. This variable tracks these extra votes taken from a
    # label other than y
    extra_votes = 0

    for j, (_, j_votes) in enumerate(filt_vote_cnt):
        n_low_lbl = j + 1
        # Make the maximum possible step without crossing the y vote counts
        # Adding 1 since j starts at 0
        diff_votes = (cur_y_votes - j_votes) // (n_low_lbl + 1)

        split_pt = diff_votes + j_votes
        # Next vote count. Use full array since may go j can equal y_idx causing an overflow
        # error if filt_vote_cnt is used
        next_lbl_votes = top_vote_cnt[j + 1][1]
        if split_pt < next_lbl_votes:
            # Handle case where insufficient y votes alone to make y the (k + 1)-th prediction,
            # meaning the algorithm needs to take extra votes from which label has the top
            if all(v == 0 for _, v in filt_vote_cnt[:j]) and j > cur_y_votes:
                extra_votes = n_low_lbl - cur_y_votes - 1
                cur_y_votes = 0
            else:
                cur_y_votes -= diff_votes * n_low_lbl
                # Additional tweaks may be possible after the last equal step. The remaining
                # optimality gap is the number of votes which still may flip.
                gap = cur_y_votes - split_pt
                remainder = _calc_topk_remainder(top_vote_cnt=filt_vote_cnt, y_lbl=y_lbl,
                                                 gap=gap, lower_cnt=n_low_lbl)
                cur_y_votes -= remainder
            break
        else:
            cur_y_votes -= (j + 1) * (next_lbl_votes - j_votes)
    assert cur_y_votes >= 0, "Negative votes for the predicted label is not possible"

    # Calculate the bound.
    r = start_y_votes - cur_y_votes + extra_votes
    assert r >= 0, "Negative robustness not possible here"
    # Sanity check the results
    _verify_topk_bound(vote_cnts=top_vote_cnt[:y_idx], y_cnt=start_y_votes, y_lbl=y_lbl, r=r)
    return r


def _calc_topk_remainder(top_vote_cnt: List[Tuple[int, int]], y_lbl: int, gap: int,
                         lower_cnt: int) -> int:
    r"""
    Calculate top-k's method above only moves the bottom set of labels as a single monolithic
    block. This leaves a small optimality gap that must be calculated more carefully.  This
    method fills in that optimality gap by determining the small number of additional flips that
    can be made without the y target dropping out of the top-k.

    :param top_vote_cnt:
    :param y_lbl:
    :param gap:
    :param lower_cnt:
    :return: Gap value
    """
    assert 0 <= gap <= lower_cnt, "Additional group steps are possible but were not made"

    bottom_lbls = sorted([bot[0] for _, bot in zip(range(lower_cnt), top_vote_cnt)],
                         reverse=True)
    if gap == 0:
        # Determine if stepped too far in the last jump and need to back track one step
        return -1 if y_lbl > bottom_lbls[-0] else 0
    if gap == lower_cnt:
        return gap - 1

    # Always can perform gap - 1 steps. Determine if an extra step is possible where
    # ties are broken by prioritizing whichever label is smaller
    if bottom_lbls[gap] < y_lbl:
        gap -= 1
    return gap


def _get_topk_votes_counts(n_cls: int, y_hat: List[int], k: int):
    r""" Get the info on the instances with the top 2 votes """
    # Initialize each element in the counter to 1 to ensure all classes represented
    # Adding 1 does not affect the calculated bound
    counter = collections.Counter(y_hat)
    for key in range(n_cls):
        if key not in counter:
            counter[key] = 0
    all_vals = counter.most_common(n=n_cls)

    # Sort by ascending by vote count then use descending label to break ties
    all_vals = sorted(all_vals, key=lambda x: (x[1], -x[0]))
    all_vals = all_vals[-(k + 1):]

    assert len(all_vals) == k + 1, "Unexpected top-K size"
    return all_vals


def _verify_topk_bound(vote_cnts: List[Tuple[int, int]],
                       y_cnt: int, y_lbl: int, r: int) -> NoReturn:
    r"""
    Verify the top-k split point

    :param vote_cnts: Vote count structure
    :param y_cnt: Number of votes for the correct label (\p y)
    :param y_lbl: Label of the instances true label
    :param r: Certified bound
    """
    # Verify the top-k bound using a heap
    heap = [HeapInstance(lbl=lbl, vote_cnt=vote_cnt) for lbl, vote_cnt in vote_cnts]
    heapq.heapify(heap)
    targ = HeapInstance(lbl=y_lbl, vote_cnt=y_cnt)

    heap_bound = 0
    # Continue transferring votes for the true label to lower ranked labels greedily until
    # the true label is ranked below all other labels being considered.
    while targ > heap[0]:
        # Only take votes from the target if its vote count is above 0 (i.e., it has votes that
        # can be taken). Otherwise, take votes from the element with the most votes (not
        # directly considered by this loop but implicit in the vote added to \p top below.
        if targ.vote_cnt > 0:
            targ.vote_cnt -= 1
        heap_bound += 1
        assert targ.vote_cnt >= 0, "Cannot have negative votes"

        top = heapq.heappop(heap)
        top.vote_cnt += 1
        heapq.heappush(heap, top)
    # Need to subtract one since loop above runs one iteration passed the max certification
    # bound as it violates the invariant
    heap_bound -= 1
    assert r == heap_bound, "Bound calculated with heap doesn't match linear time calc"


if __name__ == "__main__":
    _calc_topk_bound(k=3, n_cls=4, y_hat=10 * [0], y_lbl=0)
    _calc_topk_bound(k=3, n_cls=4, y_hat=12 * [0], y_lbl=0)
