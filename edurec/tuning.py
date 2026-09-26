"""Tuning the hybrid weights on the validation videos.

I try every mix of the three signals on a 0.1 grid (66 mixes that add up to 1) and keep the one with
the best NDCG@10 on the validation cases. Scoring all 66 mixes for a case at once keeps this fast.
"""

import itertools

import numpy as np


def simplex_grid(n_components: int = 3, step: float = 0.1) -> np.ndarray:
    """All weight vectors on a grid of the given step whose entries add up to 1."""
    parts = int(round(1 / step))
    rows = [combo for combo in itertools.product(range(parts + 1), repeat=n_components) if sum(combo) == parts]
    return np.array(rows, dtype=float) / parts


def expected_hit_and_ndcg(above, tied, k: int):
    """Vectorised expected HR@k and NDCG@k when the target is tied with other videos.

    The target is equally likely to sit anywhere from position above+1 to above+tied+1, so the
    expected value is an average of the discounts over that range.
    """
    above = np.asarray(above, dtype=float)
    tied = np.asarray(tied, dtype=float)
    discount = np.concatenate([[0.0], 1.0 / np.log2(np.arange(1, k + 1) + 1.0)])
    cumulative = np.cumsum(discount)  # cumulative[p] = sum of discounts for positions 1..p

    def total(upto):
        return cumulative[np.minimum(upto, k).astype(int)]

    last = above + tied + 1
    width = tied + 1
    hit = (np.minimum(last, k) - np.minimum(above, k)) / width
    ndcg = (total(last) - total(above)) / width
    return hit, ndcg


def grid_ndcg(component_stack: np.ndarray, target: int, history: np.ndarray,
              grid: np.ndarray, k: int = 10):
    """NDCG@k of one case for every weight vector in the grid."""
    scores = grid @ component_stack  # (mixes, videos)
    scores[:, history] = -np.inf
    target_scores = scores[:, target][:, None]
    above = (scores > target_scores).sum(axis=1)
    tied = (scores == target_scores).sum(axis=1) - 1
    return expected_hit_and_ndcg(above, tied, k)[1]


def best_mix(ndcg_by_mix: np.ndarray, case_mask: np.ndarray, grid: np.ndarray, allowed=None) -> np.ndarray:
    """The weight vector with the highest mean NDCG over the selected cases.

    allowed can switch off some mixes, which I use for the ablations (for example, pathway weight 0).
    """
    means = ndcg_by_mix[:, case_mask].mean(axis=1)
    if allowed is not None:
        means = np.where(allowed, means, -np.inf)
    return grid[int(np.argmax(means))]
