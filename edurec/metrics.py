"""Ranking metrics.

Each test case has exactly one held-out video: the one the learner really opened next. I rank every
video the learner has not opened yet, without sampling a smaller set of negatives, because sampled
metrics can change which model looks best (Krichene and Rendle, 2020). Then I record where the
held-out video landed.

Some models give many videos the same score (for example, popularity gives 0 to every unused video).
Instead of breaking those ties in an arbitrary order, I use the expected value of each metric if the
tied videos were shuffled at random.
"""

import numpy as np

from .config import K_VALUES


def target_position(scores: np.ndarray, target: int, seen: np.ndarray):
    """Return (videos ranked strictly above the target, videos tied with it, number of candidates)."""
    scores = np.array(scores, dtype=float)
    scores[seen] = -np.inf
    value = scores[target]
    above = int((scores > value).sum())
    tied = int((scores == value).sum()) - 1
    candidates = int(np.isfinite(scores).sum())
    return above, tied, candidates


def case_metrics(above: int, tied: int, candidates: int, ks=K_VALUES) -> dict:
    """Expected HR@k, NDCG@k, MRR and AUC for one case, averaging over the tied positions."""
    positions = np.arange(above + 1, above + tied + 2, dtype=float)  # every position the target could take
    result = {}
    for k in ks:
        inside = positions <= k
        result[f"hr@{k}"] = float(inside.mean())
        result[f"ndcg@{k}"] = float(np.where(inside, 1.0 / np.log2(positions + 1.0), 0.0).mean())
        # With one relevant video, recall@k equals HR@k and precision@k is HR@k divided by k.
        result[f"precision@{k}"] = result[f"hr@{k}"] / k
        result[f"recall@{k}"] = result[f"hr@{k}"]
    result["mrr"] = float((1.0 / positions).mean())
    result["auc"] = float(1.0 - (positions.mean() - 1.0) / max(candidates - 1, 1))
    result["rank"] = float(positions.mean())
    return result


def catalogue_coverage(top_lists, n_items: int) -> float:
    """Share of the catalogue that appears in at least one learner's top-k list."""
    shown = set()
    for top in top_lists:
        shown.update(int(i) for i in top)
    return len(shown) / n_items


def novelty(top_lists, item_counts: np.ndarray, n_learners: int) -> float:
    """Mean self-information of recommended videos, -log2(share of learners who opened them).

    Higher means the list leans towards less popular videos (Castells, Hurley and Vargas, 2015).
    """
    share = (item_counts + 1.0) / (n_learners + 1.0)
    information = -np.log2(share)
    values = [information[np.asarray(top, dtype=int)].mean() for top in top_lists if len(top)]
    return float(np.mean(values))


def intra_list_diversity(top_lists, similarity: np.ndarray) -> float:
    """1 minus the mean content similarity between pairs of videos in the same list."""
    values = []
    for top in top_lists:
        top = np.asarray(top, dtype=int)
        if len(top) < 2:
            continue
        block = similarity[np.ix_(top, top)]
        pairs = block[np.triu_indices(len(top), k=1)]
        values.append(1.0 - pairs.mean())
    return float(np.mean(values))
