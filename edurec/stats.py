"""Significance tests and confidence intervals for comparing models learner by learner."""

import numpy as np
from scipy import stats

from .config import SEED


def bootstrap_ci(values, n_resamples: int = 2000, level: float = 0.95, seed: int = SEED):
    """Percentile bootstrap interval for a mean, resampling learners with replacement."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_resamples, len(values)))].mean(axis=1)
    tail = (1.0 - level) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def wilcoxon_p(a, b) -> float:
    """Two-sided Wilcoxon signed-rank test on paired per-learner scores."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.allclose(a, b):
        return 1.0
    return float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)


def mcnemar_exact(hits_a, hits_b) -> float:
    """Exact McNemar test: did one model get learners right that the other missed more often than chance?"""
    a, b = np.asarray(hits_a, dtype=bool), np.asarray(hits_b, dtype=bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    if only_a + only_b == 0:
        return 1.0
    return float(stats.binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue)


def holm(p_values):
    """Holm-Bonferroni correction, because I compare the final model against several baselines at once."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted
