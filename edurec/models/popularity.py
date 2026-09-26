"""Non-personalised baselines: random, all-time popularity and trending."""

import numpy as np
import pandas as pd

from ..config import SEED
from .base import Recommender


class RandomRecommender(Recommender):
    """Random order. Only here as a floor that every other model has to beat."""

    name = "Random"

    def __init__(self, seed: int = SEED):
        self.rng = np.random.default_rng(seed)

    def fit(self, train, catalogue):
        self.n_items = catalogue.n_items
        return self

    def score(self, history, weights=None):
        return self.rng.random(self.n_items)


class PopularityRecommender(Recommender):
    """Videos opened by the most learners in the training data."""

    name = "Popularity"

    def fit(self, train, catalogue):
        counts = np.bincount(train["item_index"].to_numpy(), minlength=catalogue.n_items)
        self.counts = counts.astype(float)
        return self

    def score(self, history, weights=None):
        return self.counts.copy()


class TrendingRecommender(Recommender):
    """Videos first opened by the most learners in the days just before "now".

    This is what I use for brand-new learners. The catalogue changed a lot between 2016 and 2021
    (Microsoft Teams videos, for example, took off in 2020), so what is popular right now is a better
    first guess than what was popular over five years. All-time counts break ties.
    """

    name = "Trending"

    def __init__(self, window_days: int = 30):
        self.window_days = window_days

    def fit(self, train, catalogue, now=None):
        now = pd.Timestamp(now) if now is not None else train["first_seen"].max() + pd.Timedelta(seconds=1)
        start = now - pd.Timedelta(days=self.window_days)
        recent = train[(train["first_seen"] >= start) & (train["first_seen"] < now)]
        all_time = train[train["first_seen"] < now]
        n = catalogue.n_items
        recent_counts = np.bincount(recent["item_index"].to_numpy(), minlength=n).astype(float)
        all_counts = np.bincount(all_time["item_index"].to_numpy(), minlength=n).astype(float)
        self.counts = recent_counts + all_counts / (all_counts.max() + 1.0)
        self.now = now
        return self

    def score(self, history, weights=None):
        return self.counts.copy()
