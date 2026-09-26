"""Learning-path model: which video learners tend to open next.

In MARS, 78% of a learner's consecutive videos were opened within an hour of each other, and 27% of
the time the next video is simply the next one in the catalogue. Learners work through short series
of tutorials, so the order matters. This model is a first-order Markov chain (as used inside FPMC by
Rendle, Freudenthaler and Schmidt-Thieme, 2010): it counts how often each video was followed by each
other video in the training histories and turns the counts into probabilities.

I did not use a recurrent network such as GRU4Rec (Hidasi et al., 2016). The median learner has only
two videos, which is far too little data per sequence to train one.
"""

import numpy as np

from .base import Recommender, recency_weights


class PathwayRecommender(Recommender):
    name = "Learning path (Markov)"

    def __init__(self, decay: float = 0.2):
        # A small decay means "mostly what follows the last video", with earlier videos as a fallback
        # when the last one has hardly ever been followed by anything.
        self.decay = decay

    def fit(self, train, catalogue):
        n = catalogue.n_items
        ordered = train.sort_values(["user_id", "position"])
        users = ordered["user_id"].to_numpy()
        items = ordered["item_index"].to_numpy()
        same_learner = users[1:] == users[:-1]
        counts = np.zeros((n, n))
        np.add.at(counts, (items[:-1][same_learner], items[1:][same_learner]), 1.0)
        self.transitions = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        popularity = np.bincount(items, minlength=n).astype(float)
        # A tiny popularity term decides between videos that are otherwise tied. If none of a learner's
        # videos has ever been followed by another, every video ties, so after scaling this signal becomes
        # plain popularity. The hybrid's explanations check for that case.
        self.tiebreak = 1e-6 * popularity / max(popularity.max(), 1.0)
        self.n_transitions = int(same_learner.sum())
        return self

    def score(self, history, weights=None):
        history = np.asarray(history, dtype=int)
        if len(history) == 0:
            return self.tiebreak.copy()
        return recency_weights(len(history), self.decay) @ self.transitions[history] + self.tiebreak

    def contributions(self, history, item):
        w = recency_weights(len(history), self.decay)
        return w * self.transitions[np.asarray(history, dtype=int), item]
