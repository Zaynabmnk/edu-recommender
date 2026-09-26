"""Item-based collaborative filtering (Sarwar et al., 2001; Deshpande and Karypis, 2004)."""

import numpy as np
from sklearn.preprocessing import normalize

from .base import Recommender, learner_item_matrix, recency_weights


class ItemKNN(Recommender):
    """Two videos are similar when the same learners opened both.

    Similarity is the cosine between the videos' columns in the learner x video matrix. A learner's
    score for a video adds up its similarity to each video in their history, weighted by how recent
    that video is.
    """

    name = "Item-kNN CF"

    def __init__(self, decay: float = 1.0, use_weights: bool = False):
        self.decay = decay
        self.use_weights = use_weights

    def fit(self, train, catalogue, values=None):
        values = values if self.use_weights else None
        matrix, _ = learner_item_matrix(train, catalogue.n_items, values)
        columns = normalize(matrix.T.tocsr(), axis=1)
        similarity = (columns @ columns.T).toarray()
        np.fill_diagonal(similarity, 0.0)
        self.similarity = similarity
        return self

    def score(self, history, weights=None):
        history = np.asarray(history, dtype=int)
        if len(history) == 0:
            return np.zeros(self.similarity.shape[0])
        w = recency_weights(len(history), self.decay)
        if self.use_weights and weights is not None:
            w = w * np.asarray(weights, dtype=float)
        return w @ self.similarity[history]

    def contributions(self, history, item):
        """How much each history video adds to one candidate's score (used for explanations)."""
        w = recency_weights(len(history), self.decay)
        return w * self.similarity[np.asarray(history, dtype=int), item]
