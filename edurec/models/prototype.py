"""My preliminary prototype from June, rebuilt so it can be tested in exactly the same way as the new models.

It is kept as close to the original as possible, including the parts I later found to be problems:
the double-counted rating, the whole-history content profile, and dividing by the maximum over all
videos, including ones the learner had already opened.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import Recommender, learner_item_matrix


class PrototypeHybrid(Recommender):
    name = "Prototype hybrid (June)"

    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha

    def fit(self, train, catalogue, values=None):
        matrix, _ = learner_item_matrix(train, catalogue.n_items, values)
        self.cf_similarity = cosine_similarity(matrix.T, dense_output=True)
        items = catalogue.items
        text = (items["name"].fillna("") + " " + items["description_raw"] + " " + items["difficulty"] + " "
                + items["job"].str.replace(",", "") + " " + items["software"].str.replace(",", "") + " "
                + items["theme"].str.replace(",", "") + " " + items["type"])
        vectoriser = TfidfVectorizer(stop_words="english", min_df=2, max_features=3000)
        self.item_tfidf = vectoriser.fit_transform(text.fillna(""))
        return self

    @staticmethod
    def _max_normalise(scores):
        top = scores.max() if scores.size else 0
        return scores / top if top > 0 else scores

    def score(self, history, weights=None):
        history = np.asarray(history, dtype=int)
        n = self.cf_similarity.shape[0]
        vector = np.zeros(n)
        vector[history] = 1.0 if weights is None else np.asarray(weights, dtype=float)
        cf = vector @ self.cf_similarity
        profile = self.item_tfidf.T @ vector
        cb = np.asarray(self.item_tfidf @ profile).ravel()
        return self.alpha * self._max_normalise(cf) + (1 - self.alpha) * self._max_normalise(cb)
