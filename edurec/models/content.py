"""Content-based filtering over the video text (Lops, De Gemmis and Semeraro, 2011).

Each video becomes a TF-IDF vector of its title, tags and the start of its cleaned subtitles. Two videos
are similar when their vectors point the same way. A learner's score for a video is its similarity to
their history, weighted towards recent videos.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import Recommender, recency_weights


def truncate_words(text: str, max_words) -> str:
    if not max_words:
        return text
    return " ".join(text.split()[:max_words])


class ContentRecommender(Recommender):
    name = "Content (TF-IDF)"

    def __init__(self, decay: float = 1.0, max_words: int = 300):
        # I cut the subtitles at 300 words because half-hour webcasts mention so many topics that
        # their full text overlaps a little with almost every other video.
        self.decay = decay
        self.max_words = max_words

    def fit(self, train, catalogue):
        # Video text only, so fitting this never sees any learner activity.
        texts = [truncate_words(t, self.max_words) for t in catalogue.items["text"]]
        vectoriser = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True)
        vectors = vectoriser.fit_transform(texts)
        self.vectors, self.terms = vectors, vectoriser.get_feature_names_out()
        similarity = (vectors @ vectors.T).toarray()
        np.fill_diagonal(similarity, 0.0)
        self.similarity = similarity
        return self

    def score(self, history, weights=None):
        history = np.asarray(history, dtype=int)
        if len(history) == 0:
            return np.zeros(self.similarity.shape[0])
        return recency_weights(len(history), self.decay) @ self.similarity[history]

    def contributions(self, history, item):
        w = recency_weights(len(history), self.decay)
        return w * self.similarity[np.asarray(history, dtype=int), item]
