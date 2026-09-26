"""Shared pieces for all my recommenders.

Every model scores a learner's history, meaning the list of videos they have opened, oldest first.
Scoring a history instead of a learner ID means the same model can serve a brand-new learner straight
after their first video without retraining.
"""

import numpy as np
from scipy.sparse import csr_matrix


def recency_weights(length: int, decay: float) -> np.ndarray:
    """Weight 1 for the latest video, decay for the one before, decay**2 before that.

    decay = 1 treats the whole history equally, which is what my prototype did.
    Smaller values focus on what the learner opened most recently.
    """
    return decay ** np.arange(length - 1, -1, -1, dtype=float)


def minmax_unseen(scores: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Scale scores to 0-1 using only the videos the learner has not opened yet.

    My prototype divided by the maximum over all videos, including ones already watched. Those
    usually score highest, so the new videos were squashed and the 0.6/0.4 weights did not mean
    what they said.
    """
    scores = np.asarray(scores, dtype=float)
    mask = np.ones(scores.shape[0], dtype=bool)
    mask[seen] = False
    if not mask.any():
        return np.zeros_like(scores)
    low, high = scores[mask].min(), scores[mask].max()
    if high <= low:
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


def learner_item_matrix(train, n_items: int, values=None):
    """Binary learners x videos matrix built from the training pairs."""
    users = np.sort(train["user_id"].unique())
    row_of = {user: row for row, user in enumerate(users)}
    rows = train["user_id"].map(row_of).to_numpy()
    cols = train["item_index"].to_numpy()
    data = np.ones(len(train)) if values is None else np.asarray(values, dtype=float)
    return csr_matrix((data, (rows, cols)), shape=(len(users), n_items)), row_of


class Recommender:
    """Base class. Subclasses fill in fit() and score()."""

    name = "base"

    def fit(self, train, catalogue):
        return self

    def score(self, history: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, history, k: int = 10, weights=None) -> np.ndarray:
        """Indices of the top-k videos the learner has not opened yet, best first."""
        history = np.asarray(history, dtype=int)
        scores = np.array(self.score(history, weights), dtype=float)
        scores[history] = -np.inf
        k = min(k, int(np.isfinite(scores).sum()))
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.lexsort((top, -scores[top]))]
