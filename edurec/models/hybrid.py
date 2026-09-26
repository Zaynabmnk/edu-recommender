"""My final model: a pathway-aware hybrid with weights that adapt to how much history a learner has.

Burke (2002) calls this a weighted hybrid: several recommenders score every video and the scores are
combined. Mine combines three signals:
- collaborative: learners who opened the same videos also opened this one (item-kNN)
- content: this video covers similar material (text and tags)
- pathway: learners usually open this video next (Markov model)

Each signal is scaled to 0-1 over the videos the learner has not opened, then mixed with weights. The
weights are not fixed. They are tuned separately for four history-length groups (1, 2-4, 5-19 and 20+
videos), because my experiments showed that the best signal changes with the length of the history.
A learner with no history at all gets the trending videos instead.
"""

import numpy as np

from ..config import BUCKETS, bucket_of
from .base import Recommender, minmax_unseen

COMPONENTS = ("collaborative", "content", "pathway")

REASONS = {
    "collaborative": "Learners who opened “{}” also opened this",
    "content": "Covers similar material to “{}”",
    "pathway": "Often opened next after “{}”",
}


class PathwayHybrid(Recommender):

    def __init__(self, collaborative, content, pathway, weights, fallback=None, name="Pathway hybrid (adaptive)"):
        self.parts = {"collaborative": collaborative, "content": content, "pathway": pathway}
        self.weights = self._check_weights(weights)
        self.fallback = fallback
        self.name = name

    @staticmethod
    def _check_weights(weights):
        """Accept one weight vector for everyone, or one per history-length group."""
        if isinstance(weights, dict):
            missing = [bucket for bucket in BUCKETS if bucket not in weights]
            if missing:
                raise ValueError(f"Missing weights for history groups: {missing}")
            checked = {bucket: np.asarray(weights[bucket], dtype=float) for bucket in BUCKETS}
        else:
            checked = {bucket: np.asarray(weights, dtype=float) for bucket in BUCKETS}
        for bucket, w in checked.items():
            if w.shape != (len(COMPONENTS),) or (w < 0).any() or not np.isclose(w.sum(), 1.0):
                raise ValueError(f"Weights for group {bucket} must be 3 non-negative numbers adding to 1, got {w}")
        return checked

    def weights_for(self, history_length: int) -> np.ndarray:
        return self.weights[bucket_of(history_length)]

    def component_scores(self, history) -> np.ndarray:
        """The three signals for every video, each scaled to 0-1 over the unseen videos."""
        history = np.asarray(history, dtype=int)
        scores = [self.parts[name].score(history) for name in COMPONENTS]
        return np.vstack([minmax_unseen(s, history) for s in scores])

    def score(self, history, weights=None):
        history = np.asarray(history, dtype=int)
        if len(history) == 0:
            if self.fallback is None:
                raise ValueError("A learner with no history needs a fallback model")
            return self.fallback.score(history)
        return self.weights_for(len(history)) @ self.component_scores(history)

    def explain(self, history, items, catalogue) -> list:
        """Give each recommended video a short reason, based on the signal that added the most to it."""
        history = np.asarray(history, dtype=int)
        if len(history) == 0:
            return [{"reason": "Popular with learners over the last month", "signal": "trending",
                     "because_of": None} for _ in items]
        comps = self.component_scores(history)
        w = self.weights_for(len(history))
        explanations = []
        for item in items:
            shares = w * comps[:, item]
            reason = {"reason": "Popular with other learners", "signal": "popularity", "because_of": None}
            # Take the strongest signal that has real evidence for this video. When none of the learner's
            # videos has ever been followed by anything, the learning-path signal only holds its popularity
            # tie-break, so it must not claim the video is "often opened next".
            for k in np.argsort(-shares):
                signal = COMPONENTS[int(k)]
                contributions = self.parts[signal].contributions(history, item)
                if shares[k] > 0 and contributions.max() > 0:
                    anchor = history[int(np.argmax(contributions))]
                    reason = {"reason": REASONS[signal].format(catalogue.name(anchor)), "signal": signal,
                              "because_of": int(catalogue.item_ids[anchor])}
                    break
            reason["signal_shares"] = {name: round(float(s), 4) for name, s in zip(COMPONENTS, shares)}
            explanations.append(reason)
        return explanations
