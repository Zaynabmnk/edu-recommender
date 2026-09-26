"""The recommendation service that the command line tool and the API both use."""

import time

import joblib
import numpy as np

from .config import bucket_of
from .models import COMPONENTS
from .train import MODEL_PATH


class UnknownVideo(ValueError):
    pass


class RecommendationService:

    def __init__(self, path=MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path}. Run: python -m edurec.train")
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.catalogue = bundle["catalogue"]
        self.histories = bundle["histories"]

    def history_of(self, learner_id: int):
        return self.histories.get(int(learner_id))

    def heaviest_learner(self) -> int:
        return int(max(self.histories, key=lambda learner: len(self.histories[learner])))

    def random_learner(self, min_videos: int = 2) -> int:
        """A random learner with at least a couple of videos, for the demo page's "Random learner" button."""
        candidates = [learner for learner, history in self.histories.items() if len(history) >= min_videos]
        return int(np.random.default_rng().choice(candidates))

    def indices_for(self, video_ids):
        unknown = [v for v in video_ids if int(v) not in self.catalogue.id_to_index]
        if unknown:
            raise UnknownVideo(f"Unknown video ids: {unknown}")
        return self.catalogue.index_of(video_ids)

    def recommend(self, history: np.ndarray, k: int = 10) -> dict:
        started = time.perf_counter()
        history = np.asarray(history, dtype=int)
        top = self.model.recommend(history, k=k)
        scores = self.model.score(history)
        reasons = self.model.explain(history, top, self.catalogue)
        items = []
        for rank, (index, why) in enumerate(zip(top, reasons), start=1):
            items.append({"rank": rank, **self.catalogue.describe(int(index)),
                          "score": round(float(scores[index]), 4), "reason": why["reason"], "signal": why["signal"],
                          "signal_shares": why.get("signal_shares")})
        weights = self.model.weights_for(len(history)) if len(history) else None
        return {
            "history_length": int(len(history)),
            "history_group": bucket_of(len(history)) if len(history) else "new learner",
            "weights": None if weights is None else dict(zip(COMPONENTS, map(float, weights))),
            "recommendations": items,
            "milliseconds": round((time.perf_counter() - started) * 1000, 2),
        }

    def recent_history(self, history: np.ndarray, n: int = 5) -> list:
        return [self.catalogue.describe(int(i)) for i in np.asarray(history, dtype=int)[-n:]]

    def search(self, text: str, limit: int = 10) -> list:
        """Videos whose title contains the text, most popular first, for the demo page's search box."""
        names = self.catalogue.items["name"].str.lower()
        matches = np.flatnonzero(names.str.contains(text.lower(), regex=False).to_numpy())
        popularity = self.catalogue.items["nb_views"].fillna(0).to_numpy()
        ranked = matches[np.argsort(-popularity[matches], kind="stable")]
        return [self.catalogue.describe(int(i)) for i in ranked[:limit]]
