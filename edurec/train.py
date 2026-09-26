"""Fits the final model on all of the data and saves it to models/.

    python -m edurec.train

The settings come from the experiments. For each signal's history decay I use the value the five
folds chose most often. The hybrid weights are tuned again on every validation case (all folds
together), because the deployed model should learn from as much data as possible. The saved bundle
also keeps each learner's history, so the command line tool and the API can look learners up.
"""

import json
import time
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from .config import BUCKETS, MODELS_DIR, RESULTS_DIR, SEED, bucket_of
from .data import learner_sequences, load_dataset
from .models import (COMPONENTS, ContentRecommender, ItemKNN, PathwayHybrid, PathwayRecommender,
                     TrendingRecommender)
from .models.base import minmax_unseen
from .split import histories_before, temporal_split
from .tuning import best_mix, grid_ndcg, simplex_grid

MODEL_PATH = MODELS_DIR / "pathway_hybrid.joblib"
CARD_PATH = MODELS_DIR / "model_card.json"


def chosen_settings():
    """The history decays picked during the experiments (falls back to sensible defaults)."""
    decays = {"collaborative": 0.2, "content": 0.2, "pathway": 0.5}
    settings_file = RESULTS_DIR / "tuned_settings.csv"
    if settings_file.exists():
        settings = pd.read_csv(settings_file)
        for component in COMPONENTS:
            picks = settings.loc[settings["method"] == f"{component} decay", "setting"].astype(float)
            if len(picks):
                decays[component] = Counter(picks).most_common(1)[0][0]
    return decays


def build_parts(train, catalogue, decays):
    return {
        "collaborative": ItemKNN(decay=decays["collaborative"]).fit(train, catalogue),
        "content": ContentRecommender(decay=decays["content"]).fit(train, catalogue),
        "pathway": PathwayRecommender(decay=decays["pathway"]).fit(train, catalogue),
    }


def tune_weights_on_all_validation(data, decays, min_cases=20):
    """Per-group weights tuned on every validation case, as in the experiments but without folds."""
    train_v, val, _, _ = temporal_split(data.pairs)
    parts = build_parts(train_v, data.catalogue, decays)
    histories, _ = histories_before(train_v, val)
    targets = val["item_index"].to_numpy()
    grid = simplex_grid(3, 0.1)
    ndcg = np.column_stack([
        grid_ndcg(np.vstack([minmax_unseen(parts[c].score(h), h) for c in COMPONENTS]), t, h, grid)
        for h, t in zip(histories, targets)])
    groups = np.array([bucket_of(len(h)) for h in histories])
    everyone = np.ones(len(targets), dtype=bool)
    overall = best_mix(ndcg, everyone, grid)
    weights = {}
    for bucket in BUCKETS:
        mask = groups == bucket
        weights[bucket] = best_mix(ndcg, mask, grid) if mask.sum() >= min_cases else overall
    return weights, overall


def train_and_save():
    start = time.time()
    data = load_dataset()
    decays = chosen_settings()
    weights, overall = tune_weights_on_all_validation(data, decays)
    parts = build_parts(data.pairs, data.catalogue, decays)
    fallback = TrendingRecommender(window_days=30).fit(data.pairs, data.catalogue)
    model = PathwayHybrid(**parts, weights=weights, fallback=fallback)
    fit_seconds = time.time() - start

    # Store the big matrices as float32 to keep the saved file small.
    for name in ("collaborative", "content"):
        parts[name].similarity = parts[name].similarity.astype(np.float32)
    parts["pathway"].transitions = parts["pathway"].transitions.astype(np.float32)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "catalogue": data.catalogue, "histories": learner_sequences(data.pairs)}
    joblib.dump(bundle, MODEL_PATH, compress=3)

    card = {
        "model": model.name,
        "trained_on": {"learners": int(data.pairs["user_id"].nunique()), "videos": int(data.catalogue.n_items),
                       "learner_video_pairs": int(len(data.pairs)),
                       "last_activity": str(data.pairs["first_seen"].max())},
        "content_text": "TF-IDF over the title, tags and first 300 words of the cleaned subtitles",
        "history_decays": decays,
        "weights_by_history_group": {b: dict(zip(COMPONENTS, map(float, w))) for b, w in weights.items()},
        "single_weight_mix": dict(zip(COMPONENTS, map(float, overall))),
        "new_learner_fallback": "trending over the last 30 days of activity",
        "seed": SEED,
        "fit_seconds": round(fit_seconds, 2),
        "file_megabytes": round(MODEL_PATH.stat().st_size / 1e6, 2),
    }
    json.dump(card, open(CARD_PATH, "w"), indent=2)
    return card


if __name__ == "__main__":
    print(json.dumps(train_and_save(), indent=2))
