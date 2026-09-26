"""Worked examples and timings for the evaluation chapter.

    python -m edurec.examples

The examples use the training split (each learner's last video hidden), with the final model's
settings from models/model_card.json, so I can show what each model recommended and whether the real
next video was in it.
"""

import json
import time

import numpy as np

from .config import RESULTS_DIR, SEED
from .data import load_dataset, prototype_strength
from .models import PathwayHybrid, PopularityRecommender, PrototypeHybrid
from .split import histories_before, temporal_split
from .train import CARD_PATH, build_parts, chosen_settings, train_and_save


def describe(catalogue, index):
    info = catalogue.describe(int(index))
    return f"{info['name']} [{info['software'] or '-'}]"


def case_report(model, prototype, catalogue, user, history, weights, target, k=5):
    final_top = model.recommend(history, k=k)
    proto_top = prototype.recommend(history, k=k, weights=weights)
    reasons = model.explain(history, final_top, catalogue)
    return {
        "learner": int(user),
        "videos_before": int(len(history)),
        "recent_history": [describe(catalogue, i) for i in history[-3:]],
        "real_next_video": describe(catalogue, target),
        "final_model": [{"video": describe(catalogue, i), "reason": r["reason"], "is_real_next": bool(i == target)}
                        for i, r in zip(final_top, reasons)],
        "june_prototype": [{"video": describe(catalogue, i), "is_real_next": bool(i == target)} for i in proto_top],
    }


def feature_example(parts, catalogue, video_id=510, n=5):
    """What each signal "sees" for one video: its text and nearest videos by text, its item-kNN neighbours and
    what follows it."""
    index = catalogue.id_to_index[video_id]
    content = parts["content"]
    row = content.vectors[index]
    order = np.argsort(-row.data)[:8]
    similar_text = np.argsort(-content.similarity[index])[:n]
    neighbours = np.argsort(-parts["collaborative"].similarity[index])[:n]
    following = np.argsort(-parts["pathway"].transitions[index])[:n]
    return {
        "video": describe(catalogue, index),
        "text_start": " ".join(catalogue.items.at[index, "text"].split()[:30]),
        "top_tfidf_terms": [{"term": str(content.terms[row.indices[k]]), "weight": round(float(row.data[k]), 3)}
                            for k in order],
        "content_neighbours": [{"video": describe(catalogue, j), "cosine": round(float(content.similarity[index, j]), 3)}
                               for j in similar_text],
        "item_knn_neighbours": [{"video": describe(catalogue, j),
                                 "cosine": round(float(parts["collaborative"].similarity[index, j]), 3)}
                                for j in neighbours],
        "opened_next": [{"video": describe(catalogue, j),
                         "probability": round(float(parts["pathway"].transitions[index, j]), 3)} for j in following],
    }


def main():
    data = load_dataset()
    card = json.load(open(CARD_PATH)) if CARD_PATH.exists() else train_and_save()
    decays = chosen_settings()
    weights = {g: [card["weights_by_history_group"][g][c] for c in ("collaborative", "content", "pathway")]
               for g in card["weights_by_history_group"]}
    _, _, train, test = temporal_split(data.pairs)
    parts = build_parts(train, data.catalogue, decays)
    popularity = PopularityRecommender().fit(train, data.catalogue)
    model = PathwayHybrid(**parts, weights=weights, fallback=popularity)
    strength = prototype_strength(data.pairs)
    prototype = PrototypeHybrid().fit(train, data.catalogue, values=strength[train.index.to_numpy()])
    histories, history_weights = histories_before(train, test, strength)
    targets = test["item_index"].to_numpy()
    users = test["user_id"].to_numpy()

    ranks = []
    for history, target in zip(histories, targets):
        top = model.recommend(history, k=10)
        ranks.append(int(np.where(top == target)[0][0]) + 1 if target in top else None)
    proto_hits = [t in prototype.recommend(h, k=10, weights=w) for h, w, t in zip(histories, history_weights, targets)]

    rng = np.random.default_rng(SEED)
    software = data.catalogue.items["software"].to_numpy()

    def same_software(i):
        return software[targets[i]] != "" and software[targets[i]] == software[histories[i][-1]]

    def jumped(i):
        return "" not in (software[targets[i]], software[histories[i][-1]]) and not same_software(i)

    short = [i for i in range(len(targets)) if 2 <= len(histories[i]) <= 4]
    train_items = set(train["item_index"])
    # How each example was picked. The first three are drawn at random, with a fixed seed, from the cases
    # that fit the rule.
    rules = {
        "success": "real next video ranked first by the final model and missed by the prototype, 2-4 earlier videos",
        "same_topic_miss": "missed by the final model although the next video is about the same software",
        "topic_jump": "missed by the final model, and the next video is about different software",
        "brand_new_video": "the first test case whose video never appears in the training data",
        "heaviest_learner": "the learner with the longest history",
    }
    picks = {
        "success": int(rng.choice([i for i in short if ranks[i] == 1 and not proto_hits[i]])),
        "same_topic_miss": int(rng.choice([i for i in short if ranks[i] is None and same_software(i)])),
        "topic_jump": int(rng.choice([i for i in short if ranks[i] is None and jumped(i)])),
        "brand_new_video": [i for i, t in enumerate(targets) if t not in train_items][0],
        "heaviest_learner": int(np.argmax([len(h) for h in histories])),
    }
    examples = {name: {"picked_because": rules[name],
                       **case_report(model, prototype, data.catalogue, users[i], histories[i], history_weights[i],
                                     targets[i])}
                for name, i in picks.items()}
    examples["feature_example"] = feature_example(build_parts(data.pairs, data.catalogue, decays), data.catalogue)

    # How far away was the real next video when the final model missed? Same software or a jump?
    misses = [i for i, r in enumerate(ranks) if r is None]
    same_software_miss = np.mean([same_software(i) for i in misses])
    hits = [i for i, r in enumerate(ranks) if r is not None]
    same_software_hit = np.mean([same_software(i) for i in hits])
    examples["miss_analysis"] = {
        "misses": len(misses), "hits": len(hits),
        "share_of_misses_with_same_software_as_last_video": float(same_software_miss),
        "share_of_hits_with_same_software_as_last_video": float(same_software_hit),
    }

    timings = []
    sample = rng.choice(len(histories), 500, replace=False)
    for i in sample:
        started = time.perf_counter()
        top = model.recommend(histories[i], k=10)
        model.explain(histories[i], top, data.catalogue)
        timings.append((time.perf_counter() - started) * 1000)
    started = time.perf_counter()
    build_parts(train, data.catalogue, decays)
    fit_seconds = time.perf_counter() - started
    examples["timing"] = {"fit_seconds_on_training_split": round(fit_seconds, 2),
                          "recommend_and_explain_ms_mean": round(float(np.mean(timings)), 2),
                          "recommend_and_explain_ms_p95": round(float(np.percentile(timings, 95)), 2)}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(examples, open(RESULTS_DIR / "examples.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(examples, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
