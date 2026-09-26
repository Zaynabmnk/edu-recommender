"""A stricter check with a single cut-off date.

    python -m edurec.timesplit

My main protocol hides each learner's own last video, but the models can still learn from other
learners' later activity (Ji et al., 2023). Here nothing after the cut-off date is used for training at
all. The models are trained on activity before 1 July 2019 and then tested the way they would be used
after that date:

- newcomers: learners whose first video came after the cut-off. The model sees their first video and
  has to guess their second.
- returning learners: learners with history before the cut-off who came back after it. The model sees
  their old history and has to guess their first new video.

All the settings (history decays and hybrid weights) are tuned again using only activity from before the
cut-off, in the same way as the main experiment, so nothing from after the cut-off is used anywhere.

Many videos had no activity before the cut-off, and only the content signal can recommend those. So I
also test a simple rule that needs no tuning: keep the tenth place in the list for the best-matching
video that nobody has opened yet (new_video_slot). I test it after the cut-off, on an earlier period
(trained before January 2019, newcomers from January to June 2019) and in the main protocol.
"""

import json

import numpy as np
import pandas as pd

from .config import BUCKETS, RESULTS_DIR, bucket_of
from .data import load_dataset, prototype_strength
from .experiments import DECAYS
from .metrics import case_metrics, target_position
from .models import (COMPONENTS, ContentRecommender, ItemKNN, PathwayHybrid, PathwayRecommender,
                     PopularityRecommender, PrototypeHybrid)
from .models.base import minmax_unseen
from .split import histories_before, temporal_split
from .stats import bootstrap_ci, mcnemar_exact
from .train import CARD_PATH, train_and_save
from .tuning import best_mix, expected_hit_and_ndcg, grid_ndcg, simplex_grid

CUTOFF = "2019-07-01"
EARLIER_START = "2019-01-01"


def fit_parts(train, catalogue, decays):
    return {
        "collaborative": ItemKNN(decay=decays["collaborative"]).fit(train, catalogue),
        "content": ContentRecommender(decay=decays["content"]).fit(train, catalogue),
        "pathway": PathwayRecommender(decay=decays["pathway"]).fit(train, catalogue),
    }


def tune_before(pairs, catalogue, cutoff):
    """Decays and per-group weights tuned only on activity before the cut-off.

    Each learner's last video before the cut-off is the validation case and everything earlier is used
    for fitting, as in the main experiment (without the folds, since nothing here is tested on these
    learners' pre-cut-off videos).
    """
    before = pairs[pairs["first_seen"] < cutoff].copy()
    before["history_length"] = before.groupby("user_id")["item_id"].transform("size")
    _, _, fit_part, validation = temporal_split(before)
    histories, _ = histories_before(fit_part, validation)
    targets = validation["item_index"].to_numpy()
    models = fit_parts(fit_part, catalogue, {c: 1.0 for c in COMPONENTS})

    def rows(component, decay):
        models[component].decay = decay
        return [minmax_unseen(models[component].score(h), h) for h in histories]

    def mean_ndcg(scored):
        positions = np.array([target_position(s, t, h)[:2] for s, t, h in zip(scored, targets, histories)])
        return expected_hit_and_ndcg(positions[:, 0], positions[:, 1], 10)[1].mean()

    decays = {c: max(DECAYS[c], key=lambda d, c=c: mean_ndcg(rows(c, d))) for c in COMPONENTS}
    stack = np.stack([np.vstack(rows(c, decays[c])) for c in COMPONENTS], axis=1)
    grid = simplex_grid(3, 0.1)
    ndcg_by_mix = np.column_stack([grid_ndcg(stack[i].astype(float), t, h, grid)
                                   for i, (t, h) in enumerate(zip(targets, histories))])
    groups = np.array([bucket_of(len(h)) for h in histories])
    overall = best_mix(ndcg_by_mix, np.ones(len(targets), dtype=bool), grid)
    weights = {b: (best_mix(ndcg_by_mix, groups == b, grid) if (groups == b).sum() >= 20 else overall).tolist()
               for b in BUCKETS}
    return decays, weights


def cases_between(pairs, start, end):
    """(group, learner, history, target, history row labels) for learners active between two dates."""
    first_ever = pairs.groupby("user_id")["first_seen"].min()
    rows = []
    for user, group in pairs.sort_values(["user_id", "position"]).groupby("user_id"):
        before = group[group["first_seen"] < start]
        after = group[(group["first_seen"] >= start) & (group["first_seen"] < end)]
        if start <= first_ever[user] < end and len(after) >= 2:
            rows.append(("newcomer", user, after["item_index"].to_numpy()[:1], int(after["item_index"].iloc[1]),
                         after.index[:1]))
        elif len(before) and len(after):
            rows.append(("returning", user, before["item_index"].to_numpy(), int(after["item_index"].iloc[0]),
                         before.index))
    return rows


def hit(scores, target, history):
    above, tied, candidates = target_position(scores, target, history)
    return case_metrics(above, tied, candidates, ks=(10,))


def compare_models(pairs, catalogue, decays, weights):
    cutoff = pd.Timestamp(CUTOFF)
    train = pairs[pairs["first_seen"] < cutoff]
    parts = fit_parts(train, catalogue, decays)
    strength = prototype_strength(pairs)
    popularity = PopularityRecommender().fit(train, catalogue)
    models = {
        "Pathway hybrid (adaptive)": PathwayHybrid(**parts, weights=weights, fallback=popularity),
        "Learning path (Markov)": parts["pathway"],
        "Item-kNN CF": parts["collaborative"],
        "Content (TF-IDF)": parts["content"],
        "Prototype hybrid (June)": PrototypeHybrid().fit(train, catalogue, values=strength[train.index.to_numpy()]),
        "Popularity": popularity,
    }
    known = set(train["item_index"])
    rows = []
    for case, (group, user, history, target, labels) in enumerate(cases_between(pairs, cutoff, pd.Timestamp.max)):
        for name, model in models.items():
            m = hit(model.score(history, strength[labels]), target, history)
            rows.append({"case": case, "group": group, "method": name, "new_video": target not in known,
                         "hr@10": m["hr@10"], "ndcg@10": m["ndcg@10"]})
    per_case = pd.DataFrame(rows)
    summary = []
    for (group, method), frame in per_case.groupby(["group", "method"]):
        low, high = bootstrap_ci(frame["hr@10"].to_numpy())
        summary.append({
            "group": group, "method": method, "cases": len(frame), "hr@10": frame["hr@10"].mean(),
            "hr_ci_low": low, "hr_ci_high": high, "ndcg@10": frame["ndcg@10"].mean(),
            "share_new_videos": frame["new_video"].mean(),
            "hr@10_existing_videos": frame.loc[~frame["new_video"], "hr@10"].mean(),
            "hr@10_new_videos": frame.loc[frame["new_video"], "hr@10"].mean(),
        })
    summary = pd.DataFrame(summary).sort_values(["group", "hr@10"], ascending=[True, False])
    return summary, paired_tests(per_case)


def paired_tests(per_case, final="Pathway hybrid (adaptive)"):
    """Exact McNemar test on hits, the final model against each other method, as in the main experiment."""
    hits = per_case.pivot(index="case", columns="method", values="hr@10") >= 0.5
    groups = per_case.drop_duplicates("case").set_index("case")["group"]
    rows = []
    for group in sorted(groups.unique()):
        part = hits[groups == group]
        for method in part.columns.drop(final):
            rows.append({"group": group, "compared_with": method,
                         "only_final_hit": int((part[final] & ~part[method]).sum()),
                         "only_other_hit": int((~part[final] & part[method]).sum()),
                         "mcnemar_p": mcnemar_exact(part[final], part[method])})
    return pd.DataFrame(rows)


def top_ten(model, parts, history, known, reserve_slot):
    scores = np.array(model.score(history), dtype=float)
    scores[history] = -np.inf
    top = np.argsort(-scores, kind="stable")[:10]
    if reserve_slot and known[top].all():
        unopened = np.flatnonzero(~known)
        unopened = unopened[~np.isin(unopened, history)]
        if len(unopened):
            content = minmax_unseen(parts["content"].score(history), history)
            top[-1] = unopened[np.argmax(content[unopened])]
    return top


def slot_check(pairs, catalogue, label, train, cases, decays, weights):
    """HR@10 with and without a place kept for a video with no training activity."""
    parts = fit_parts(train, catalogue, decays)
    model = PathwayHybrid(**parts, weights=weights, fallback=PopularityRecommender().fit(train, catalogue))
    known = np.zeros(catalogue.n_items, dtype=bool)
    known[train["item_index"].unique()] = True
    rows = []
    for history, target in cases:
        for reserve in (False, True):
            hit = target in top_ten(model, parts, history, known, reserve)
            rows.append({"period": label, "slot": reserve, "new_video": not known[target], "hit": hit})
    frame = pd.DataFrame(rows)
    out = []
    for reserve, part in frame.groupby("slot"):
        new = part["new_video"]
        out.append({"period": label, "new_video_slot": reserve, "cases": len(part), "share_new_videos": new.mean(),
                    "hr@10": part["hit"].mean(), "hr@10_existing_videos": part.loc[~new, "hit"].mean(),
                    "hr@10_new_videos": part.loc[new, "hit"].mean() if new.any() else np.nan})
    return out


def newcomer_cases(pairs, start, end):
    return [(history, target) for group, _, history, target, _ in cases_between(pairs, start, end)
            if group == "newcomer"]


def run():
    data = load_dataset()
    pairs, catalogue = data.pairs, data.catalogue
    cutoff, earlier = pd.Timestamp(CUTOFF), pd.Timestamp(EARLIER_START)
    decays, weights = tune_before(pairs, catalogue, cutoff)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"cutoff": CUTOFF, "decays": decays, "weights": weights},
              open(RESULTS_DIR / "time_split_settings.json", "w"), indent=2)
    summary, tests = compare_models(pairs, catalogue, decays, weights)
    summary.to_csv(RESULTS_DIR / "time_split.csv", index=False)
    tests.to_csv(RESULTS_DIR / "time_split_tests.csv", index=False)

    rows = slot_check(pairs, catalogue, "after July 2019", pairs[pairs["first_seen"] < cutoff],
                      newcomer_cases(pairs, cutoff, pd.Timestamp.max), decays, weights)
    early_decays, early_weights = tune_before(pairs, catalogue, earlier)
    rows += slot_check(pairs, catalogue, "January to June 2019", pairs[pairs["first_seen"] < earlier],
                       newcomer_cases(pairs, earlier, cutoff), early_decays, early_weights)
    card = json.load(open(CARD_PATH)) if CARD_PATH.exists() else train_and_save()
    card_weights = {g: [card["weights_by_history_group"][g][c] for c in COMPONENTS] for g in BUCKETS}
    _, _, train, test = temporal_split(pairs)
    histories, _ = histories_before(train, test)
    rows += slot_check(pairs, catalogue, "main protocol", train, list(zip(histories, test["item_index"].to_numpy())),
                       card["history_decays"], card_weights)
    slots = pd.DataFrame(rows)
    slots.to_csv(RESULTS_DIR / "new_video_slot.csv", index=False)
    return summary, slots


if __name__ == "__main__":
    summary, slots = run()
    print(summary.round(3).to_string(index=False))
    print()
    print(slots.round(3).to_string(index=False))
