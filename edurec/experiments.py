"""Runs every experiment in my report and writes the results to results/.

    python -m edurec.experiments

1. Each learner's last video is the test case and the one before it is the validation case.
2. Every model is fitted on the training data and scores every validation and test case.
3. Settings (history decays, ALS settings, hybrid weights) are tuned on the validation cases of four
   folds of learners and applied to the test cases of the fifth, so nothing is tuned on the learners
   it is tested on.
4. Per-learner results, summaries, significance tests and breakdowns are saved as CSV/JSON. The worked
   examples come from examples.py.
"""

import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BUCKETS, MAIN_K, N_FOLDS, RESULTS_DIR, SEED, bucket_of
from .data import load_dataset, prototype_strength
from .metrics import (case_metrics, catalogue_coverage, intra_list_diversity, novelty, target_position)
from .models import (COMPONENTS, ContentRecommender, ImplicitALS, ItemKNN, PathwayRecommender, PopularityRecommender,
                     PrototypeHybrid, RandomRecommender)
from .models.base import minmax_unseen
from .split import assign_folds, histories_before, temporal_split
from .stats import bootstrap_ci, holm, mcnemar_exact, wilcoxon_p
from .tuning import best_mix, expected_hit_and_ndcg, grid_ndcg, simplex_grid

FINAL = "Pathway hybrid (adaptive)"
DECAYS = {"collaborative": (1.0, 0.8, 0.6, 0.4, 0.2), "content": (1.0, 0.8, 0.6, 0.4, 0.2),
          "pathway": (0.5, 0.2, 0.05)}
ALS_GRID = [dict(factors=f, regularisation=r, alpha=a)
            for f in (32, 64) for r in (0.1, 1.0, 10.0, 30.0, 100.0) for a in (5.0, 20.0)]


@dataclass
class Cases:
    frame: pd.DataFrame
    histories: list
    weights: list
    targets: np.ndarray


def make_cases(train_part, held_out, folds, strength) -> Cases:
    histories, weights = histories_before(train_part, held_out, strength)
    frame = held_out[["user_id", "item_id", "item_index"]].reset_index(drop=True)
    frame["fold"] = frame["user_id"].map(folds).to_numpy()
    frame["history_length"] = [len(h) for h in histories]
    frame["bucket"] = frame["history_length"].map(bucket_of)
    return Cases(frame, histories, weights, frame["item_index"].to_numpy())


def top_k(scores, history, k=MAIN_K):
    scores = np.array(scores, dtype=float)
    scores[history] = -np.inf
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.lexsort((top, -scores[top]))]


def positions_from_scores(score_rows, cases):
    """Where each held-out video lands, plus the top-10 list, for a list/array of score vectors."""
    rows, tops = [], []
    for scores, history, target in zip(score_rows, cases.histories, cases.targets):
        rows.append(target_position(scores, target, history))
        tops.append(top_k(scores, history))
    return pd.DataFrame(rows, columns=["above", "tied", "candidates"]), tops


def run_model(model, cases):
    scores = (model.score(h, w) for h, w in zip(cases.histories, cases.weights))
    return positions_from_scores(scores, cases)


def ndcg_of(positions):
    return expected_hit_and_ndcg(positions["above"], positions["tied"], MAIN_K)[1]


def pick_per_fold(val_ndcg: dict, val_cases: Cases):
    """For each fold, the config with the best mean validation NDCG on the other four folds."""
    chosen = {}
    for fold in range(N_FOLDS):
        mask = (val_cases.frame["fold"] != fold).to_numpy()
        chosen[fold] = max(val_ndcg, key=lambda config: val_ndcg[config][mask].mean())
    return chosen


def assemble(chosen: dict, test_runs: dict, test_cases: Cases):
    """Build one set of test results where each fold uses its own chosen config."""
    positions = pd.DataFrame(index=range(len(test_cases.targets)), columns=["above", "tied", "candidates"])
    tops = [None] * len(test_cases.targets)
    for fold, config in chosen.items():
        idx = np.flatnonzero((test_cases.frame["fold"] == fold).to_numpy())
        positions.iloc[idx] = test_runs[config][0].iloc[idx].to_numpy()
        for i in idx:
            tops[i] = test_runs[config][1][i]
    return positions.astype(float), tops


def plain_item_knn():
    """Item-kNN with the whole history weighted equally, like Sarwar et al. (2001). ALS has no recency
    weighting either, so this is the fairer comparison for it."""
    model = ItemKNN(decay=1.0)
    model.name = "Item-kNN CF (no recency)"
    return model


class Experiment:

    def __init__(self, log=print):
        self.log = log
        self.data = load_dataset()
        self.catalogue = self.data.catalogue
        pairs = self.data.pairs
        self.strength = prototype_strength(pairs)
        self.train_v, val, self.train, test = temporal_split(pairs)
        self.folds = assign_folds(test)
        self.val = make_cases(self.train_v, val, self.folds, self.strength)
        self.test = make_cases(self.train, test, self.folds, self.strength)
        self.results = {}      # method -> (positions, tops)
        self.settings = []     # rows describing what each fold chose
        self._fitted = {}
        self._rows, self._mixes = {}, {}
        self.grid_rows = []
        self.log(f"{len(test)} test learners, {len(val)} validation cases, {self.catalogue.n_items} videos")

    def fitted(self, key, part, factory, decay=None):
        """Fit each model once per training set. Changing the decay only changes scoring, not fitting."""
        cache_key = (key, id(part))
        if cache_key not in self._fitted:
            self._fitted[cache_key] = factory(part)
        model = self._fitted[cache_key]
        if decay is not None:
            model.decay = decay
        return model

    # ---------- standalone models ----------

    def fixed_model(self, model_factory, fit_kwargs=None):
        model = model_factory().fit(self.train, self.catalogue, **(fit_kwargs or {}))
        self.results[model.name] = run_model(model, self.test)
        return model

    def tuned_model(self, label, configs: dict, fit):
        """configs maps a readable name to settings; fit(settings, train) returns a fitted model."""
        val_ndcg, test_runs = {}, {}
        for config_name, settings in configs.items():
            val_model = fit(settings, self.train_v)
            val_ndcg[config_name] = ndcg_of(run_model(val_model, self.val)[0])
            test_runs[config_name] = run_model(fit(settings, self.train), self.test)
        self.log_grid(label, val_ndcg)
        chosen = pick_per_fold(val_ndcg, self.val)
        self.results[label] = assemble(chosen, test_runs, self.test)
        for fold, config_name in chosen.items():
            self.settings.append({"method": label, "fold": fold, "setting": config_name})
        return val_ndcg, chosen

    # ---------- hybrid ----------

    def component_rows(self, model, cases, decay, scaling="unseen"):
        """One scaled score row per case. "unseen" is my fix; "max" is how my prototype scaled scores."""
        model.decay = decay
        rows = []
        for history in cases.histories:
            scores = model.score(history)
            if scaling == "unseen":
                rows.append(minmax_unseen(scores, history))
            else:
                top = scores.max()
                rows.append(scores / top if top > 0 else scores)
        return np.vstack(rows).astype(np.float32)

    def rows_for(self, split, component, decay, scaling="unseen"):
        key = (split, component, decay, scaling)
        if key not in self._rows:
            cases = self.val if split == "val" else self.test
            self._rows[key] = self.component_rows(self.signal_models[split][component], cases, decay, scaling)
        return self._rows[key]

    def mix_scores(self, name, settings_for_fold, mode="adaptive", allowed=None, record=False):
        """Tune a weight mix on the other folds' validation cases, then score each fold's test cases.

        settings_for_fold(fold) gives the (decays, scaling) to use. mode is "adaptive" (a mix for each
        history group), "global" (one mix for everyone) or "equal" (a third each).
        """
        grid = self.grid
        allowed = np.ones(len(grid), bool) if allowed is None else allowed
        val_bucket = self.val.frame["bucket"].to_numpy()
        test_bucket = self.test.frame["bucket"].to_numpy()
        scores = [None] * len(self.test.targets)
        for fold in range(N_FOLDS):
            decays, scaling = settings_for_fold(fold)
            if (decays, scaling) not in self._mixes:
                stack = lambda split: np.stack([self.rows_for(split, c, d, scaling) for c, d in zip(COMPONENTS, decays)],
                                               axis=1)
                val_stack = stack("val")
                ndcg_by_mix = np.column_stack([grid_ndcg(val_stack[i].astype(float), t, h, grid) for i, (t, h)
                                               in enumerate(zip(self.val.targets, self.val.histories))])
                self._mixes[(decays, scaling)] = (ndcg_by_mix, stack("test"))
            ndcg_by_mix, test_stack = self._mixes[(decays, scaling)]
            train_folds = (self.val.frame["fold"] != fold).to_numpy()
            fold_cases = np.flatnonzero((self.test.frame["fold"] == fold).to_numpy())
            overall = best_mix(ndcg_by_mix, train_folds, grid, allowed)
            for bucket in BUCKETS:
                if mode == "equal":
                    w = np.full(3, 1 / 3)
                elif mode == "global":
                    w = overall
                else:
                    mask = train_folds & (val_bucket == bucket)
                    w = best_mix(ndcg_by_mix, mask, grid, allowed) if mask.sum() >= 20 else overall
                    if record:
                        self.weight_rows.append({"fold": fold, "bucket": bucket, "validation_cases": int(mask.sum()),
                                                 **dict(zip(COMPONENTS, w))})
                for i in fold_cases[test_bucket[fold_cases] == bucket]:
                    scores[i] = w @ test_stack[i]
            if record:
                self.weight_rows.append({"fold": fold, "bucket": "all (global)",
                                         "validation_cases": int(train_folds.sum()), **dict(zip(COMPONENTS, overall))})
        self.results[name] = positions_from_scores(scores, self.test)

    def run_hybrid_family(self):
        self.log("Hybrid: tuning each signal's decay")
        self.signal_models = {}
        for split, part in (("val", self.train_v), ("test", self.train)):
            self.signal_models[split] = {
                "collaborative": self.fitted("itemknn", part, lambda p: ItemKNN().fit(p, self.catalogue)),
                "content": self.fitted("content", part, lambda p: ContentRecommender().fit(p, self.catalogue)),
                "pathway": self.fitted("pathway", part, lambda p: PathwayRecommender().fit(p, self.catalogue)),
            }
        chosen = {}
        for component in COMPONENTS:
            val_ndcg = {d: ndcg_of(positions_from_scores(self.rows_for("val", component, d), self.val)[0])
                        for d in DECAYS[component]}
            self.log_grid(f"{component} signal decay", {f"decay={d}": v for d, v in val_ndcg.items()})
            chosen[component] = pick_per_fold(val_ndcg, self.val)
            for fold, decay in chosen[component].items():
                self.settings.append({"method": f"{component} decay", "fold": fold, "setting": decay})

        def tuned(fold):
            return tuple(chosen[c][fold] for c in COMPONENTS), "unseen"

        self.grid = simplex_grid(3, 0.1)
        self.weight_rows = []
        self.mix_scores(FINAL, tuned, record=True)
        self.mix_scores("Hybrid (one global weight mix)", tuned, mode="global")
        self.mix_scores("Hybrid (equal weights)", tuned, mode="equal")
        for name, column in (("Hybrid without pathway", 2), ("Hybrid without content", 1),
                             ("Hybrid without collaborative", 0)):
            self.mix_scores(name, tuned, allowed=self.grid[:, column] == 0)
        # The two fixes from Section 3.4 of my report, each undone on its own.
        self.mix_scores("Hybrid without recency weighting", lambda fold: ((1.0, 1.0, 1.0), "unseen"))
        self.mix_scores("Hybrid with prototype scaling", lambda fold: (tuned(fold)[0], "max"))
        pd.DataFrame(self.weight_rows).to_csv(RESULTS_DIR / "hybrid_weights.csv", index=False)

    # ---------- a check on my prototype ----------

    def random_holdout_check(self):
        """My June prototype when a random video is held out instead of the latest one, as in my preliminary report."""
        pairs = self.data.pairs
        held = pairs[pairs["history_length"] >= 2].groupby("user_id").sample(1, random_state=SEED)
        train = pairs.drop(held.index)
        histories, weights = histories_before(train, held, self.strength)
        cases = Cases(held.reset_index(drop=True), histories, weights, held["item_index"].to_numpy())
        rows = []
        prototype = PrototypeHybrid().fit(train, self.catalogue, values=self.strength[train.index.to_numpy()])
        for model in (prototype, PopularityRecommender().fit(train, self.catalogue)):
            positions, _ = run_model(model, cases)
            hit, ndcg = expected_hit_and_ndcg(positions["above"], positions["tied"], MAIN_K)
            rows.append({"method": model.name, "hold_out": "random video", "learners": len(held),
                         "hr@10": hit.mean(), "ndcg@10": ndcg.mean()})
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "random_holdout.csv", index=False)

    # ---------- everything ----------

    def log_grid(self, method, val_ndcg):
        for setting, values in val_ndcg.items():
            self.grid_rows.append({"method": method, "setting": setting,
                                   "validation_ndcg@10": float(np.mean(values))})

    def run(self):
        start = time.time()
        self.log("Baselines")
        self.fixed_model(RandomRecommender)
        self.fixed_model(PopularityRecommender)
        train_strength = self.strength[self.train.index.to_numpy()]
        self.fixed_model(PrototypeHybrid, {"values": train_strength})
        self.fixed_model(plain_item_knn)
        self.tuned_model("Item-kNN CF", {f"decay={d}": d for d in DECAYS["collaborative"]},
                         lambda d, part: self.fitted("itemknn", part, lambda p: ItemKNN().fit(p, self.catalogue), d))
        self.tuned_model("Content (TF-IDF)", {f"decay={d}": d for d in DECAYS["content"]},
                         lambda d, part: self.fitted("content", part,
                                                     lambda p: ContentRecommender().fit(p, self.catalogue), d))
        self.tuned_model("Learning path (Markov)", {f"decay={d}": d for d in DECAYS["pathway"]},
                         lambda d, part: self.fitted("pathway", part,
                                                     lambda p: PathwayRecommender().fit(p, self.catalogue), d))
        self.log("ALS")
        self.tuned_model("ALS matrix factorisation",
                         {f"f={s['factors']} reg={s['regularisation']} alpha={s['alpha']}": s for s in ALS_GRID},
                         lambda s, part: ImplicitALS(**s).fit(part, self.catalogue))
        self.log("Hybrid family")
        self.run_hybrid_family()
        self.log("Random hold-out check")
        self.random_holdout_check()
        self.save()
        self.log(f"Finished in {time.time() - start:.0f}s")

    def per_learner(self):
        frames = []
        for method, (positions, _) in self.results.items():
            metrics = pd.DataFrame([case_metrics(int(a), int(t), int(c)) for a, t, c in positions.to_numpy()])
            cases = self.test.frame.assign(method=method)
            frames.append(pd.concat([cases, positions.reset_index(drop=True), metrics], axis=1))
        return pd.concat(frames, ignore_index=True)

    def save(self):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        per = self.per_learner()
        per.to_csv(RESULTS_DIR / "per_learner.csv", index=False)
        pd.DataFrame(self.settings).to_csv(RESULTS_DIR / "tuned_settings.csv", index=False)

        counts = np.bincount(self.train["item_index"], minlength=self.catalogue.n_items).astype(float)
        n_learners = self.train["user_id"].nunique()
        similarity = ContentRecommender().fit(self.train, self.catalogue).similarity
        summary = []
        for method, (positions, tops) in self.results.items():
            rows = per[per["method"] == method]
            record = {"method": method}
            for column in ["hr@5", "hr@10", "hr@20", "ndcg@10", "mrr", "auc", "precision@10", "recall@10"]:
                record[column] = rows[column].mean()
            for column in ["hr@10", "ndcg@10"]:
                low, high = bootstrap_ci(rows[column].to_numpy())
                record[f"{column}_ci_low"], record[f"{column}_ci_high"] = low, high
                record[f"{column}_fold_sd"] = rows.groupby("fold")[column].mean().std()
            record["coverage@10"] = catalogue_coverage(tops, self.catalogue.n_items)
            record["novelty@10"] = novelty(tops, counts, n_learners)
            record["diversity@10"] = intra_list_diversity(tops, similarity)
            summary.append(record)
        summary = pd.DataFrame(summary).sort_values("ndcg@10", ascending=False)
        summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

        final = per[per["method"] == FINAL].sort_values("user_id")
        tests = []
        for method in summary["method"]:
            if method == FINAL:
                continue
            other = per[per["method"] == method].sort_values("user_id")
            diff_ndcg = final["ndcg@10"].to_numpy() - other["ndcg@10"].to_numpy()
            diff_hr = final["hr@10"].to_numpy() - other["hr@10"].to_numpy()
            tests.append({
                "compared_with": method,
                "hr@10_difference": diff_hr.mean(),
                "hr@10_diff_ci_low": bootstrap_ci(diff_hr)[0], "hr@10_diff_ci_high": bootstrap_ci(diff_hr)[1],
                "ndcg@10_difference": diff_ndcg.mean(),
                "ndcg@10_diff_ci_low": bootstrap_ci(diff_ndcg)[0], "ndcg@10_diff_ci_high": bootstrap_ci(diff_ndcg)[1],
                "wilcoxon_p_ndcg": wilcoxon_p(final["ndcg@10"], other["ndcg@10"]),
                "mcnemar_p_hr": mcnemar_exact(final["hr@10"] >= 0.5, other["hr@10"] >= 0.5),
            })
        tests = pd.DataFrame(tests)
        tests["wilcoxon_p_holm"] = holm(tests["wilcoxon_p_ndcg"])
        tests["mcnemar_p_holm"] = holm(tests["mcnemar_p_hr"])
        tests.to_csv(RESULTS_DIR / "significance.csv", index=False)

        segments = []
        for (method, bucket), frame in per.groupby(["method", "bucket"]):
            low, high = bootstrap_ci(frame["hr@10"].to_numpy())
            segments.append({"method": method, "bucket": bucket, "hr@10": frame["hr@10"].mean(), "hr_ci_low": low,
                             "hr_ci_high": high, "ndcg@10": frame["ndcg@10"].mean(), "learners": len(frame)})
        pd.DataFrame(segments).to_csv(RESULTS_DIR / "segments.csv", index=False)
        pd.DataFrame(self.grid_rows).to_csv(RESULTS_DIR / "validation_grid.csv", index=False)

        seen_items = set(self.train["item_index"])
        cold = per[~per["item_index"].isin(seen_items)]
        cold.groupby("method")[["hr@10", "ndcg@10"]].mean().join(cold.groupby("method").size().rename("cases")) \
            .reset_index().to_csv(RESULTS_DIR / "cold_items.csv", index=False)

        curve = []
        for method, (positions, _) in self.results.items():
            for k in range(1, 51):
                hit, _ = expected_hit_and_ndcg(positions["above"], positions["tied"], k)
                curve.append({"method": method, "k": k, "hr": float(np.mean(hit))})
        pd.DataFrame(curve).to_csv(RESULTS_DIR / "hr_curve.csv", index=False)
        json.dump({"seed": SEED,
                   "test_learners": int(len(self.test.targets)), "validation_cases": int(len(self.val.targets))},
                  open(RESULTS_DIR / "run_info.json", "w"), indent=2)


def main():
    Experiment().run()


if __name__ == "__main__":
    main()
