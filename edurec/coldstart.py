"""Brand-new learners: what should the system show before a learner has opened anything?

    python -m edurec.coldstart

For every learner I take the very first video they opened and ask each strategy to guess it, using
only activity from before that moment. I also include all-time popularity counted over the whole
dataset. That version quietly uses the future, and comparing it with the honest version shows how
much this kind of leakage can flatter a model.
"""

import numpy as np
import pandas as pd

from .config import RESULTS_DIR
from .data import load_dataset
from .metrics import case_metrics, target_position
from .stats import bootstrap_ci


def _counts(items, n_items):
    return np.bincount(items, minlength=n_items).astype(float)


def run(windows=(7, 30, 90), k=10):
    data = load_dataset()
    pairs = data.pairs.sort_values("first_seen")
    n = data.catalogue.n_items
    times = pairs["first_seen"].to_numpy()
    items = pairs["item_index"].to_numpy()
    job_of = data.users.set_index("user_id")["job"]
    learner_jobs = pairs["user_id"].map(job_of).to_numpy()
    firsts = pairs[pairs["position"] == 0]
    all_time = _counts(items, n)

    rows = []
    for _, case in firsts.iterrows():
        t = np.datetime64(case["first_seen"])
        before = times < t
        past = _counts(items[before], n)
        tiebreak = past / (past.max() + 1.0)
        strategies = {
            "All-time popularity (uses future data)": all_time,
            "Popularity before the first visit": past,
        }
        for days in windows:
            recent = before & (times >= t - np.timedelta64(days, "D"))
            strategies[f"Trending, last {days} days"] = _counts(items[recent], n) + tiebreak
        job = job_of.get(case["user_id"])
        if isinstance(job, str):
            same_job = before & (learner_jobs == job)
            strategies["Same-job popularity"] = _counts(items[same_job], n) + tiebreak
        for name, scores in strategies.items():
            above, tied, candidates = target_position(scores, int(case["item_index"]), np.array([], dtype=int))
            m = case_metrics(above, tied, candidates, ks=(k,))
            rows.append({"strategy": name, "user_id": case["user_id"], "has_job": isinstance(job, str),
                         f"hr@{k}": m[f"hr@{k}"], f"ndcg@{k}": m[f"ndcg@{k}"]})
    per_case = pd.DataFrame(rows)

    summary = []
    for (name, subset), frame in [((s, "all learners"), f) for s, f in per_case.groupby("strategy")] + \
            [((s, "learners with a job"), f[f["has_job"]]) for s, f in per_case.groupby("strategy")]:
        if frame.empty:
            continue
        low, high = bootstrap_ci(frame[f"hr@{k}"].to_numpy())
        summary.append({"strategy": name, "learners": subset, "cases": len(frame),
                        f"hr@{k}": frame[f"hr@{k}"].mean(), "hr_ci_low": low, "hr_ci_high": high,
                        f"ndcg@{k}": frame[f"ndcg@{k}"].mean()})
    summary = pd.DataFrame(summary).drop_duplicates(["strategy", "learners", "cases"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "coldstart.csv", index=False)
    return summary


if __name__ == "__main__":
    print(run().round(3).to_string(index=False))
