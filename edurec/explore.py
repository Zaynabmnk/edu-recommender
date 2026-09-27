"""Exploring MARS before designing anything.

    python -m edurec.explore

Prints the numbers I quote in the report, saves them to results/dataset_stats.json, and draws the
four-panel dataset figure. Every number about the data in my report comes from here, so it can be
checked and re-run.
"""

import json

from pathlib import Path

import numpy as np

from . import plotstyle
from .config import FIGURES_DIR, RESULTS_DIR
from .data import load_dataset
from .text import GluedWordSplitter, word_counts

SYSTEM_DICTIONARY = Path("/usr/share/dict/words")


def repair_check(items):
    """How many distinct tokens the subtitle repair splits, and how many of those are real dictionary words.

    A repaired token that is a real word is probably a false repair, so this is a rough error rate.
    The check needs the word list that ships with macOS and Linux, and is skipped without it.
    """
    counts = word_counts(items["description_raw"])
    splitter = GluedWordSplitter(counts)
    repaired = [token for token in counts if splitter.split_token(token)]
    if not SYSTEM_DICTIONARY.exists():
        return len(repaired), None
    dictionary = {line.strip().lower() for line in SYSTEM_DICTIONARY.open()}
    return len(repaired), sum(token in dictionary for token in repaired) / len(repaired)


def dataset_stats(data):
    pairs, items, raw = data.pairs, data.catalogue.items, data.raw
    per_learner = pairs.groupby("user_id").size()
    item_use = pairs.groupby("item_index").size().sort_values(ascending=False)
    top_tenth = item_use.head(int(len(item_use) * 0.1)).sum() / item_use.sum()

    ordered = pairs.sort_values(["user_id", "position"])
    same = ordered["user_id"].to_numpy()[1:] == ordered["user_id"].to_numpy()[:-1]
    gaps = (ordered["first_seen"].diff().dt.total_seconds().to_numpy()[1:])[same]
    prev_items = ordered["item_id"].to_numpy()[:-1][same]
    next_items = ordered["item_id"].to_numpy()[1:][same]
    software = items.set_index("item_id")["software"]
    same_software = (software.reindex(prev_items).to_numpy() == software.reindex(next_items).to_numpy()) & \
                    (software.reindex(prev_items).to_numpy() != "")

    watch = raw["watch"]
    rebuilt_rating = np.clip(np.ceil(watch["watch_percentage"] / 10), 1, 10)
    words = items["description_raw"].str.split().str.len()
    missing = lambda column: float((items[column].fillna("") == "").mean())  # noqa: E731

    users = raw["users"]
    active_jobs = users[users["user_id"].isin(per_learner.index)]["job"].notna().sum()
    repaired_types, in_dictionary = repair_check(items)
    return {
        "learners_registered": int(users["user_id"].nunique()),
        "learners_with_activity": int(per_learner.size),
        "share_registered_without_activity": float(1 - per_learner.size / users["user_id"].nunique()),
        "videos": int(len(items)),
        "videos_ever_opened": int(item_use.size),
        "videos_never_opened": int(len(items) - item_use.size),
        "watch_event_rows": int(len(watch)),
        "page_view_rows": int(len(raw["views"])),
        "learners_with_watch_events": int(watch["user_id"].nunique()),
        "learner_video_pairs": int(len(pairs)),
        "videos_per_learner_median": float(per_learner.median()),
        "videos_per_learner_mean": float(per_learner.mean()),
        "videos_per_learner_max": int(per_learner.max()),
        "learners_with_one_video": int((per_learner == 1).sum()),
        "learners_with_two_or_more": int((per_learner >= 2).sum()),
        "learners_with_20_or_more": int((per_learner >= 20).sum()),
        "matrix_density_active_learners": float(len(pairs) / (per_learner.size * len(items))),
        "share_of_pairs_on_top_10pct_videos": float(top_tenth),
        "consecutive_pairs": int(same.sum()),
        "share_next_within_1_hour": float((gaps < 3600).mean()),
        "share_next_within_1_day": float((gaps < 86400).mean()),
        "median_gap_minutes": float(np.median(gaps) / 60),
        "share_next_is_next_catalogue_id": float((next_items - prev_items == 1).mean()),
        "share_next_same_software": float(same_software.mean()),
        "rating_equals_rescaled_watch_pct": float((rebuilt_rating == watch["rating"]).mean()),
        "share_watch_events_91_to_100pct": float((watch["watch_percentage"] > 90).mean()),
        "median_watch_pct": float(watch["watch_percentage"].median()),
        "missing_description": missing("description_raw"),
        "missing_difficulty": missing("difficulty"),
        "missing_job_tag": missing("job"),
        "missing_software_tag": missing("software"),
        "missing_theme_tag": missing("theme"),
        "median_description_words_by_type": {t: float(v) for t, v in words.groupby(items["type"]).median().items()},
        "videos_by_type": {t: int(v) for t, v in items["type"].value_counts().items()},
        "stuck_words_repaired": int(items["words_repaired"].sum()),
        "videos_with_repairs": int((items["words_repaired"] > 0).sum()),
        "distinct_tokens_repaired": int(repaired_types),
        "repaired_tokens_in_dictionary": in_dictionary,
        "learners_registered_with_job": int(users["job"].notna().sum()),
        "active_learners_with_job": int(active_jobs),
        "first_activity": str(pairs["first_seen"].min()),
        "last_activity": str(pairs["first_seen"].max()),
    }, per_learner, item_use, gaps, watch


def draw(per_learner, item_use, gaps, watch, stats):
    plotstyle.apply()
    plt = plotstyle.plt
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    blue = plotstyle.SERIES[0]

    ax = axes[0, 0]
    labels = ["1", "2", "3", "4", "5-9", "10-19", "20+"]
    bins = [per_learner.eq(1).sum(), per_learner.eq(2).sum(), per_learner.eq(3).sum(), per_learner.eq(4).sum(),
            per_learner.between(5, 9).sum(), per_learner.between(10, 19).sum(), per_learner.ge(20).sum()]
    ax.bar(labels, bins, width=0.6, color=blue)
    ax.set_title("a. Videos opened per learner")
    ax.set_xlabel("videos opened")
    ax.set_ylabel("learners")
    ax.grid(axis="x", visible=False)
    leader = dict(arrowstyle="-", color=plotstyle.INK_MUTED, lw=0.8)
    ax.annotate(f"{bins[0]:,} learners\nopened only one", (0, bins[0]), xytext=(1.2, bins[0] * 0.92),
                fontsize=8, color=plotstyle.INK_SECONDARY, arrowprops=leader)

    ax = axes[0, 1]
    share_videos = np.arange(1, len(item_use) + 1) / len(item_use) * 100
    share_pairs = item_use.cumsum().to_numpy() / item_use.sum() * 100
    ax.plot(share_videos, share_pairs, color=blue)
    ax.plot([0, 100], [0, 100], color=plotstyle.BASELINE, lw=1)
    top = stats["share_of_pairs_on_top_10pct_videos"] * 100
    ax.plot([10], [top], "o", color=blue, markersize=6, markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(f"top 10% of videos:\n{top:.0f}% of all activity", (10, top), xytext=(28, 38), fontsize=8,
                color=plotstyle.INK_SECONDARY, arrowprops=leader)
    ax.set_title("b. Activity is concentrated on few videos")
    ax.set_xlabel("% of videos (most used first)")
    ax.set_ylabel("% of learner and video pairs")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax = axes[1, 0]
    edges = [0, 60, 300, 900, 3600, 86400, 7 * 86400, 30 * 86400, np.inf]
    names = ["<1m", "1-5m", "5-15m", "15-60m", "1-24h", "1-7d", "7-30d", ">30d"]
    counts = np.histogram(gaps, bins=edges)[0] / len(gaps) * 100
    colours = [blue] * 4 + [plotstyle.OTHER] * 4  # the first four bins are "within an hour"
    ax.bar(range(len(names)), counts, width=0.6, color=colours)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_title("c. Time until the next video")
    ax.set_ylabel("% of consecutive videos")
    ax.grid(axis="x", visible=False)
    within = stats["share_next_within_1_hour"] * 100
    ax.text(3.9, max(counts) * 0.78, f"blue: {within:.0f}% within an hour", fontsize=8,
            color=plotstyle.INK_SECONDARY, ha="left")

    ax = axes[1, 1]
    pct = np.histogram(watch["watch_percentage"], bins=[-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])[0]
    ax.bar([str(i) for i in range(1, 11)], pct / pct.sum() * 100, width=0.6, color=blue)
    ax.set_title("d. Watch events by “rating”")
    ax.set_xlabel("rating (= watch percentage in tenths)")
    ax.set_ylabel("% of watch events")
    ax.grid(axis="x", visible=False)
    share_full = stats["share_watch_events_91_to_100pct"] * 100
    ax.set_ylim(0, share_full + 12)
    ax.text(9, share_full + 2, f"{share_full:.0f}%", fontsize=8, color=plotstyle.INK_SECONDARY, ha="center")

    fig.tight_layout(h_pad=1.6, w_pad=1.8)
    plotstyle.save(fig, FIGURES_DIR / "fig_dataset")


def main():
    data = load_dataset()
    stats, per_learner, item_use, gaps, watch = dataset_stats(data)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(RESULTS_DIR / "dataset_stats.json", "w"), indent=2)
    draw(per_learner, item_use, gaps, watch, stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
