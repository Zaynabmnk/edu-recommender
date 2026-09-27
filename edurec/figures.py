"""Draws the result charts for the report from the CSV files in results/.

    python -m edurec.figures

Colours follow the method, not its rank, so a method has the same colour in every chart. I checked the
series order with a colour-blindness validator; orange and yellow are never placed side by side.
"""

import json

import numpy as np
import pandas as pd

from . import plotstyle
from .config import FIGURES_DIR, MODELS_DIR, RESULTS_DIR

FINAL = "Pathway hybrid (adaptive)"
LABELS = {
    FINAL: "Final model: pathway hybrid, adaptive weights",
    "Hybrid (one global weight mix)": "Pathway hybrid, one weight mix",
    "Hybrid (equal weights)": "Pathway hybrid, equal weights",
    "Learning path (Markov)": "Learning path only (Markov)",
    "Item-kNN CF": "Item kNN collaborative filtering",
    "Item-kNN CF (no recency)": "Item kNN, whole history weighted equally",
    "Prototype hybrid (June)": "June prototype (CF + content)",
    "ALS matrix factorisation": "ALS matrix factorisation",
    "Content (TF-IDF)": "Content only (TF-IDF)",
    "Popularity": "Popularity",
    "Random": "Random",
}
COLOURS = {
    FINAL: plotstyle.SERIES[0],
    "Prototype hybrid (June)": plotstyle.SERIES[1],
    "Learning path (Markov)": plotstyle.SERIES[2],
    "Item-kNN CF": plotstyle.SERIES[3],
    "Content (TF-IDF)": plotstyle.SERIES[4],
}
# Validated order for charts with several series side by side (blue, orange, aqua, yellow).
SERIES_ORDER = [FINAL, "Prototype hybrid (June)", "Learning path (Markov)", "Item-kNN CF"]
MAIN_METHODS = [FINAL, "Hybrid (one global weight mix)", "Hybrid (equal weights)", "Learning path (Markov)",
                "Item-kNN CF", "Item-kNN CF (no recency)", "ALS matrix factorisation", "Prototype hybrid (June)",
                "Content (TF-IDF)", "Popularity", "Random"]


def colour(method):
    return COLOURS.get(method, plotstyle.OTHER)


def results_bars(summary):
    methods = [m for m in MAIN_METHODS if m in set(summary["method"])]
    rows = summary.set_index("method").loc[methods].sort_values("hr@10", ascending=False).iloc[::-1]
    fig, ax = plotstyle.plt.subplots(figsize=(7.2, 0.33 * len(rows) + 0.9))
    y = np.arange(len(rows))
    values = rows["hr@10"].to_numpy()
    errors = np.vstack([values - rows["hr@10_ci_low"], rows["hr@10_ci_high"] - values])
    ax.barh(y, values, height=0.55, color=[colour(m) for m in rows.index])
    ax.errorbar(values, y, xerr=errors, fmt="none", ecolor=plotstyle.INK_SECONDARY, elinewidth=0.9, capsize=2.5)
    for yi, value, high in zip(y, values, rows["hr@10_ci_high"]):
        ax.text(high + 0.008, yi, f"{value:.3f}", va="center", fontsize=8, color=plotstyle.INK_SECONDARY)
    ax.set_yticks(y, [LABELS[m] for m in rows.index])
    ax.set_xlabel("HR@10 on held out learners (bars) with 95% bootstrap interval")
    ax.set_xlim(0, max(rows["hr@10_ci_high"]) + 0.08)
    ax.grid(axis="y", visible=False)
    ax.set_title("Share of learners whose real next video was in the top 10")
    plotstyle.save(fig, FIGURES_DIR / "fig_results")


def hit_curve(curve):
    fig, ax = plotstyle.plt.subplots(figsize=(7.2, 3.2))
    for method in SERIES_ORDER + ["Popularity"]:
        part = curve[curve["method"] == method]
        ax.plot(part["k"], part["hr"], color=colour(method), label=LABELS[method])
        end = part.iloc[-1]
        ax.plot(end["k"], end["hr"], "o", color=colour(method), markersize=5,
                markeredgecolor="white", markeredgewidth=1.5)
    ax.set_xlabel("length of the recommendation list (K)")
    ax.set_ylabel("HR@K")
    ax.set_xlim(0, 52)
    ax.set_ylim(0, None)
    ax.set_title("Hit rate as the list gets longer")
    ax.legend(loc="lower right", ncol=1)
    plotstyle.save(fig, FIGURES_DIR / "fig_hit_curve")


def segments_chart(segments):
    groups = ["1", "2-4", "5-19", "20+"]
    counts = segments[segments["method"] == FINAL].set_index("bucket").loc[groups, "learners"]
    fig, ax = plotstyle.plt.subplots(figsize=(7.2, 3.2))
    width = 0.19
    for offset, method in enumerate(SERIES_ORDER):
        part = segments[segments["method"] == method].set_index("bucket").loc[groups]
        x = np.arange(len(groups)) + (offset - 1.5) * (width + 0.01)
        ax.bar(x, part["hr@10"], width=width, color=colour(method), label=LABELS[method])
        errors = np.vstack([part["hr@10"] - part["hr_ci_low"], part["hr_ci_high"] - part["hr@10"]])
        ax.errorbar(x, part["hr@10"], yerr=errors, fmt="none", ecolor=plotstyle.INK_SECONDARY, elinewidth=0.8,
                    capsize=2)
    ax.set_xticks(np.arange(len(groups)), [f"{g} earlier video{'s' if g != '1' else ''}\n({n} learners)"
                                             for g, n in zip(groups, counts)])
    ax.set_ylabel("HR@10")
    ax.grid(axis="x", visible=False)
    ax.set_title("Results by how many videos a learner had opened before (95% intervals)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    plotstyle.save(fig, FIGURES_DIR / "fig_segments")


def weights_chart(card):
    order = ["pathway", "collaborative", "content"]  # validated adjacent order: aqua, yellow, magenta
    colours = {"pathway": plotstyle.SERIES[2], "collaborative": plotstyle.SERIES[3], "content": plotstyle.SERIES[4]}
    groups = ["1", "2-4", "5-19", "20+"]
    rows = [(f"{g} video{'s' if g != '1' else ''}", card["weights_by_history_group"][g]) for g in groups]
    rows.append(("one mix for everyone", card["single_weight_mix"]))
    fig, ax = plotstyle.plt.subplots(figsize=(7.2, 2.4))
    for yi, (label, weights) in enumerate(rows[::-1]):
        left = 0.0
        for part in order:
            w = weights[part]
            if w > 0:
                ax.barh(yi, w - 0.004, left=left, height=0.55, color=colours[part],
                        label=part if yi == 0 else None)
                if w >= 0.1:
                    ax.text(left + w / 2, yi, f"{w:.1f}", ha="center", va="center", fontsize=8, color=plotstyle.INK)
            left += w
    ax.set_yticks(range(len(rows)), [label for label, _ in rows[::-1]])
    ax.set_xlim(0, 1)
    ax.set_xlabel("weight of each signal (adds up to 1)")
    ax.grid(False)
    ax.set_title("Tuned weights by history length")
    handles, labels = ax.get_legend_handles_labels()
    names = {"pathway": "learning path", "collaborative": "collaborative", "content": "content"}
    ax.legend(handles, [names[l] for l in labels], loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
    plotstyle.save(fig, FIGURES_DIR / "fig_weights")


def coldstart_chart(coldstart):
    rows = coldstart[coldstart["learners"] == "all learners"].sort_values("hr@10")
    fig, ax = plotstyle.plt.subplots(figsize=(7.2, 2.6))
    y = np.arange(len(rows))

    def pick(name):
        if name.startswith("Trending, last 30"):
            return plotstyle.SERIES[0]
        if "future" in name:
            return plotstyle.SERIES[1]
        return plotstyle.OTHER

    values = rows["hr@10"].to_numpy()
    ax.barh(y, values, height=0.55, color=[pick(n) for n in rows["strategy"]])
    ax.errorbar(values, y, xerr=np.vstack([values - rows["hr_ci_low"], rows["hr_ci_high"] - values]),
                fmt="none", ecolor=plotstyle.INK_SECONDARY, elinewidth=0.9, capsize=2.5)
    for yi, value, high in zip(y, values, rows["hr_ci_high"]):
        ax.text(high + 0.006, yi, f"{value:.3f}", va="center", fontsize=8, color=plotstyle.INK_SECONDARY)
    shown = {"All-time popularity (uses future data)": "All time popularity (uses future data)",
             "Same-job popularity": "Same job popularity"}
    names = [shown.get(s, s) for s in rows["strategy"]]
    labels = [f"{s} ({n} learners)" if n != rows["cases"].max() else s for s, n in zip(names, rows["cases"])]
    ax.set_yticks(y, labels)
    ax.set_xlim(0, rows["hr_ci_high"].max() + 0.07)
    ax.set_xlabel("HR@10 for each learner's first video, using only earlier activity")
    ax.grid(axis="y", visible=False)
    ax.set_title("Brand new learners")
    plotstyle.save(fig, FIGURES_DIR / "fig_coldstart")


def main():
    plotstyle.apply()
    summary = pd.read_csv(RESULTS_DIR / "summary.csv")
    results_bars(summary)
    hit_curve(pd.read_csv(RESULTS_DIR / "hr_curve.csv"))
    segments_chart(pd.read_csv(RESULTS_DIR / "segments.csv"))
    weights_chart(json.load(open(MODELS_DIR / "model_card.json")))
    coldstart_chart(pd.read_csv(RESULTS_DIR / "coldstart.csv"))
    print("Figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
