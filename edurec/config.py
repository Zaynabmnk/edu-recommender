"""Folder locations and the settings I reuse across the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"

SEED = 42
N_FOLDS = 5
K_VALUES = (5, 10, 20)
MAIN_K = 10

# History-length groups. I use them for the adaptive weights and for breaking the results down.
# The edges come from the data: most learners have fewer than 20 videos, and the "20+" group
# holds the small number of very heavy learners that the dataset authors also point out.
BUCKETS = ("1", "2-4", "5-19", "20+")


def bucket_of(history_length: int) -> str:
    """Return the history-length group for a learner who has opened this many videos."""
    if history_length <= 1:
        return "1"
    if history_length <= 4:
        return "2-4"
    if history_length <= 19:
        return "5-19"
    return "20+"
