"""Loading MARS and turning it into the tables my models use.

MARS has two kinds of activity (Hafsa et al., 2023):
- "explicit" ratings, which are really watch events. The rating column is only the watch percentage
  rescaled to 1-10, so I keep the watch percentage and drop the rating.
- "implicit" ratings, which are page views of a video.

I merge both into one row per learner and video, ordered by the first time the learner opened it.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DIR
from .text import GluedWordSplitter, parse_tag_list, word_counts

RAW_FILES = {
    "watch": "explicit_ratings_en.csv",
    "views": "implicit_ratings_en.csv",
    "items": "items_en.csv",
    "users": "users_en.csv",
}


def load_raw(raw_dir=RAW_DIR) -> dict:
    """Read the four English MARS files and parse the dates."""
    raw = {}
    for key, filename in RAW_FILES.items():
        frame = pd.read_csv(Path(raw_dir) / filename)
        if "created_at" in frame.columns:
            frame["created_at"] = pd.to_datetime(frame["created_at"])
        raw[key] = frame
    return raw


class Catalogue:
    """All videos in one fixed order, so every model agrees on which row or column is which video."""

    def __init__(self, items: pd.DataFrame):
        self.items = items.reset_index(drop=True)
        self.item_ids = self.items["item_id"].to_numpy()
        self.id_to_index = {int(item_id): index for index, item_id in enumerate(self.item_ids)}

    @property
    def n_items(self) -> int:
        return len(self.items)

    def index_of(self, item_ids) -> np.ndarray:
        return np.array([self.id_to_index[int(item_id)] for item_id in item_ids], dtype=int)

    def name(self, index: int) -> str:
        return str(self.items.at[index, "name"])

    def describe(self, index: int) -> dict:
        row = self.items.loc[index]
        return {
            "item_id": int(row["item_id"]),
            "name": str(row["name"]),
            "software": row["software"],
            "type": row["type"],
            "difficulty": row["difficulty"],
            "duration_seconds": int(row["duration"]),
        }


def build_catalogue(items_raw: pd.DataFrame) -> Catalogue:
    """Clean the video metadata and build the text that the content model reads."""
    items = items_raw.copy()
    for source, target in (("Software", "software"), ("Theme", "theme"), ("Job", "job")):
        items[target] = items[source].map(parse_tag_list).map(", ".join)
    items["difficulty"] = items["Difficulty"].fillna("")
    items["type"] = items["type"].fillna("")

    descriptions = items["description"].fillna("").astype(str)
    items["description_raw"] = descriptions
    splitter = GluedWordSplitter(word_counts(descriptions))
    repaired = descriptions.map(splitter.repair)
    items["description_clean"] = repaired.map(lambda pair: pair[0])
    items["words_repaired"] = repaired.map(lambda pair: pair[1])
    items["description_display"] = descriptions.map(splitter.repair_keeping_capitals)

    # Title and tags go first. They say what the video is about in a few words, and the content
    # model only reads the first 300 words of each text.
    tags = items[["software", "theme", "job", "difficulty", "type"]].agg(". ".join, axis=1)
    items["text"] = (items["name"].fillna("") + ". " + tags + ". " + items["description_clean"]).str.strip()

    columns = ["item_id", "name", "type", "difficulty", "software", "theme", "job", "duration",
               "nb_views", "created_at", "description_raw", "description_clean", "description_display",
               "words_repaired", "text"]
    return Catalogue(items[columns])


def build_interactions(watch: pd.DataFrame, views: pd.DataFrame, catalogue: Catalogue) -> pd.DataFrame:
    """One row per learner and video, in the order the learner first opened each video."""
    events = pd.concat([
        views[["user_id", "item_id", "created_at"]],
        watch[["user_id", "item_id", "created_at"]],
    ], ignore_index=True)
    events = events[events["item_id"].isin(catalogue.id_to_index)]

    pairs = (events.groupby(["user_id", "item_id"])["created_at"]
             .agg(first_seen="min", last_seen="max").reset_index())
    view_counts = views.groupby(["user_id", "item_id"]).size().rename("n_views")
    best_watch = watch.groupby(["user_id", "item_id"])["watch_percentage"].max().rename("watch_pct")
    pairs = pairs.join(view_counts, on=["user_id", "item_id"]).join(best_watch, on=["user_id", "item_id"])
    pairs["n_views"] = pairs["n_views"].fillna(0).astype(int)

    pairs["item_index"] = catalogue.index_of(pairs["item_id"])
    # Ties on the timestamp are very rare (0.1% of learners); sorting by item id as well keeps the order stable.
    pairs = pairs.sort_values(["user_id", "first_seen", "item_id"]).reset_index(drop=True)
    pairs["position"] = pairs.groupby("user_id").cumcount()
    pairs["history_length"] = pairs.groupby("user_id")["item_id"].transform("size")
    return pairs


def prototype_strength(pairs: pd.DataFrame) -> np.ndarray:
    """The interaction score from my preliminary prototype, kept only so I can re-test the prototype.

    The prototype mixed 70% "rating" with 30% watch percentage. Because the rating is the watch
    percentage on a 1-10 scale, this counted the same thing twice. Page views scored 0.6, plus 0.1
    for each repeat view, capped at 1.
    """
    watch = pairs["watch_pct"]
    rating = np.clip(np.ceil(watch / 10), 1, 10)
    watch_score = (0.7 * rating / 10 + 0.3 * watch.clip(0, 100) / 100).fillna(0)
    view_score = np.where(pairs["n_views"] > 0, np.minimum(0.6 + 0.1 * (pairs["n_views"] - 1), 1.0), 0)
    return np.maximum(watch_score.to_numpy(), view_score)


def learner_sequences(pairs: pd.DataFrame) -> dict:
    """Map each learner to their video indices in the order they opened them."""
    return {user: group["item_index"].to_numpy()
            for user, group in pairs.sort_values(["user_id", "position"]).groupby("user_id")}


@dataclass
class Dataset:
    catalogue: Catalogue
    pairs: pd.DataFrame
    users: pd.DataFrame
    raw: dict


def load_dataset(raw_dir=RAW_DIR) -> Dataset:
    raw = load_raw(raw_dir)
    catalogue = build_catalogue(raw["items"])
    pairs = build_interactions(raw["watch"], raw["views"], catalogue)
    return Dataset(catalogue=catalogue, pairs=pairs, users=raw["users"], raw=raw)
