"""How I split the data for tuning and testing.

Each learner's most recent video is the test item and the one before it is the validation item.
Everything earlier is training data. This copies how the system would be used: it only knows what a
learner has done so far and has to guess what they open next. My prototype held out a random video
instead, which let the model learn from a learner's later activity.

On top of that I split the learners into five folds. The hybrid weights are always tuned on the
validation items of four folds and tested on the fifth, so the weights are never tuned on the same
learners they are tested on.
"""

import numpy as np
import pandas as pd

from .config import N_FOLDS, SEED, bucket_of


def temporal_split(pairs: pd.DataFrame):
    """Return (train_for_validation, validation, train, test).

    test:       last video of every learner with at least 2 videos
    validation: second-to-last video of every learner with at least 3 videos
    train:      everything except the test videos
    train_for_validation: everything except the test and validation videos
    """
    last = pairs["position"] == pairs["history_length"] - 1
    second_last = pairs["position"] == pairs["history_length"] - 2
    test = pairs[last & (pairs["history_length"] >= 2)]
    validation = pairs[second_last & (pairs["history_length"] >= 3)]
    train = pairs.drop(test.index)
    train_for_validation = train.drop(validation.index)
    return train_for_validation, validation, train, test


def assign_folds(test: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> pd.Series:
    """Give every test learner a fold number, balancing the history-length groups across folds."""
    rng = np.random.default_rng(seed)
    history = test.set_index("user_id")["position"]  # videos seen before the test video
    groups = history.map(bucket_of)
    folds = pd.Series(0, index=history.index, dtype=int)
    for _, members in groups.groupby(groups):
        users = members.index.to_numpy().copy()
        rng.shuffle(users)
        folds.loc[users] = np.arange(len(users)) % n_folds
    return folds.rename("fold")


def histories_before(train: pd.DataFrame, cases: pd.DataFrame, weights: np.ndarray = None):
    """For each held-out case, the learner's earlier videos (oldest first) and optional weights."""
    ordered = train.sort_values(["user_id", "position"])
    by_user = {user: group for user, group in ordered.groupby("user_id")}
    histories, history_weights = [], []
    for user in cases["user_id"]:
        group = by_user[user]
        histories.append(group["item_index"].to_numpy())
        history_weights.append(None if weights is None else weights[group.index.to_numpy()])
    return histories, history_weights
