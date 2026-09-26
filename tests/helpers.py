"""Small made-up datasets for the tests, so they run in seconds and I know the right answers."""

import pandas as pd

from edurec.data import build_catalogue, build_interactions


def tiny_items():
    return pd.DataFrame({
        "item_id": [10, 11, 12, 13, 14, 15],
        "name": ["Excel basics", "Excel formulas", "Excel charts", "Teams meetings", "Teams chat", "Outlook calendar"],
        "description": [
            "learn excel cells and rows", "excel formulas sum and average", "excel charts and graphs",
            "schedule a teams meeting", "chat with your team in teams", "outlook calendar and meetings",
        ],
        "Difficulty": ["Beginner", "Intermediate", None, "Beginner", None, "Beginner"],
        "Job": ["[]", "['Accounting']", "[]", "[]", "[]", "[]"],
        "Software": ["['Excel']", "['Excel']", "['Excel']", "['Microsoft Teams']", "['Microsoft Teams']",
                     "['Outlook']"],
        "Theme": ["['Produce']", "['Produce']", "['Produce']", "['Communicate']", "['Communicate']", "['Organize']"],
        "type": ["tutorial"] * 5 + ["webcast"],
        "duration": [100, 120, 110, 90, 95, 1800],
        "nb_views": [50, 40, 30, 60, 20, 10],
        "created_at": ["2019-01-01"] * 6,
    })


def tiny_events():
    """Learners follow the Excel series in order, or the Teams pair, plus a few repeat views."""
    views = pd.DataFrame([
        (1, 10, "2020-01-01 10:00"), (1, 11, "2020-01-01 10:05"), (1, 12, "2020-01-01 10:10"),
        (2, 10, "2020-01-02 09:00"), (2, 11, "2020-01-02 09:03"), (2, 12, "2020-01-02 09:08"),
        (3, 10, "2020-01-03 12:00"), (3, 11, "2020-01-03 12:02"),
        (4, 13, "2020-01-04 08:00"), (4, 14, "2020-01-04 08:10"), (4, 14, "2020-01-05 08:10"),
        (5, 13, "2020-01-05 15:00"), (5, 14, "2020-01-05 15:05"),
        (6, 15, "2020-01-06 11:00"),
    ], columns=["user_id", "item_id", "created_at"])
    watch = pd.DataFrame([
        (1, 10, 100, 10, "2020-01-01 10:01"),
        (4, 13, 45, 5, "2020-01-04 08:02"),
    ], columns=["user_id", "item_id", "watch_percentage", "rating", "created_at"])
    for frame in (views, watch):
        frame["created_at"] = pd.to_datetime(frame["created_at"])
    return views, watch


def tiny_dataset():
    catalogue = build_catalogue(tiny_items())
    views, watch = tiny_events()
    pairs = build_interactions(watch, views, catalogue)
    return catalogue, pairs
