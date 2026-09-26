import unittest
from collections import Counter

import numpy as np

from edurec.data import prototype_strength
from edurec.text import GluedWordSplitter, add_missing_spaces, parse_tag_list
from tests.helpers import tiny_dataset


class TextTests(unittest.TestCase):

    def test_parse_tag_list(self):
        self.assertEqual(parse_tag_list("['Excel', 'Word']"), ["Excel", "Word"])
        self.assertEqual(parse_tag_list("[]"), [])
        self.assertEqual(parse_tag_list(float("nan")), [])
        self.assertEqual(parse_tag_list("Excel"), ["Excel"])

    def test_missing_spaces_after_punctuation(self):
        self.assertEqual(add_missing_spaces("documents.As a part of Office 365,OneDrive"),
                         "documents. As a part of Office 365, OneDrive")
        self.assertEqual(add_missing_spaces("version 3.5 is out"), "version 3.5 is out")

    def test_glued_words_are_split_but_real_words_are_not(self):
        counts = Counter({"library": 300, "to": 5000, "libraryto": 1, "workplace": 5, "work": 60, "place": 40})
        splitter = GluedWordSplitter(counts)
        self.assertEqual(splitter.split_token("libraryto"), "library to")
        self.assertIsNone(splitter.split_token("workplace"))  # halves are not 20x more common
        self.assertIsNone(splitter.split_token("to"))
        self.assertEqual(splitter.repair("The libraryto store"), ("the library to store", 1))
        # The version shown to learners keeps the capitals.
        self.assertEqual(splitter.repair_keeping_capitals("OneDrive: The Libraryto store"),
                         "OneDrive: The Library to store")


class DataTests(unittest.TestCase):

    def setUp(self):
        self.catalogue, self.pairs = tiny_dataset()

    def test_catalogue_text_starts_with_title_and_tags(self):
        text = self.catalogue.items.loc[0, "text"]
        self.assertTrue(text.startswith("Excel basics. Excel. Produce"))
        self.assertEqual(self.catalogue.index_of([12, 10]).tolist(), [2, 0])

    def test_one_row_per_learner_and_video(self):
        self.assertFalse(self.pairs.duplicated(["user_id", "item_id"]).any())
        self.assertEqual(len(self.pairs), 13)

    def test_first_seen_views_and_watch(self):
        row = self.pairs[(self.pairs.user_id == 4) & (self.pairs.item_id == 14)].iloc[0]
        self.assertEqual(row["n_views"], 2)
        self.assertEqual(str(row["first_seen"]), "2020-01-04 08:10:00")
        watched = self.pairs[(self.pairs.user_id == 1) & (self.pairs.item_id == 10)].iloc[0]
        self.assertEqual(watched["watch_pct"], 100)

    def test_positions_follow_time(self):
        learner = self.pairs[self.pairs.user_id == 1]
        self.assertEqual(learner.sort_values("position")["item_id"].tolist(), [10, 11, 12])
        self.assertTrue((learner["history_length"] == 3).all())

    def test_prototype_strength_matches_the_june_formula(self):
        strength = dict(zip(zip(self.pairs.user_id, self.pairs.item_id), prototype_strength(self.pairs)))
        self.assertAlmostEqual(strength[(1, 10)], 1.0)            # 100% watched: 0.7 * 1 + 0.3 * 1
        self.assertAlmostEqual(strength[(4, 13)], 0.6)            # 45% watched scores 0.485, the view scores 0.6
        self.assertAlmostEqual(strength[(4, 14)], 0.7)            # two views
        self.assertTrue(np.all(prototype_strength(self.pairs) <= 1.0))


if __name__ == "__main__":
    unittest.main()
