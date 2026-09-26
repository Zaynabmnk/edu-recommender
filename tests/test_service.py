"""Tests for the saved model and the command line tool. They use the real model from python -m edurec.train."""

import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from edurec import cli
from edurec.service import RecommendationService, UnknownVideo
from edurec.train import MODEL_PATH

HAS_MODEL = MODEL_PATH.exists()


@unittest.skipUnless(HAS_MODEL, "run python -m edurec.train first")
class ServiceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.service = RecommendationService()

    def test_known_learner_gets_new_videos_with_reasons(self):
        history = self.service.history_of(111316)
        result = self.service.recommend(history, k=10)
        ids = [item["item_id"] for item in result["recommendations"]]
        seen = set(self.service.catalogue.item_ids[history].tolist())
        self.assertEqual(len(ids), 10)
        self.assertFalse(set(ids) & seen)
        self.assertTrue(all(item["reason"] for item in result["recommendations"]))
        self.assertAlmostEqual(sum(result["weights"].values()), 1.0)

    def test_new_learner_gets_trending_videos(self):
        result = self.service.recommend(np.array([], dtype=int), k=5)
        self.assertEqual(result["history_group"], "new learner")
        self.assertEqual(len(result["recommendations"]), 5)

    def test_unknown_video_is_rejected(self):
        with self.assertRaises(UnknownVideo):
            self.service.indices_for([1])

    def test_command_line_tool(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = cli.main(["videos", "510", "511", "-k", "3"])
        self.assertEqual(code, 0)
        self.assertIn("Top 3 recommendations", output.getvalue())
        self.assertIn("why:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
