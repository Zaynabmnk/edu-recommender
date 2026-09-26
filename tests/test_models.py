import unittest

import numpy as np

from edurec.models import (ContentRecommender, ImplicitALS, ItemKNN, PathwayHybrid, PathwayRecommender,
                           PopularityRecommender, PrototypeHybrid, TrendingRecommender)
from edurec.models.base import minmax_unseen, recency_weights
from tests.helpers import tiny_dataset


class HelperTests(unittest.TestCase):

    def test_recency_weights(self):
        self.assertTrue(np.allclose(recency_weights(3, 0.5), [0.25, 0.5, 1.0]))
        self.assertTrue(np.allclose(recency_weights(3, 1.0), [1, 1, 1]))

    def test_minmax_ignores_seen_videos(self):
        scaled = minmax_unseen(np.array([10.0, 2.0, 4.0, 3.0]), np.array([0]))
        self.assertTrue(np.allclose(scaled[1:], [0.0, 1.0, 0.5]))


class ModelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.catalogue, cls.pairs = tiny_dataset()
        cls.idx = cls.catalogue.id_to_index

    def test_popularity_counts_learners(self):
        model = PopularityRecommender().fit(self.pairs, self.catalogue)
        self.assertEqual(model.counts[self.idx[10]], 3)
        self.assertEqual(model.recommend(np.array([self.idx[10]]), k=1).tolist(), [self.idx[11]])

    def test_trending_uses_only_the_window(self):
        model = TrendingRecommender(window_days=2).fit(self.pairs, self.catalogue, now="2020-01-06 00:00")
        # Only learners 4 and 5 (Teams) opened videos in the two days before "now".
        self.assertEqual(model.recommend(np.array([], dtype=int), k=2).tolist(), [self.idx[13], self.idx[14]])

    def test_item_knn_links_videos_opened_together(self):
        model = ItemKNN().fit(self.pairs, self.catalogue)
        self.assertTrue(np.allclose(model.similarity, model.similarity.T))
        self.assertTrue(np.allclose(np.diag(model.similarity), 0))
        top = model.recommend(np.array([self.idx[13]]), k=1)
        self.assertEqual(top.tolist(), [self.idx[14]])

    def test_pathway_learns_what_comes_next(self):
        model = PathwayRecommender().fit(self.pairs, self.catalogue)
        row = model.transitions[self.idx[10]]
        self.assertAlmostEqual(row.sum(), 1.0)
        self.assertEqual(int(np.argmax(row)), self.idx[11])
        self.assertEqual(model.recommend(np.array([self.idx[10], self.idx[11]]), k=1).tolist(), [self.idx[12]])

    def test_content_prefers_the_same_software(self):
        model = ContentRecommender().fit(self.pairs, self.catalogue)
        top = model.recommend(np.array([self.idx[10]]), k=2)
        self.assertEqual(set(top.tolist()), {self.idx[11], self.idx[12]})

    def test_als_ranks_co_watched_videos_higher(self):
        model = ImplicitALS(factors=4, regularisation=0.1, alpha=10, iterations=20).fit(self.pairs, self.catalogue)
        scores = model.score(np.array([self.idx[10], self.idx[11]]))
        self.assertGreater(scores[self.idx[12]], scores[self.idx[15]])

    def test_recommend_never_repeats_history(self):
        model = PopularityRecommender().fit(self.pairs, self.catalogue)
        history = np.array([self.idx[10], self.idx[13]])
        top = model.recommend(history, k=4)
        self.assertFalse(set(top.tolist()) & set(history.tolist()))
        self.assertEqual(len(top), 4)

    def test_prototype_runs_with_weights(self):
        model = PrototypeHybrid().fit(self.pairs, self.catalogue, values=np.ones(len(self.pairs)))
        scores = model.score(np.array([self.idx[10]]), weights=np.array([0.8]))
        self.assertEqual(scores.shape, (self.catalogue.n_items,))
        self.assertLessEqual(scores.max(), 1.0 + 1e-9)


class HybridTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.catalogue, cls.pairs = tiny_dataset()
        cls.idx = cls.catalogue.id_to_index
        cls.parts = dict(
            collaborative=ItemKNN().fit(cls.pairs, cls.catalogue),
            content=ContentRecommender().fit(cls.pairs, cls.catalogue),
            pathway=PathwayRecommender().fit(cls.pairs, cls.catalogue),
        )
        cls.fallback = PopularityRecommender().fit(cls.pairs, cls.catalogue)

    def make(self, weights):
        return PathwayHybrid(**self.parts, weights=weights, fallback=self.fallback)

    def test_weights_must_add_up_to_one(self):
        with self.assertRaises(ValueError):
            self.make([0.5, 0.5, 0.5])
        with self.assertRaises(ValueError):
            self.make({"1": [1, 0, 0]})  # missing groups

    def test_adaptive_weights_depend_on_history_length(self):
        weights = {"1": [0, 0, 1], "2-4": [1, 0, 0], "5-19": [1, 0, 0], "20+": [0, 1, 0]}
        model = self.make(weights)
        self.assertTrue(np.allclose(model.weights_for(1), [0, 0, 1]))
        self.assertTrue(np.allclose(model.weights_for(3), [1, 0, 0]))
        self.assertTrue(np.allclose(model.weights_for(25), [0, 1, 0]))

    def test_new_learner_gets_the_fallback(self):
        model = self.make([1 / 3, 1 / 3, 1 / 3])
        top = model.recommend(np.array([], dtype=int), k=1)
        self.assertEqual(top.tolist(), [self.idx[10]])

    def test_scores_are_a_weighted_mix_of_scaled_signals(self):
        model = self.make([0.2, 0.3, 0.5])
        history = np.array([self.idx[10], self.idx[11]])
        comps = model.component_scores(history)
        self.assertEqual(comps.shape, (3, self.catalogue.n_items))
        self.assertTrue(np.allclose(model.score(history), np.array([0.2, 0.3, 0.5]) @ comps))
        self.assertEqual(model.recommend(history, k=1).tolist(), [self.idx[12]])

    def test_no_evidence_means_no_learning_path_reason(self):
        # Video 15 was never followed by anything, so the learning path only holds its popularity tie-break
        # and must not claim that anything is "often opened next" after it.
        model = self.make([0.0, 0.0, 1.0])
        history = np.array([self.idx[15]])
        top = model.recommend(history, k=2)
        for reason in model.explain(history, top, self.catalogue):
            self.assertNotEqual(reason["signal"], "pathway")

    def test_explanations_name_a_video_from_the_history(self):
        model = self.make([0.0, 0.0, 1.0])
        history = np.array([self.idx[10], self.idx[11]])
        top = model.recommend(history, k=1)
        reason = model.explain(history, top, self.catalogue)[0]
        self.assertEqual(reason["signal"], "pathway")
        self.assertIn("Excel formulas", reason["reason"])
        self.assertEqual(reason["because_of"], 11)


if __name__ == "__main__":
    unittest.main()
