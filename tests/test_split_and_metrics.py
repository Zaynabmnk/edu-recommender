import unittest

import numpy as np
import pandas as pd

from edurec.metrics import case_metrics, catalogue_coverage, intra_list_diversity, novelty, target_position
from edurec.split import assign_folds, histories_before, temporal_split
from edurec.timesplit import cases_between, paired_tests
from edurec.stats import bootstrap_ci, holm, mcnemar_exact
from edurec.tuning import best_mix, expected_hit_and_ndcg, grid_ndcg, simplex_grid
from tests.helpers import tiny_dataset


class SplitTests(unittest.TestCase):

    def setUp(self):
        _, self.pairs = tiny_dataset()
        self.train_v, self.val, self.train, self.test = temporal_split(self.pairs)

    def test_last_video_is_test_and_second_last_is_validation(self):
        self.assertEqual(dict(zip(self.test.user_id, self.test.item_id)), {1: 12, 2: 12, 3: 11, 4: 14, 5: 14})
        self.assertEqual(dict(zip(self.val.user_id, self.val.item_id)), {1: 11, 2: 11})

    def test_nothing_held_out_leaks_into_training(self):
        held = set(zip(self.test.user_id, self.test.item_id))
        self.assertFalse(held & set(zip(self.train.user_id, self.train.item_id)))
        validation = set(zip(self.val.user_id, self.val.item_id))
        self.assertFalse(validation & set(zip(self.train_v.user_id, self.train_v.item_id)))
        self.assertIn(6, set(self.train.user_id))  # a single-video learner stays in training only

    def test_histories_are_in_time_order(self):
        histories, _ = histories_before(self.train, self.test)
        self.assertEqual([len(h) for h in histories], [2, 2, 1, 1, 1])
        self.assertEqual(histories[0].tolist(), [0, 1])

    def test_cut_off_cases_only_use_the_past(self):
        cutoff, end = pd.Timestamp("2020-01-04"), pd.Timestamp("2030-01-01")
        cases = cases_between(self.pairs, cutoff, end)
        # Learners 4 and 5 arrived after the cut-off; the model sees their first video and guesses the second.
        self.assertEqual(sorted((group, user) for group, user, *_ in cases), [("newcomer", 4), ("newcomer", 5)])
        for _, _, history, target, _ in cases:
            self.assertEqual(len(history), 1)
            self.assertNotIn(target, history)

    def test_every_test_learner_gets_one_fold(self):
        folds = assign_folds(self.test, n_folds=2)
        self.assertEqual(sorted(folds.index), [1, 2, 3, 4, 5])
        self.assertTrue(set(folds.unique()) <= {0, 1})


class MetricTests(unittest.TestCase):

    def test_first_place(self):
        m = case_metrics(0, 0, 100)
        self.assertEqual((m["hr@10"], m["ndcg@10"], m["mrr"], m["auc"]), (1.0, 1.0, 1.0, 1.0))

    def test_eleventh_place(self):
        m = case_metrics(10, 0, 100)
        self.assertEqual(m["hr@10"], 0.0)
        self.assertEqual(m["hr@20"], 1.0)
        self.assertAlmostEqual(m["mrr"], 1 / 11)
        self.assertAlmostEqual(m["precision@10"], 0.0)

    def test_ties_use_the_expected_value(self):
        m = case_metrics(8, 3, 100)  # the target could be at positions 9, 10, 11 or 12
        self.assertAlmostEqual(m["hr@10"], 0.5)
        self.assertAlmostEqual(m["ndcg@10"], (1 / np.log2(10) + 1 / np.log2(11)) / 4)

    def test_vectorised_version_agrees(self):
        rng = np.random.default_rng(0)
        above, tied = rng.integers(0, 30, 200), rng.integers(0, 6, 200)
        hit, ndcg = expected_hit_and_ndcg(above, tied, 10)
        for a, t, h, n in zip(above, tied, hit, ndcg):
            m = case_metrics(int(a), int(t), 1000)
            self.assertAlmostEqual(h, m["hr@10"])
            self.assertAlmostEqual(n, m["ndcg@10"])

    def test_target_position_ignores_seen_videos(self):
        scores = np.array([9.0, 8.0, 7.0, 7.0, 1.0])
        self.assertEqual(target_position(scores, 2, np.array([0])), (1, 1, 4))

    def test_list_metrics(self):
        self.assertEqual(catalogue_coverage([[0, 1], [1, 2]], 10), 0.3)
        counts = np.array([9.0, 0.0, 0.0])
        self.assertGreater(novelty([[1, 2]], counts, 9), novelty([[0]], counts, 9))
        similarity = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
        self.assertEqual(intra_list_diversity([[0, 1]], similarity), 0.0)
        self.assertEqual(intra_list_diversity([[0, 2]], similarity), 1.0)


class TuningAndStatsTests(unittest.TestCase):

    def test_grid_has_66_mixes_that_add_to_one(self):
        grid = simplex_grid(3, 0.1)
        self.assertEqual(len(grid), 66)
        self.assertTrue(np.allclose(grid.sum(axis=1), 1.0))

    def test_grid_search_finds_the_useful_signal(self):
        # Signal 2 puts the target first, the other two put it last.
        stack = np.array([[0.9, 0.1, 0.5, 0.0], [0.9, 0.1, 0.5, 0.0], [0.0, 1.0, 0.2, 0.1]])
        grid = simplex_grid(3, 0.1)
        ndcg = grid_ndcg(stack, target=1, history=np.array([3]), grid=grid)
        best = best_mix(ndcg[:, None], np.array([True]), grid)
        self.assertGreater(best[2], 0.5)

    def test_holm(self):
        self.assertTrue(np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))

    def test_mcnemar_and_bootstrap(self):
        a = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
        self.assertEqual(mcnemar_exact(a, a), 1.0)
        low, high = bootstrap_ci(np.array([0.0, 1.0] * 50))
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_time_split_pairs_cases_before_testing(self):
        final, other = "Pathway hybrid (adaptive)", "Popularity"
        hits = {final: [1, 1, 1, 0.5, 0], other: [0, 0, 1, 0, 1]}
        per_case = pd.DataFrame([{"case": case, "group": "newcomer", "method": method, "hr@10": value}
                                 for method, values in hits.items() for case, value in enumerate(values)])
        row = paired_tests(per_case).iloc[0]
        # An expected hit of 0.5 counts as a hit, as in the main experiment.
        self.assertEqual((row["only_final_hit"], row["only_other_hit"]), (3, 1))
        self.assertAlmostEqual(row["mcnemar_p"], mcnemar_exact(np.array([1, 1, 1, 1, 0], dtype=bool),
                                                               np.array([0, 0, 1, 0, 1], dtype=bool)))


if __name__ == "__main__":
    unittest.main()
