"""Regression checks for the corrected weak pooled-null test."""

import json
import unittest

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout3_shots as acquisition
from relational_atlas_i01.holdout import qiskit_rank_holdout4b_weaknull as corrected


class WeakPooledNullTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads(acquisition.RESULT.read_text(encoding="utf-8"))
        cls.counts = np.asarray(cls.source["raw_zero_counts"], dtype=int)
        cls.exact = np.asarray(cls.source["exact_zero_probabilities"], dtype=float)

    def test_uniform_label_weak_null_is_not_a_stratumwise_null(self):
        first = self.exact[0, 0, 0]
        second = self.exact[0, 0, 1]
        self.assertGreater(float(np.max(np.abs(second - first))), 0.01)
        self.assertLess(float(np.max(np.abs(np.mean(second - first, axis=1)))),
                        1e-10)

    def test_logistic_null_fit_respects_probability_bounds_and_weighted_equality(self):
        allocation = np.asarray(self.source["shot_allocations"]["skewed"])
        first = self.counts[1, 0, 0, 0]
        second = self.counts[1, 0, 0, 1]
        fitted_first, fitted_second, shifts = corrected.fitted_weak_null(
            first, second, allocation)
        self.assertTrue(np.all((fitted_first > 0) & (fitted_first < 1)))
        self.assertTrue(np.all((fitted_second > 0) & (fitted_second < 1)))
        self.assertLess(float(np.max(np.abs(
            (fitted_second - fitted_first) @ (allocation / allocation.sum())))),
            1e-12)
        self.assertEqual(len(shifts), 3)

    def test_corrected_eight_case_analysis_runs_on_preserved_counts(self):
        output = corrected.analyze_raw(self.source, bootstraps=99, seed=7)
        self.assertEqual(output["familywise_tests"], 8)
        self.assertAlmostEqual(output["familywise_bonferroni_threshold"], 0.00625)
        for row in output["rows"]:
            self.assertLess(row["test"]["fitted_null_max_weighted_difference"],
                            1e-12)


if __name__ == "__main__":
    unittest.main()
