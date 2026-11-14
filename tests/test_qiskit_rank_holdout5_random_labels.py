"""Regression checks for the randomized-label Aer protocol."""

import unittest

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout5_power as power
from relational_atlas_i01.holdout import qiskit_rank_holdout5_random_labels as experiment


class RandomLabelTests(unittest.TestCase):
    def test_frozen_random_allocation_preserves_totals(self):
        counts = experiment.allocation(20261023)
        self.assertEqual(counts.shape, (3, 2, 2, 3, 7))
        self.assertTrue(np.all(counts.sum(axis=-1) == experiment.TOTAL_SHOTS))
        self.assertFalse(np.array_equal(counts[0], counts[1]))

    def test_exact_pilot_separates_drift_and_P_zero(self):
        rows = power.study()["rows"]
        self.assertEqual(len(rows), 4)
        for row in rows:
            if row["physical_P_zero"]:
                self.assertLess(row[
                    "uniform_to_skewed_max_pooled_probability_difference"], 1e-10)
                self.assertAlmostEqual(row[
                    "iid_label_to_fixed_label_variance_ratio_range"][0], 1.0)
            else:
                self.assertGreater(row[
                    "uniform_to_skewed_max_pooled_probability_difference"], 0.01)
                self.assertGreater(row[
                    "iid_label_to_fixed_label_variance_ratio_range"][0], 1.1)

    def test_recorded_label_null_handles_unequal_trials(self):
        first_n = np.full((3, 7), 100)
        second_n = np.full((3, 7), 200)
        first = np.full((3, 7), 20)
        second = np.full((3, 7), 40)
        result = experiment.uniform_standardization_pvalue(
            first, second, first_n, second_n, 99, np.random.default_rng(5))
        self.assertEqual(result["statistic"], 0.0)
        self.assertEqual(result["permutation_pvalue"], 1.0)


if __name__ == "__main__":
    unittest.main()
