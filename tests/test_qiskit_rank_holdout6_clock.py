"""Regression tests for the explicit-clock intervention."""

import json
import unittest

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout6_clock as experiment
from relational_atlas_i01.holdout import qiskit_rank_holdout6_replication as replication


class ExplicitClockTests(unittest.TestCase):
    def test_exact_rank_and_power_pilot(self):
        rows = experiment.structural_pilot()["rows"]
        self.assertEqual(len(rows), 8)
        for row in rows:
            expected_kept = 2 if row["physical_P_zero"] else 3
            expected_erased = (3 if row["label_distribution"] == "skewed" and
                               not row["physical_P_zero"] else 2)
            self.assertEqual(row["label_kept_fisher"]["rank_relative_1e_minus_8"],
                             expected_kept)
            self.assertEqual(row["label_erased_fisher"]["rank_relative_1e_minus_8"],
                             expected_erased)
        strong = next(row for row in rows if row["label_distribution"] == "skewed"
                      and not row["physical_P_zero"] and row["tau_over_pi"] == 0)
        self.assertGreater(strong["approx_noncentrality"], 20)

    def test_statevector_and_density_match_exact_joint_probability(self):
        checks = experiment.circuit_checks()
        self.assertEqual(len(checks["cases"]), 4)
        self.assertLess(checks["max_absolute_error"], 1e-10)

    def test_joint_success_is_conditioned_on_clock_acceptance(self):
        exact = experiment.exact_joint_probabilities()
        self.assertEqual(exact.shape, (2, 2, 2, 3, 7))
        self.assertTrue(np.all((exact >= 0) & (exact <= 0.25)))
        self.assertTrue(np.allclose(exact / experiment.CLOCK_ACCEPTANCE,
                                    experiment.previous.exact_probabilities()))

    def test_saved_combination_is_cellwise_sum_and_counts_resources(self):
        first = json.loads(experiment.RESULT.read_text(encoding="utf-8"))
        second = json.loads(replication.REPLICATE.read_text(encoding="utf-8"))
        combined = json.loads(replication.COMBINED.read_text(encoding="utf-8"))
        for key in ("raw_joint_zero_counts", "raw_clock_accept_counts"):
            self.assertTrue(np.array_equal(
                np.asarray(first[key]) + np.asarray(second[key]),
                np.asarray(combined[key])))
        self.assertEqual(combined["analysis"]["total_preparations"],
                         2 * 44_040_192)
        self.assertEqual(combined["analysis"]["familywise_tests"], 8)

    def test_combined_null_conditions_are_not_familywise_detections(self):
        combined = json.loads(replication.COMBINED.read_text(encoding="utf-8"))
        rows = combined["analysis"]["rows"]
        threshold = combined["analysis"]["familywise_bonferroni_threshold"]
        self.assertEqual(threshold, 0.00625)
        for row in rows:
            if row["physical_P_zero"] or row["label_distribution"] == "uniform":
                self.assertGreater(row["test"]["weak_null_bootstrap_pvalue"],
                                   threshold)


if __name__ == "__main__":
    unittest.main()
