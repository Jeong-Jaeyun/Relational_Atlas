"""Checks for the paired phase-label intervention and resource distinction."""

import unittest

import numpy as np

from relational_atlas_i01.phase_reference import qiskit_phase_label_intervention as intervention


class PhaseLabelInterventionTests(unittest.TestCase):
    def test_exact_intervention_distinguishes_detector_from_constraint(self):
        rows = intervention.exact_study()
        full = rows["unrestricted"]
        physical = rows["P_zero"]
        self.assertEqual(full["label_kept_joint_cfi"]["rank_tol_1e_minus_8"], 4)
        self.assertEqual(full["label_erased_marginal_cfi"]["rank_tol_1e_minus_8"], 3)
        self.assertEqual(physical["label_kept_joint_cfi"]["rank_tol_1e_minus_8"], 3)
        self.assertEqual(physical["label_erased_marginal_cfi"]["rank_tol_1e_minus_8"], 3)
        self.assertGreaterEqual(full["data_processing_difference_min_eigenvalue"], -1e-10)
        self.assertLess(full["erased_probability_max_change_under_common_displacement"], 1e-12)

    def test_same_counts_have_distinct_labeled_and_erased_statistics(self):
        a = np.array([[10, 0], [0, 10]])
        b = np.array([[0, 10], [10, 0]])
        self.assertGreater(intervention.deviance(a, b, 10), 20)
        self.assertAlmostEqual(intervention.deviance(a.sum(axis=1),
                                                      b.sum(axis=1), 20), 0.0)

    def test_unflagged_quadratic_and_flagged_linear_scaling(self):
        result = intervention.loss_scaling()
        self.assertAlmostEqual(result["last_four_loglog_slope_unflagged"], 2.0,
                               delta=0.03)
        self.assertAlmostEqual(result["last_four_loglog_slope_flagged"], 1.0,
                               delta=1e-9)


if __name__ == "__main__":
    unittest.main()
