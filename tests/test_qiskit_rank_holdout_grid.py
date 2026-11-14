"""Regression checks for the frozen I01 holdout matrix."""

import math
import unittest

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout_grid as grid


class RankHoldoutGridTests(unittest.TestCase):
    def test_adequate_phase_grid_and_constraint_null(self):
        for n in grid.NS:
            for physical in (False, True):
                entry = grid.model(n, 0.45, physical)
                result = grid.information(entry, 0.0, 1.25, 2 * n + 1)
                self.assertEqual(result["label_kept"]["rank_relative_1e_minus_8"],
                                 n - int(physical))
                self.assertEqual(result["label_erased"]["rank_relative_1e_minus_8"],
                                 n - 1)
                self.assertLess(result["erased_common_shift_max_probability_change"],
                                1e-10)

    def test_short_phase_grid_has_a_real_failure_boundary(self):
        entry = grid.model(3, 0.45, False)
        short_zero_clock = grid.information(entry, 0.0, 1.25, 5)
        short_half_clock = grid.information(entry, math.pi / 2, 1.25, 5)
        self.assertEqual(short_zero_clock["label_erased"]["rank_relative_1e_minus_8"], 3)
        self.assertGreater(
            short_zero_clock["erased_common_shift_max_probability_change"], 0.005)
        self.assertEqual(short_half_clock["label_erased"]["rank_relative_1e_minus_8"], 2)
        self.assertLess(
            short_half_clock["erased_common_shift_max_probability_change"], 1e-10)

    def test_analytic_gradient_matches_independent_central_difference(self):
        entry = grid.model(5, 1.20, False)
        error = grid.finite_difference_gradient_check(entry, math.pi / 2,
                                                      0.55, 11)
        self.assertLess(error, 1e-8)

    def test_frozen_grid_summary(self):
        summary = grid.study()["summary"]
        self.assertEqual(summary["total_rows"], 128)
        self.assertEqual(summary["adequate_label_predeclared_pattern_failures"], 0)
        self.assertEqual(summary["state_qfi_pattern_failures"], 0)
        self.assertEqual(summary[
            "short_label_rows_with_nonzero_erased_common_change_above_1e_minus_10"], 16)
        self.assertLess(summary["maximum_gradient_crosscheck_error"], 1e-8)


if __name__ == "__main__":
    unittest.main()
