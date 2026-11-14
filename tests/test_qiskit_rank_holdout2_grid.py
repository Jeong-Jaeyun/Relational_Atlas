"""Regression checks for the frozen detector/spectrum/label holdout."""

import math
import unittest

from relational_atlas_i01.holdout import qiskit_rank_holdout2_grid as grid


class RankHoldout2GridTests(unittest.TestCase):
    def test_frozen_432_row_summary(self):
        summary = grid.study()["summary"]
        self.assertEqual(summary["rows"], 432)
        self.assertEqual(summary["guaranteed_invariant_rows"], 360)
        self.assertEqual(summary["guaranteed_invariance_failures_above_1e_minus_10"], 0)
        self.assertEqual(summary[
            "guaranteed_erased_rank_failures_relative_1e_minus_8"], 0)
        self.assertEqual(summary["expected_kept_rank_failures_relative_1e_minus_8"], 0)
        self.assertEqual(summary["state_qfi_rank_failures_relative_1e_minus_8"], 0)
        self.assertEqual(summary["skewed_coherent_full_rows"], 72)
        self.assertEqual(summary[
            "skewed_coherent_full_rows_with_residual_above_1e_minus_10"], 72)
        self.assertEqual(summary[
            "skewed_coherent_full_erased_rank_n_rows_relative_1e_minus_8"], 72)
        self.assertLess(summary["maximum_gradient_crosscheck_error"], 1e-8)

    def test_uniform_and_twirled_invariance_across_spectra(self):
        for spectrum in grid.SPECTRA:
            entry = grid.model(3, 0.45, False, spectrum)
            coherent = grid.information(entry, "mixed", math.pi / 2, "uniform")
            twirled = grid.information(entry, "twirled", math.pi / 2, "skewed")
            self.assertLess(
                coherent["erased_common_shift_max_probability_change"], 1e-10)
            self.assertLess(
                twirled["erased_common_shift_max_probability_change"], 1e-10)
            self.assertEqual(coherent["label_kept"]["rank_relative_1e_minus_8"], 3)
            self.assertEqual(twirled["label_kept"]["rank_relative_1e_minus_8"], 2)

    def test_skewed_erasure_has_residual_without_constraint(self):
        entry = grid.model(3, 0.45, False, "nonlinear")
        result = grid.information(entry, "mixed", math.pi / 2, "skewed")
        self.assertGreater(result["erased_common_shift_max_probability_change"],
                           1e-5)
        self.assertEqual(result["label_erased"]["rank_relative_1e_minus_8"], 3)
        physical = grid.model(3, 0.45, True, "nonlinear")
        physical_result = grid.information(physical, "mixed", math.pi / 2,
                                           "skewed")
        self.assertLess(physical_result[
            "erased_common_shift_max_probability_change"], 1e-10)
        self.assertEqual(physical_result["label_erased"][
            "rank_relative_1e_minus_8"], 2)


if __name__ == "__main__":
    unittest.main()
