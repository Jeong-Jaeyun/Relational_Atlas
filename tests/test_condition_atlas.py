"""Small deterministic regression tests for the I01 condition atlas."""

import math
import unittest

import numpy as np

from relational_atlas_i01.geometry import condition_atlas as atlas


class ConditionAtlasTests(unittest.TestCase):
    def test_exact_diagonal_quotient_and_rank(self):
        sectors = atlas.exact_sectors()
        self.assertEqual(sectors["n4_cutoff1"]["gauge_qfi"]["rank"], 3)
        self.assertEqual(sectors["n4_cutoff3"]["gauge_qfi"]["rank"], 3)
        self.assertEqual(sectors["n5_cutoff1"]["gauge_qfi"]["rank"], 4)
        self.assertEqual(sectors["n6_cutoff1"]["gauge_qfi"]["rank"], 5)
        topology = atlas.orbit_and_topology()
        self.assertEqual(topology["B_rank"], 3)
        self.assertLess(topology["diagonal_shift_invariance_max_error"], 1e-12)
        self.assertEqual(topology["flow_modes"]["fully_rationally_independent"]["invariant_qfi_rank"], 0)

    def test_gap_window_and_rank_return(self):
        scan = atlas.hard_gap_scan()["scan"]
        rows = {row["energy_limit"]: row for row in scan["4.0"]}
        self.assertEqual(rows[1.5]["qfi"]["rank"], 0)
        self.assertEqual(rows[2.0]["qfi"]["rank"], 3)
        self.assertEqual(rows[4.9]["qfi"]["rank"], 3)
        self.assertEqual(rows[5.0]["qfi"]["rank"], 4)
        self.assertEqual({row["energy_limit"]: row for row in scan["1.0"]}[2.0]["qfi"]["rank"], 4)

    def test_clock_blindness_is_measurement_dependent(self):
        clock = atlas.clock_observables()
        self.assertEqual(clock["unrestricted_relative_detector_rank_at_tau0"], 3)
        self.assertEqual(clock["unrestricted_augmented_detector_rank_at_tau0"], 4)
        self.assertEqual(clock["physical_augmented_detector_rank_at_tau0"], 3)
        self.assertAlmostEqual(clock["conditional_state_fidelity_at_pi"], 1.0, places=12)
        half = min(clock["trajectory"], key=lambda row: abs(row["tau"] - math.pi / 2))
        self.assertEqual(half["fisher_rank_at_per_shot_threshold_0p01"], 0)
        self.assertEqual(half["two_harmonic_rank_at_per_shot_threshold_0p01"], 3)

    def test_clock_recurrence_robust_to_preparation(self):
        rows = atlas.clock_sensitivity()
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(np.isclose(row["conditional_fidelity_at_pi"], 1.0) for row in rows))
        self.assertTrue(all(row["fisher_nonzero_eigenvalue_at_tau0"] >
                            row["fisher_nonzero_eigenvalue_at_pi_over_2"] for row in rows))

    def test_permutation_symmetric_branch_selection(self):
        selection = atlas.symmetric_branch_selection()
        for n in (4, 5, 6):
            row = selection["by_n"][f"n{n}"]
            self.assertEqual(row["ground_active_counts"], [3])
            self.assertEqual(row["one_branch_qfi"]["rank"], 3)
            self.assertEqual(row["branch_resolved_average_qfi"]["rank"], n)
            fractions = [point["probability_of_three_active_axes"]
                         for point in row["branch_probability_scan"]]
            self.assertTrue(all(left < right for left, right in zip(fractions, fractions[1:])))


if __name__ == "__main__":
    unittest.main()
