"""Circuit-level holdout checks independent of the Fisher implementation."""

import unittest

from relational_atlas_i01.holdout import qiskit_rank_holdout_circuit as check
from relational_atlas_i01.holdout import qiskit_rank_holdout_shots as shots


class RankHoldoutCircuitTests(unittest.TestCase):
    def test_statevector_and_aer_density_matrix_agree_with_formula(self):
        row = check.check_case(check.CASES[0])
        self.assertLess(row["statevector_absolute_error"], 1e-10)
        self.assertLess(row["aer_density_matrix_absolute_error"], 1e-10)
        self.assertAlmostEqual(row["statevector_clock_success_probability"],
                               1 / (row["n"] + 1), places=10)

    def test_n3_shot_schedule_is_frozen(self):
        self.assertEqual(shots.N, 3)
        self.assertEqual(shots.LABEL_COUNTS, (5, 7))
        self.assertEqual(len(shots.schedule()), 144)


if __name__ == "__main__":
    unittest.main()
