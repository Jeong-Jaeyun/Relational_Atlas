"""Regression tests for the complete paired-support transfer matrix."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from relational_atlas_i01.holdout import qiskit_rank_holdout14_transfer_matrix as study


class TransferMatrixTests(unittest.TestCase):
    def test_all_36_source_return_pairs_are_frozen(self):
        specs = study.scenario_specs()
        self.assertEqual(len(specs), 36)
        self.assertEqual(specs[0][0], "q0_to_q0")
        self.assertEqual(specs[-1][0], "q5_to_q5")

    def test_conjugate_pairing_matches_two_qubit_momentum_encoding(self):
        self.assertEqual([study.conjugate_partner(q) for q in range(6)], [1, 0, 3, 2, 5, 4])

    def test_all_circuit_supports_and_placements_are_checked(self):
        self.assertEqual(study.circuit_checks()["count"], 36)

    def test_checkpoint_rejects_a_different_scenario_order(self):
        with TemporaryDirectory() as temporary:
            original = study.CHECKPOINT
            try:
                study.CHECKPOINT = Path(temporary) / "checkpoint.json"
                study._save_checkpoint(__import__("numpy").zeros((6, 6)), {})
                matrix, vectors = study._checkpoint_state()
                self.assertEqual(matrix.shape, (6, 6))
                self.assertEqual(vectors, {})
            finally:
                study.CHECKPOINT = original


if __name__ == "__main__":
    unittest.main()
