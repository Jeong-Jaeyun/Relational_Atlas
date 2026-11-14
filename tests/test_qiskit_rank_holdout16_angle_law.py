"""Regression tests for the predeclared paired-transfer angle law grid."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from relational_atlas_i01.holdout import qiskit_rank_holdout16_angle_law as study


class AngleLawTests(unittest.TestCase):
    def test_grid_covers_four_classes_and_pi_endpoint(self):
        self.assertEqual(len(study.SCENARIOS), 36)
        self.assertEqual(study.SCENARIOS[0][0], "same_support")
        self.assertAlmostEqual(study.SCENARIOS[-1][1], study.math.pi)

    def test_checkpoint_roundtrip(self):
        with TemporaryDirectory() as temporary:
            original = study.CHECKPOINT
            try:
                study.CHECKPOINT = Path(temporary) / "checkpoint.json"
                rows = [{"support_class": "same_support", "angle_radians": 0.1}]
                study._save_checkpoint(rows)
                self.assertEqual(study._load_checkpoint(), rows)
            finally:
                study.CHECKPOINT = original

    def test_generic_circuit_checks_are_reused(self):
        self.assertEqual(study.circuit_checks()["scenario_count"], 36)


if __name__ == "__main__":
    unittest.main()
