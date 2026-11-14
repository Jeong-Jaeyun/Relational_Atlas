"""Regression tests for the independent same-sign high-budget replication."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from relational_atlas_i01.holdout import qiskit_rank_holdout18_same_sign_replication as study


class SameSignReplicationTests(unittest.TestCase):
    def test_support_angle_and_doubled_budget_are_frozen(self):
        self.assertEqual((study.SOURCE, study.RETURN), (0, 2))
        self.assertAlmostEqual(study.ANGLE, study.math.pi / 2)
        self.assertEqual(study.SHOTS_PER_LABEL, 262144)

    def test_allocation_is_valid_and_has_the_declared_total(self):
        allocation = study.allocation()
        self.assertEqual(int(allocation.sum()), study.LABELS * study.SHOTS_PER_LABEL)
        self.assertTrue((allocation > 0).all())

    def test_checkpoint_is_seed_specific(self):
        with TemporaryDirectory() as temporary:
            original = study.CHECKPOINT
            try:
                study.CHECKPOINT = Path(temporary) / "checkpoint.json"
                raw = {"sentinel": 1}
                study._save_checkpoint(raw, study.SEED)
                self.assertEqual(study._load_checkpoint(study.SEED), raw)
                self.assertIsNone(study._load_checkpoint(study.SEED + 1))
            finally:
                study.CHECKPOINT = original


if __name__ == "__main__":
    unittest.main()
