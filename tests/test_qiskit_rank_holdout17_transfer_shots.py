"""Regression tests for the finite-shot paired-support transfer family."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from relational_atlas_i01.holdout import qiskit_rank_holdout17_transfer_shots as study


class TransferShotTests(unittest.TestCase):
    def test_four_class_family_and_roles_are_frozen(self):
        cases = study.case_specs()
        self.assertEqual(len(cases), 4)
        self.assertEqual(cases[0][0], "same_support")
        self.assertEqual(cases[1][-1], "null")
        self.assertTrue(all(case[-1] == "positive" for case in cases[::2]))

    def test_resource_count_matches_the_predeclared_four_case_schedule(self):
        cases = {name: {"shot_allocation": [study.SHOTS_PER_LABEL] * study.LABELS}
                 for name, *_ in study.CASES}
        self.assertEqual(study.total_preparations(cases),
                         4 * 2 * study.N * study.LABELS * study.SHOTS_PER_LABEL)
        self.assertEqual(0.05 / len(study.CASES), 0.0125)

    def test_checkpoint_roundtrip_is_seed_specific(self):
        with TemporaryDirectory() as temporary:
            original = study.CHECKPOINT
            try:
                study.CHECKPOINT = Path(temporary) / "checkpoint.json"
                cases = {"same_support": {"sentinel": 1}}
                study._save_checkpoint(cases, study.SEED)
                self.assertEqual(study._load_checkpoint(study.SEED), cases)
                self.assertEqual(study._load_checkpoint(study.SEED + 1), {})
            finally:
                study.CHECKPOINT = original


if __name__ == "__main__":
    unittest.main()
