"""Regression tests for the stage-11 paired finite-shot design."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from relational_atlas_i01.holdout import qiskit_rank_holdout11_paired_shots as study


class PairedShotTests(unittest.TestCase):
    def test_four_case_family_and_expected_roles_are_frozen(self):
        cases = study.case_specs()
        self.assertEqual(len(cases), 4)
        self.assertEqual(cases[0][0], "candidate_constraint")
        self.assertEqual(cases[1][-1], "null")
        self.assertEqual(cases[-1][-1], "positive_but_low_power")

    def test_resource_count_matches_four_condition_schedule(self):
        raw = {
            name: {"shot_allocation": [study.SHOTS_PER_LABEL] *
                   study.previous.LABELS}
            for name, *_ in study.CASES
        }
        self.assertEqual(study.total_preparations(raw),
                         4 * 2 * study.previous.N *
                         study.previous.LABELS * study.SHOTS_PER_LABEL)
        self.assertEqual(0.05 / len(study.CASES), 0.0125)

    def test_paired_circuit_check_is_reused(self):
        self.assertEqual(study.circuit_checks()["fault_qubit"], 0)

    def test_completed_cases_are_checkpointed_and_can_be_resumed(self):
        with TemporaryDirectory() as temporary:
            original = study.CHECKPOINT
            try:
                study.CHECKPOINT = Path(temporary) / "checkpoint.json"
                cases = {"candidate_constraint": {"sentinel": 1}}
                study._checkpoint_cases(cases, study.SEED)
                self.assertEqual(study._resumable_cases(study.SEED), cases)
                self.assertEqual(study._resumable_cases(study.SEED + 1), {})
            finally:
                study.CHECKPOINT = original


if __name__ == "__main__":
    unittest.main()
