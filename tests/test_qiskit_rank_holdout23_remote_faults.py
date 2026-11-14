"""Checks for the stage-23 per-qubit remote amplitude-damping experiment."""

import json
import unittest

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout23_remote_faults as study


def _saved() -> dict:
    return json.loads(study.RESULT.read_text(encoding="utf-8"))


def _remote_all() -> dict:
    return json.loads(study.locality.RESULT.read_text(encoding="utf-8"))[
        "scenarios"]


class RemoteFaultTests(unittest.TestCase):
    def test_design_matches_the_stage21_remote_subset(self):
        self.assertEqual(study.QUBITS, study.locality.SUBSETS["remote_system"])
        self.assertEqual(study.LEVELS, study.locality.LEVELS)

    def test_saved_summary_follows_from_saved_scenarios(self):
        saved = _saved()
        remote = _remote_all()
        for level in study.LEVELS:
            with self.subTest(level=level):
                rows = [np.asarray(saved["scenarios"][f"q{qubit}_{level:.3f}"][
                    "analysis"]["axis_pooled_joint_difference"])
                    for qubit in study.QUBITS]
                total = sum(rows, np.zeros(3))
                summary = saved["summary"][str(level)]
                np.testing.assert_allclose(
                    total, summary["sum_single_qubit_differences"],
                    rtol=0, atol=1e-20)
                reference = np.asarray(remote[f"remote_system_{level:.3f}"][
                    "analysis"]["axis_pooled_joint_difference"])
                self.assertAlmostEqual(
                    float(np.max(np.abs(total - reference))),
                    summary["maximum_difference_from_remote_all"], delta=1e-20)

    def test_single_qubit_faults_add_up_to_second_order(self):
        summary = _saved()["summary"]
        remote = _remote_all()
        low, high = study.LEVELS
        gaps = {level: summary[str(level)]["maximum_difference_from_remote_all"]
                for level in study.LEVELS}
        size = remote[f"remote_system_{low:.3f}"]["analysis"][
            "maximum_absolute_joint_difference"]
        # Single-qubit responses sum to the joint response at first order ...
        self.assertLess(gaps[low] / size, 1e-3)
        # ... and the non-additive remainder scales with the square of the rate.
        expected = (high / low) ** 2
        self.assertGreater(gaps[high] / gaps[low], 0.8 * expected)
        self.assertLess(gaps[high] / gaps[low], 1.25 * expected)

    def test_spectator_axes_respond_symmetrically(self):
        # Axes 1 (q2, q3) and 2 (q4, q5) are exchangeable spectators of the
        # q0 -> q1 access, so matching bits must leak by the same amount.
        for level in study.LEVELS:
            per_qubit = _saved()["summary"][str(level)][
                "per_qubit_maximum_contrast"]
            with self.subTest(level=level):
                self.assertAlmostEqual(per_qubit["2"] / per_qubit["4"], 1,
                                       delta=1e-9)
                self.assertAlmostEqual(per_qubit["3"] / per_qubit["5"], 1,
                                       delta=1e-9)


if __name__ == "__main__":
    unittest.main()
