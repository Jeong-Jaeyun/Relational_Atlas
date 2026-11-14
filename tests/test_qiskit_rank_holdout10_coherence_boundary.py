"""Regression tests for the paired-coherence boundary design."""

import unittest

from relational_atlas_i01.holdout import qiskit_rank_holdout10_coherence_boundary as study


class CoherenceBoundaryTests(unittest.TestCase):
    def test_frozen_angle_grid_is_ordered_and_nonzero(self):
        self.assertEqual(tuple(sorted(study.ANGLES)), study.ANGLES)
        self.assertTrue(all(angle > 0 for angle in study.ANGLES))

    def test_first_crossing_returns_first_and_none(self):
        scenarios = [
            {"intervention_angle": 0.1, "summary": {"flag": False}},
            {"intervention_angle": 0.2, "summary": {"flag": True}},
            {"intervention_angle": 0.3, "summary": {"flag": True}},
        ]
        self.assertEqual(study.first_crossing(scenarios, "flag"), 0.2)
        self.assertIsNone(study.first_crossing(scenarios, "missing"))

    def test_paired_circuit_check_is_reused(self):
        self.assertEqual(study.paired.circuit_checks()["fault_qubit"], 0)


if __name__ == "__main__":
    unittest.main()
