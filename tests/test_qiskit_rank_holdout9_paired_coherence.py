"""Regression tests for the paired-coherence intervention."""

import unittest

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout9_paired_coherence as study


class PairedCoherenceTests(unittest.TestCase):
    def test_two_rotations_bracket_the_scientific_phase_layer(self):
        checks = study.circuit_checks()
        self.assertEqual(checks["first_rotation"], -checks["second_rotation"])
        self.assertEqual(len(checks["barrier_indices"]), 2)
        self.assertEqual(checks["rotation_indices"][0],
                         checks["barrier_indices"][0] + 1)
        self.assertEqual(checks["rotation_indices"][1],
                         checks["barrier_indices"][1] + 1)

    def test_zero_angle_recovers_stage6_exact_probabilities(self):
        templates = study.stage7.compiled_templates()
        joint, clock = study.full_axis_probabilities(0.0, 2, templates)
        self.assertLess(np.max(np.abs(
            joint - study.stage6.exact_joint_probabilities())), 1e-10)
        self.assertLess(np.max(np.abs(clock - study.CLOCK_ACCEPTANCE)), 1e-10)

    def test_predeclared_angles_are_positive_and_distinct(self):
        self.assertTrue(all(angle > 0 for angle in study.ANGLES))
        self.assertEqual(len(study.ANGLES), len(set(study.ANGLES)))


if __name__ == "__main__":
    unittest.main()

