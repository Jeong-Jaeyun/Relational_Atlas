"""Regression tests for the stage-8 symmetry-breaking fault design."""

import unittest

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout8_symmetry_breaking as study


class SymmetryBreakingTests(unittest.TestCase):
    def test_frozen_scenarios_are_one_fault_at_a_time(self):
        scenarios = study.scenario_specs()
        self.assertEqual(scenarios[0], ("ideal", 0.0))
        self.assertEqual(len(scenarios), 7)
        self.assertEqual([kind for kind, _ in scenarios[1:4]],
                         ["coherent_preparation_rx"] * 3)
        self.assertEqual([kind for kind, _ in scenarios[4:]],
                         ["label_dependent_readout"] * 3)

    def test_label_error_profile_is_positive_and_has_declared_mean(self):
        profile = np.array([study.label_readout_error(0.01, label)
                            for label in range(study.LABELS)])
        self.assertTrue(np.all(profile > 0))
        self.assertAlmostEqual(profile.mean(), 0.01)
        self.assertGreater(profile.max(), profile.min())

    def test_coherent_fault_is_immediately_after_preparation_barrier(self):
        checks = study.circuit_checks()
        self.assertEqual(checks["fault_instruction_after_first_barrier"], "rx")
        self.assertEqual(checks["fault_qubit"], 0)

    def test_zero_fault_recovers_stage6_exact_probabilities(self):
        templates = study.stage7.compiled_templates()
        joint, clock = study.full_axis_probabilities(2, templates=templates)
        self.assertLess(np.max(np.abs(
            joint - study.stage6.exact_joint_probabilities())), 1e-10)
        self.assertLess(np.max(np.abs(clock - study.CLOCK_ACCEPTANCE)), 1e-10)


if __name__ == "__main__":
    unittest.main()

