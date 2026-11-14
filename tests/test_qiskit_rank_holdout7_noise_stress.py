"""Regression tests for the localized-noise stage-7 design."""

import unittest

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout7_noise_stress as study


class NoiseStressTests(unittest.TestCase):
    def test_predeclared_scenarios_are_one_error_at_a_time(self):
        scenarios = study.scenario_specs()
        self.assertEqual(scenarios[0], ("ideal", 0.0))
        self.assertEqual(len(scenarios), 10)
        self.assertEqual({kind for kind, _ in scenarios[1:]},
                         set(study.ERROR_KINDS))

    def test_symmetric_readout_response_preserves_probability_ordering(self):
        probabilities = np.zeros(1 << study.TOTAL_QUBITS)
        probabilities[0] = 0.7
        probabilities[3] = 0.2
        probabilities[1 << study.SYSTEM_QUBITS] = 0.1
        joint, clock = study.readout_events(probabilities, 0.01)
        self.assertGreater(clock, joint)
        self.assertTrue(0 < joint < 1)
        self.assertTrue(0 < clock < 1)

    def test_symmetric_readout_response_is_exact_on_basis_events(self):
        probabilities = np.zeros(1 << study.TOTAL_QUBITS)
        probabilities[0] = 1
        joint, clock = study.readout_events(probabilities, 0.01)
        self.assertAlmostEqual(joint, 0.99 ** study.TOTAL_QUBITS)
        self.assertAlmostEqual(clock, 0.99 ** study.CLOCK_QUBITS)
        probabilities[:] = 0
        probabilities[-1] = 1
        joint, clock = study.readout_events(probabilities, 0.01)
        self.assertAlmostEqual(joint, 0.01 ** study.TOTAL_QUBITS)
        self.assertAlmostEqual(clock, 0.01 ** study.CLOCK_QUBITS)

    def test_stage6_ideal_limit_and_axis_symmetry(self):
        checks = study.circuit_checks(workers=2)
        self.assertLess(checks["ideal_joint_probability_maximum_absolute_error"],
                        1e-10)
        self.assertLess(checks["ideal_clock_probability_maximum_absolute_error"],
                        1e-10)
        self.assertLess(max(checks[
            "axis_invariance_maximum_absolute_errors"].values()), 1e-10)
        self.assertGreater(checks[
            "symmetric_readout_axis_asymmetry_requires_full_axis_evaluation"],
            1e-6)

    def test_condition_rows_recover_ideal_strong_and_null_structure(self):
        exact = study.stage6.exact_joint_probabilities()
        joint = exact
        clock = np.full_like(joint, study.CLOCK_ACCEPTANCE)
        rows = study.condition_rows(joint, clock)
        by_role = {row["role"]: row for row in rows}
        self.assertGreater(by_role["strong_signal"][
            "maximum_absolute_joint_difference"], 5e-4)
        for role in ("distribution_null_tau_0", "distribution_null_tau_half_pi",
                     "constraint_null_tau_0", "constraint_null_tau_half_pi"):
            self.assertLess(by_role[role]["maximum_absolute_joint_difference"],
                            1e-12)
        summary = study.scenario_summary(rows, rows)
        self.assertEqual(summary["classification"],
                         "all_four_nulls_numerically_preserved")
        self.assertTrue(np.isclose(summary["strong_signal_fraction_of_ideal"], 1))


if __name__ == "__main__":
    unittest.main()
