"""Regression tests for the stage-13 support atlas."""

import unittest

from relational_atlas_i01.holdout import qiskit_rank_holdout13_support_atlas as study


class SupportAtlasTests(unittest.TestCase):
    def test_all_system_supports_and_cross_controls_are_frozen(self):
        specs = study.scenario_specs()
        self.assertEqual(len(specs), 11)
        self.assertEqual([spec[0] for spec in specs[:6]], [f"same_q{i}" for i in range(6)])
        self.assertEqual([spec[0] for spec in specs[6:]], [f"q0_to_q{i}" for i in range(1, 6)])

    def test_circuit_support_and_placement_checks(self):
        checks = study.circuit_checks()
        self.assertEqual(checks["same_q5"]["supports"], [5, 5])
        self.assertEqual(checks["q0_to_q4"]["supports"], [0, 4])


if __name__ == "__main__":
    unittest.main()
