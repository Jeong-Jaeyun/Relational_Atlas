"""Regression tests for the stage-12 intervention condition atlas."""

import unittest

from relational_atlas_i01.holdout import qiskit_rank_holdout12_intervention_atlas as study


class InterventionAtlasTests(unittest.TestCase):
    def test_six_frozen_conditions_cover_axis_support_and_sign(self):
        specs = study.scenario_specs()
        self.assertEqual(len(specs), 6)
        self.assertEqual(specs[0][0], "one_sided_rx_q0")
        self.assertEqual(specs[1][5], -1.0)
        self.assertEqual(specs[3][1], "z")
        self.assertEqual(specs[4][4], 1)
        self.assertEqual(specs[-1][5], 1.0)

    def test_every_intervention_is_placed_at_the_declared_barriers(self):
        checks = study.circuit_checks()
        self.assertEqual(len(checks), 6)
        self.assertEqual(len(checks["one_sided_rx_q0"]["rotation_indices"]), 1)
        self.assertEqual(len(checks["paired_rx_inverse_same_q0"]["rotation_indices"]), 2)

    def test_invalid_rotation_axis_is_rejected(self):
        base, _ = study.stage7.compiled_templates()[(0, 0)]
        with self.assertRaises(ValueError):
            study._rotate(base, "bad", 0.1, 0)


if __name__ == "__main__":
    unittest.main()
