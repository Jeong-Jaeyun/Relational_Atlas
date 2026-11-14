"""Regression checks for the local encounter baseline."""

import math
import unittest

import numpy as np

from relational_atlas_i01.geometry import encounter_threshold as model


class EncounterThresholdTests(unittest.TestCase):
    def test_codimension_zero_and_small_radius_scaling(self):
        self.assertEqual(model.chi_ball_probability(0, 0.001, 0.35), 1.0)
        for codimension in (1, 2, 3):
            p1 = model.chi_ball_probability(codimension, 0.001, 0.35)
            p2 = model.chi_ball_probability(codimension, 0.002, 0.35)
            self.assertAlmostEqual(math.log2(p2 / p1), codimension, delta=2e-5)

    def test_direct_mc_agrees_with_projection_identity(self):
        rng = np.random.default_rng(125)
        for dimension in (3, 4, 5):
            row = model.sample_one_dimension(dimension, 0.35, 20000, rng)
            target = next(item for item in row["encounter_scan"]
                          if item["encounter_radius"] == 0.08)
            self.assertAlmostEqual(
                target["direct_mc_encounter_probability"],
                target["rao_blackwell_encounter_probability"],
                delta=0.012,
            )

    def test_three_stage_optimum(self):
        p3, p4 = 0.4, 0.01
        optimum = math.log1p(p3 / p4) / p3
        peak = model.staged_three_probability(p3, p4, optimum)
        self.assertGreater(peak, model.staged_three_probability(p3, p4, optimum / 2))
        self.assertGreater(peak, model.staged_three_probability(p3, p4, optimum * 2))


if __name__ == "__main__":
    unittest.main()
