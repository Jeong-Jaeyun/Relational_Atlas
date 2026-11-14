"""Regression checks for stochastic phase bridges and uncertainty floors."""

import math
import unittest

import numpy as np

from relational_atlas_i01.phase_reference import qiskit_clock_stochastic_bridge as study


class ClockStochasticBridgeTests(unittest.TestCase):
    def test_bridge_limits_and_closed_form(self):
        coefficient, variance = study.bridge_parameters(0.0)
        self.assertAlmostEqual(coefficient, 0.5)
        self.assertAlmostEqual(variance, 0.0)
        for gap in (0.05, 1.0, 4.0):
            coefficient, variance = study.bridge_parameters(gap)
            rho = math.exp(-gap)
            self.assertAlmostEqual(coefficient, rho / (1 + rho * rho))
            self.assertAlmostEqual(variance, study.SIGMA ** 2 *
                                   (1 - rho * rho) / (1 + rho * rho))
        self.assertRaises(ValueError, study.bridge_parameters, -0.1)

    def test_unobserved_innovation_leaves_exact_rank_three(self):
        for tau in study.TAUS:
            self.assertEqual(study.free_innovation_profile_rank(tau), 3)

    def test_generated_bridge_residual_variance(self):
        draws = study.bridge_draws(seed=123, blocks=20_000)
        for gap in study.GAP_OVER_CORRELATION:
            residuals = np.array([row["unobserved_bridge_residual"]
                                  for row in draws[str(gap)]])
            _, expected_variance = study.bridge_parameters(gap)
            self.assertAlmostEqual(float(np.var(residuals)), expected_variance,
                                   delta=0.05 * expected_variance)

    def test_shot_only_interval_is_undercovered_under_bridge_prior(self):
        result = study.uncertainty_map(shots=4096, seed=20261010,
                                       replicates=5000, blocks=8)
        for row in result["rows"]:
            self.assertLess(row["shot_only_95pct_coverage"],
                            row["bridge_aware_95pct_coverage"])
            self.assertAlmostEqual(row["bridge_aware_95pct_coverage"], 0.95,
                                   delta=0.025)
            self.assertGreater(row["blocks_for_0_point_02_rad_halfwidth"], 0)


if __name__ == "__main__":
    unittest.main()
