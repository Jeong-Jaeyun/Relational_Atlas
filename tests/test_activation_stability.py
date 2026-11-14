"""Exact checks for the symmetric activation and single-flip model."""

import math
import unittest

import numpy as np

from relational_atlas_i01.geometry import activation_stability as model


class ActivationStabilityTests(unittest.TestCase):
    def test_ground_count_has_finite_basin(self):
        m = np.arange(8)
        for ratio in (2.1, 2.5, 2.9):
            self.assertEqual(int(np.argmin(model.energy(m, ratio))), 3)
        self.assertEqual(int(np.argmin(model.energy(m, 1.9))), 2)
        self.assertEqual(int(np.argmin(model.energy(m, 3.1))), 4)

    def test_stationary_distribution(self):
        for n in (4, 6, 20):
            self.assertAlmostEqual(float(model.count_probabilities(n, 0.0, 2.5).sum()), 1.0)
            self.assertAlmostEqual(
                float(model.count_probabilities(n, 0.0, 2.5)[3]),
                math.comb(n, 3) / 2**n,
            )
            self.assertGreater(model.count_probabilities(n, 10.0, 2.5)[3], 0.9)

    def test_first_exit_identity_and_boundary_loss(self):
        for n in (4, 6, 20):
            self.assertAlmostEqual(model.first_exit_time_from_three(n, 10.0, 2.5),
                                   math.exp(5) / n)
            center = model.first_exit_time_from_three(n, 10.0, 2.5)
            self.assertGreater(center, model.first_exit_time_from_three(n, 10.0, 2.05))


if __name__ == "__main__":
    unittest.main()
