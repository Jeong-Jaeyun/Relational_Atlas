"""Check exact box-constrained distances and alignment limits."""

import unittest

import numpy as np

from relational_atlas_i01.geometry import tangent_alignment as model


class TangentAlignmentTests(unittest.TestCase):
    def test_bounded_minimum_uses_box_faces(self):
        a = np.eye(3)[None, :, :]
        b = np.array([[2.0, 0.1, -0.2]])
        lower = np.full(3, -0.5)
        upper = np.full(3, 0.5)
        result = model.bounded_min_distances(a, b, lower, upper)
        self.assertAlmostEqual(float(result[0]), 1.5, places=10)

    def test_exactly_aligned_two_parameter_case(self):
        a = np.array([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])
        b = np.array([[0.1, 0.2, 0.3]])
        result = model.bounded_min_distances(
            a, b, np.array([-0.5, -1.0]), np.array([0.5, 1.0])
        )
        self.assertAlmostEqual(float(result[0]), 0.3, places=10)

    def test_fluctuation_changes_asymptotic_codimension(self):
        rng = np.random.default_rng(33)
        aligned = model.sample_configuration(3, 0.0, 5000, rng)
        fluctuating = model.sample_configuration(3, 0.1, 5000, rng)
        self.assertEqual(aligned["codimension"], 1)
        self.assertEqual(fluctuating["codimension"], 0)
        self.assertGreater(aligned["radius_scan"][0]["bounded_patch_mc_probability"], 0)


if __name__ == "__main__":
    unittest.main()
