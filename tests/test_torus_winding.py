"""Geometry and conditional-gate checks for the periodic winding-pair toy."""

import unittest

import numpy as np

from relational_atlas_i01.geometry import torus_winding as model


class TorusWindingTests(unittest.TestCase):
    def test_periodic_image_contact(self):
        images = model.periodic_images(2)
        offset = np.array([[0.49, 0.0]])
        drift = np.array([[0.10, 0.0]])
        tangent = np.zeros((1, 2))
        squared = model.affine_patch_squared_distances(offset, drift, tangent, images)
        self.assertAlmostEqual(float(squared[0]), 0.41**2, places=10)

    def test_shape_is_periodic_and_bounded(self):
        rng = np.random.default_rng(4)
        geometry = model.make_geometry(4, 30, rng)
        nodes = model.profile_nodes(geometry, 0.15, 12)
        self.assertTrue(np.allclose(nodes[:, 0], nodes[:, -1], atol=1e-12))
        self.assertLessEqual(float(np.max(np.abs(nodes))), 0.151)

    def test_zero_shape_is_resolution_independent(self):
        rng = np.random.default_rng(7)
        geometry = model.make_geometry(3, 20, rng)
        low = model.pair_minimum_distances(geometry, 0.0, 6)
        high = model.pair_minimum_distances(geometry, 0.0, 24)
        self.assertTrue(np.allclose(low, high))

    def test_contact_and_annihilation_are_separate(self):
        self.assertEqual(model.pair_clearance(0.8, 20, 0.0), 0.0)
        self.assertGreater(model.pair_clearance(0.8, 20, 1.0), 0.99)
        p = model.stage_clearance(0.1, 3, 2, 10)
        self.assertGreaterEqual(p, 0)
        self.assertLessEqual(p, 1)

    def test_zero_relative_drift_removes_generic_exact_intersection(self):
        rng = np.random.default_rng(43)
        geometry = model.make_geometry(3, 1000, rng, drift_max=0.0)
        self.assertTrue(np.all(geometry["drift"] == 0))
        distances = model.pair_minimum_distances(geometry, 0.15)
        self.assertEqual(int(np.sum(distances <= 1e-7)), 0)


if __name__ == "__main__":
    unittest.main()
