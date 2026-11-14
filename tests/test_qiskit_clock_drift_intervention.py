"""Regression checks for intra-block drift and midpoint calibration."""

import math
import unittest

import numpy as np

from relational_atlas_i01.phase_reference import qiskit_clock_drift_intervention as drift
from relational_atlas_i01.phase_reference import qiskit_clock_anchor_intervention as clock
from relational_atlas_i01.phase_reference import qiskit_phase_label_intervention as labels


class ClockDriftInterventionTests(unittest.TestCase):
    def test_endpoints_hide_curvature_but_midpoint_reveals_it(self):
        self.assertAlmostEqual(drift.drift(0.0), drift.drift(0.0, 0.0))
        self.assertAlmostEqual(drift.drift(1.0), drift.drift(1.0, 0.0))
        self.assertAlmostEqual(drift.drift(0.5) - drift.drift(0.5, 0.0),
                               drift.CURVATURE)
        phases = drift.segment_phases()
        self.assertAlmostEqual(phases["science_mid"], 0.62)
        self.assertAlmostEqual(phases["anchor_mid"], 0.32)

    def test_curvature_nuisance_changes_identifiability(self):
        for tau in (0.0, math.pi / 2, math.pi):
            assumed_affine = drift.profiled_information(False, tau, False, False)
            endpoints = drift.profiled_information(False, tau, False, True)
            midpoint = drift.profiled_information(False, tau, True, True)
            physical = drift.profiled_information(True, tau, True, True)
            self.assertEqual(assumed_affine["profiled"]["rank_tol_1e_minus_8"], 4)
            self.assertEqual(endpoints["profiled"]["rank_tol_1e_minus_8"], 3)
            self.assertEqual(midpoint["profiled"]["rank_tol_1e_minus_8"], 4)
            self.assertEqual(physical["profiled"]["rank_tol_1e_minus_8"], 3)
            self.assertAlmostEqual(endpoints["common_direction_information"], 0,
                                   delta=1e-9)
            self.assertGreater(midpoint["common_direction_information"], 0)

    def test_synthetic_segment_fits_expose_false_affine_assumption(self):
        accepted = np.full((labels.N, labels.L), 200_000, dtype=int)
        phases = drift.segment_phases()
        for tau in (0.0, math.pi / 2):
            fits = {}
            for name in drift.SEGMENTS:
                p = clock.full_conditional_probabilities(np.array([phases[name]]), tau)[0]
                joint = np.rint(accepted * p).astype(int)
                fits[name] = clock.fit_full_counts(joint, accepted, tau)
            naive = drift.independent_contrast(
                fits, {"science_mid": 1, "anchor_start": -0.5, "anchor_end": -0.5})
            corrected = drift.independent_contrast(
                fits, {"science_mid": 1, "anchor_mid": -1})
            curvature = drift.independent_contrast(
                fits, {"anchor_mid": 1, "anchor_start": -0.5, "anchor_end": -0.5})
            self.assertAlmostEqual(naive["estimate_radians"],
                                   drift.ALPHA + drift.CURVATURE, delta=0.002)
            self.assertAlmostEqual(corrected["estimate_radians"], drift.ALPHA,
                                   delta=0.002)
            self.assertAlmostEqual(curvature["estimate_radians"], drift.CURVATURE,
                                   delta=0.002)

    def test_P_zero_is_insensitive_to_both_common_phases(self):
        entry = drift.model(True)
        phases = drift.segment_phases()
        for tau in (0.0, math.pi / 2):
            for i in range(labels.N):
                for t in range(labels.L):
                    p_curved, _ = clock.conditional_probability_gradient(
                        entry, tau, labels.angle(i, t, phases["P_zero_science_curved"]))
                    p_affine, _ = clock.conditional_probability_gradient(
                        entry, tau, labels.angle(i, t, phases["P_zero_science_affine"]))
                    self.assertAlmostEqual(p_curved, p_affine, places=11)


if __name__ == "__main__":
    unittest.main()
