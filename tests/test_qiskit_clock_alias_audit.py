"""Regression checks for phase-reference temporal aliasing."""

import math
import unittest

import numpy as np

from relational_atlas_i01.phase_reference import qiskit_clock_alias_audit as audit
from relational_atlas_i01.phase_reference import qiskit_clock_anchor_intervention as clock
from relational_atlas_i01.phase_reference import qiskit_phase_label_intervention as labels


class ClockAliasAuditTests(unittest.TestCase):
    def test_hidden_mode_is_zero_on_coarse_grid_only(self):
        for index in audit.COARSE_INDICES:
            self.assertAlmostEqual(audit.hidden_mode(audit.ANCHOR_TIMES[index]), 0,
                                   delta=1e-12)
        self.assertAlmostEqual(audit.hidden_mode(audit.SCIENCE_TIME), -1,
                               delta=1e-12)
        self.assertAlmostEqual(audit.drift(audit.SCIENCE_TIME) -
                               audit.drift(audit.SCIENCE_TIME, 0.0),
                               -audit.HIDDEN_AMPLITUDE, delta=1e-12)

    def test_exact_rank_requires_a_nonaliased_anchor_under_sine_model(self):
        for tau in (0.0, math.pi / 2):
            for physical, expected in ((False, (3, 4, 4)),
                                       (True, (3, 3, 3))):
                summaries = [audit.profiled_information(physical, tau, indices)
                             for indices in (audit.COARSE_INDICES,
                                             audit.COARSE_INDICES + (1,),
                                             tuple(range(9)))]
                actual = tuple(row["profiled"]["rank_tol_1e_minus_8"]
                               for row in summaries)
                self.assertEqual(actual, expected)

    def test_synthetic_counts_expose_interpolation_bias(self):
        phases = audit.segment_phases()
        accepted = np.full((labels.N, labels.L), 200_000, dtype=int)
        for tau in (0.0, math.pi / 2):
            fits = {}
            for name, value in [("science", phases["science"])] + [
                    (f"anchor_{j}", phase) for j, phase in enumerate(phases["anchors"])]:
                p = clock.full_conditional_probabilities(np.array([value]), tau)[0]
                fits[name] = clock.fit_full_counts(np.rint(accepted * p).astype(int),
                                                   accepted, tau)
            coarse = audit.contrast(fits, {"science": 1,
                                           "anchor_2": -0.5,
                                           "anchor_4": -0.5})
            direct = audit.contrast(fits, {"science": 1, "anchor_3": -1})
            holdout = audit.contrast(fits, {"anchor_1": 1,
                                            "anchor_0": -0.5,
                                            "anchor_2": -0.5})
            self.assertAlmostEqual(coarse["estimate_radians"],
                                   audit.ALPHA - audit.HIDDEN_AMPLITUDE,
                                   delta=0.002)
            self.assertAlmostEqual(direct["estimate_radians"], audit.ALPHA,
                                   delta=0.002)
            self.assertAlmostEqual(holdout["estimate_radians"],
                                   audit.HIDDEN_AMPLITUDE, delta=0.002)

    def test_P_zero_common_shift_invariance_and_schedule_size(self):
        entry = audit.model(True)
        phases = audit.segment_phases()
        for tau in (0.0, math.pi / 2):
            for i in range(labels.N):
                for t in range(labels.L):
                    actual, _ = clock.conditional_probability_gradient(
                        entry, tau, labels.angle(i, t,
                                                 phases["P_zero_science_true"]))
                    absent, _ = clock.conditional_probability_gradient(
                        entry, tau, labels.angle(i, t,
                                                 phases["P_zero_science_no_hidden_mode"]))
                    self.assertAlmostEqual(actual, absent, places=11)
        self.assertEqual(len(audit.schedule(False)), 720)
        self.assertEqual(len(audit.schedule(True)), 144)

    def test_binomial_null_audit_structure(self):
        calibration = audit.binomial_null_calibration(
            shots=1024, seed=17, replicates=3, permutations=19)
        self.assertEqual(calibration["replicates_per_tau"], 3)
        self.assertEqual(len(calibration["rows"]), 2)
        for row in calibration["rows"]:
            self.assertEqual(len(row["pvalues"]), 3)
            self.assertTrue(all(0 < p <= 1 for p in row["pvalues"]))


if __name__ == "__main__":
    unittest.main()
