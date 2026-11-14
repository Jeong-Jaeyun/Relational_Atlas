"""Regression checks for the finite-clock, label, and anchor intervention."""

import math
import unittest

import numpy as np

from relational_atlas_i01.phase_reference import qiskit_clock_anchor_intervention as study
from relational_atlas_i01.phase_reference import qiskit_phase_anchor_audit as anchor
from relational_atlas_i01.phase_reference import qiskit_phase_label_intervention as labels


class ClockAnchorInterventionTests(unittest.TestCase):
    def test_clock_postselection_and_joint_probability(self):
        for physical in (False, True):
            entry = study.clock_model(physical)
            self.assertEqual(entry["joint_preparation"].num_qubits, 11)
            self.assertTrue(np.all((entry["energies"] >= 0) &
                                   (entry["energies"] < study.CLOCK_LEVELS)))
            check = study.exact_circuit_crosscheck(
                entry, math.pi / 2, labels.angle(0, 2, 0.3))
            self.assertAlmostEqual(check["clock_postselection_probability"], 0.2,
                                   places=11)
            self.assertAlmostEqual(check["joint_zero_probability"],
                                   check["formula_joint_zero_probability"], places=11)

    def test_factorized_conditional_probability(self):
        entry = study.clock_model(False)
        for tau in (0.0, math.pi / 4, math.pi / 2, math.pi):
            for common in (-0.15, 0.3, 0.65):
                factorized = study.full_conditional_probabilities(
                    np.array([common]), tau)[0]
                for t in range(labels.L):
                    direct, _ = study.conditional_probability_gradient(
                        entry, tau, labels.angle(0, t, common))
                    self.assertAlmostEqual(factorized[t], direct, places=11)

    def test_clock_conditioned_rank_interventions(self):
        for tau in (0.0, math.pi / 2, math.pi):
            for physical, expected_kept in ((False, 4), (True, 3)):
                entry = study.clock_model(physical)
                kept = study.rank_summary(study.conditional_fisher(entry, tau, 0.0, True))
                erased = study.rank_summary(study.conditional_fisher(entry, tau, 0.0, False))
                no_anchor = study.rank_summary(study.profiled_fisher(entry, tau, 0.0))
                with_anchor = study.rank_summary(study.profiled_fisher(entry, tau, 1.0))
                self.assertEqual(kept["rank_tol_1e_minus_8"], expected_kept)
                self.assertEqual(erased["rank_tol_1e_minus_8"], 3)
                self.assertEqual(no_anchor["rank_tol_1e_minus_8"], 3)
                self.assertEqual(with_anchor["rank_tol_1e_minus_8"], expected_kept)
                if physical:
                    self.assertAlmostEqual(kept["common_direction_information"], 0,
                                           delta=1e-10)
                else:
                    self.assertGreater(kept["common_direction_information"], 0.01)

    def test_synthetic_calibrated_fit_recovers_common_phase(self):
        tau = math.pi / 2
        accepted = np.full((labels.N, labels.L), 200_000, dtype=int)
        science = np.rint(accepted * study.full_conditional_probabilities(
            np.array([anchor.ALPHA_TRUE + anchor.OFFSETS[0]]), tau)[0]).astype(int)
        calibration = np.rint(accepted * study.full_conditional_probabilities(
            np.array([anchor.OFFSETS[0]]), tau)[0]).astype(int)
        estimate = (study.fit_full_counts(science, accepted, tau)["estimate_radians"] -
                    study.fit_full_counts(calibration, accepted, tau)["estimate_radians"])
        self.assertAlmostEqual(estimate, anchor.ALPHA_TRUE, delta=0.002)

    def test_dense_clock_scan_preserves_rank_on_sampled_grid(self):
        scan = study.clock_information_scan(points=31)
        self.assertEqual(len(scan["label_kept"]["unrestricted"]), 31)
        for name, expected_rank in (("unrestricted", 4), ("P_zero", 3)):
            rows = scan["label_kept"][name]
            self.assertEqual({row["rank_tol_1e_minus_8"] for row in rows},
                             {expected_rank})
            self.assertGreater(min(row["smallest_positive_eigenvalue"] for row in rows),
                               0.0)


if __name__ == "__main__":
    unittest.main()
