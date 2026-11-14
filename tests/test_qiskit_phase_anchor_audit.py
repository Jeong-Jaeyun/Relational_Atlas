"""Regression checks for calibration as an explicit phase-reference resource."""

import unittest

import numpy as np

from relational_atlas_i01.phase_reference import qiskit_condition_probe as probe
from relational_atlas_i01.phase_reference import qiskit_phase_anchor_audit as audit
from relational_atlas_i01.phase_reference import qiskit_phase_label_intervention as intervention


class PhaseAnchorAuditTests(unittest.TestCase):
    def test_factorized_full_probability_matches_state_model(self):
        entry = probe.model(4, False)
        for common in (-0.15, 0.0, 0.3, 0.65):
            factorized = audit.full_return_probabilities(np.array([common]))[0]
            for t in range(intervention.L):
                direct = probe.return_probability(entry, intervention.angle(0, t, common))
                self.assertAlmostEqual(factorized[t], direct, places=11)

    def test_nuisance_null_and_anchor_recovery(self):
        no_anchor = audit.profile_nuisance(False, 0.0)
        with_anchor = audit.profile_nuisance(False, 1.0)
        physical = audit.profile_nuisance(True, 1.0)
        self.assertEqual(no_anchor["profiled_rank_tol_1e_minus_8"], 3)
        self.assertLess(no_anchor["joint_null_residual_for_theta_plus_offset_trade"],
                        1e-10)
        self.assertEqual(with_anchor["profiled_rank_tol_1e_minus_8"], 4)
        self.assertGreater(with_anchor["common_direction_information"], 0.1)
        self.assertEqual(physical["profiled_rank_tol_1e_minus_8"], 3)

    def test_known_phase_anchor_separates_science_phase_and_offset(self):
        shots = 100000
        science = np.rint(shots * audit.full_return_probabilities(
            np.array([audit.ALPHA_TRUE + audit.OFFSETS[0]]))[0]).astype(int)
        anchor = np.rint(shots * audit.full_return_probabilities(
            np.array([audit.OFFSETS[0]]))[0]).astype(int)
        science_counts = np.repeat(science[None, :], intervention.N, axis=0)
        anchor_counts = np.repeat(anchor[None, :], intervention.N, axis=0)
        science_fit = audit.fit_known_phase_reference(science_counts, shots)
        anchor_fit = audit.fit_known_phase_reference(anchor_counts, shots)
        recovered = (science_fit["estimated_common_phase_radians"] -
                     anchor_fit["estimated_common_phase_radians"])
        self.assertAlmostEqual(recovered, audit.ALPHA_TRUE, delta=0.002)

    def test_unrecorded_jitter_reduces_common_information(self):
        entry = probe.model(4, False)
        common = np.ones(4) / 2
        clean = audit.labeled_fisher_at_offset(entry, 0.0, 0.0)
        noisy = audit.labeled_fisher_at_offset(entry, 0.0, 0.4)
        self.assertGreater(float(common @ clean @ common),
                           float(common @ noisy @ common))


if __name__ == "__main__":
    unittest.main()
