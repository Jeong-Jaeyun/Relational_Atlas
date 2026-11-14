"""Regression checks for the Aer P=0 null-count audit."""

import unittest

import numpy as np

from relational_atlas_i01.phase_reference import qiskit_clock_anchor_intervention as clock
from relational_atlas_i01.phase_reference import qiskit_pzero_sampling_audit as audit
from relational_atlas_i01.phase_reference import qiskit_phase_label_intervention as labels


class PZeroSamplingAuditTests(unittest.TestCase):
    def test_exact_probabilities_match_alternate_common_phase(self):
        probabilities = audit.exact_joint_probabilities()
        self.assertEqual(probabilities.shape, (2, labels.N, labels.L))
        entry = audit.alias.model(True)
        for tau_index, tau in enumerate(audit.alias.SHOT_TAUS):
            for i in range(labels.N):
                for t in range(labels.L):
                    alternative, _ = clock.conditional_probability_gradient(
                        entry, tau, labels.angle(i, t, -0.43))
                    self.assertAlmostEqual(probabilities[tau_index, i, t],
                                           alternative / clock.CLOCK_LEVELS,
                                           places=11)

    def test_synthetic_panel_has_finite_diagnostics(self):
        shots = 1024
        probabilities = audit.exact_joint_probabilities()
        rng = np.random.default_rng(12)
        counts = rng.binomial(shots, probabilities[:, None, :, :],
                              size=(2, 2, labels.N, labels.L))
        source = {
            "parameters": {"shots_per_circuit": shots},
            "shots": {"P_zero_joint_zero_counts": counts.tolist(),
                      "tau_results": [
                          {"P_zero_hidden_mode_vs_absent_labeled": {
                              "permutation_pvalue": 0.5}} for _ in range(2)]},
            "P_zero_control_repeats": [],
        }
        result = audit.audit(source, panels=50, seed=12)
        self.assertEqual(result["seed_count"], 1)
        self.assertEqual(result["number_of_p_tests"], 2)
        self.assertTrue(np.isfinite(result["observed_variance_standardized_difference"]))


if __name__ == "__main__":
    unittest.main()
