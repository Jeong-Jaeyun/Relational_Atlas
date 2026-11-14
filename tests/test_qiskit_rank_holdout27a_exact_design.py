"""Checks for the stage-27a shot-free design module."""

import json
import math
import unittest

import numpy as np
from qiskit_aer import AerSimulator

from relational_atlas_i01.holdout import qiskit_rank_holdout_grid as prevalidation1
from relational_atlas_i01.holdout import qiskit_rank_holdout20_sector_paths as sector
from relational_atlas_i01.holdout import qiskit_rank_holdout27a_exact_design as study


def _vectors():
    return sector.vectors(study.N, study.BETA, study.REFERENCE_BETA)


def _weights():
    return study.model15.frozen_label_weights(study.LABELS)


class LabelAliasingTests(unittest.TestCase):
    def test_alias_term_reproduces_a_stored_short_label_failure(self):
        stored = json.loads(prevalidation1.RESULT.read_text(encoding="utf-8"))
        row = next(r for r in stored["rows"]
                   if r["n"] == 3 and r["phase_labels"] == 5 and r["beta"] == 0.45
                   and r["bias"] == 1.25 and r["tau_over_pi"] == 0.0
                   and not r["physical_P_zero"])
        entry = prevalidation1.model(3, 0.45, False)
        predicted = max(abs(study.alias_change(entry, 0.0, 1.25, 5, axis))
                        for axis in range(3))
        self.assertGreater(predicted, 1e-3)
        self.assertAlmostEqual(
            predicted, row["fisher"]["erased_common_shift_max_probability_change"],
            delta=1e-12)

    def test_adequate_labels_and_p_zero_states_cannot_alias(self):
        for physical, labels in ((False, 7), (True, 5)):
            entry = prevalidation1.model(3, 0.45, physical)
            for axis in range(3):
                self.assertEqual(
                    study.alias_change(entry, 0.0, 1.25, labels, axis), 0.0)

    def test_alias_term_scales_as_cos_tau(self):
        # Post hoc: the two aliasing charge pairs differ in energy by +1 and -1.
        entry = prevalidation1.model(4, 1.2, False)
        base = study.alias_change(entry, 0.0, 0.55, 7, 1)
        self.assertGreater(abs(base), 1e-6)
        for tau in (math.pi / 3, math.pi / 2, 2.0):
            self.assertAlmostEqual(study.alias_change(entry, tau, 0.55, 7, 1),
                                   math.cos(tau) * base, delta=1e-15)


class PulseSignTwirlTests(unittest.TestCase):
    def test_sign_average_keeps_only_the_transfer_squared_term(self):
        prepared, reference = _vectors()
        law = sector.angle_law_coefficients(study.N, study.BETA,
                                            study.REFERENCE_BETA)
        angle = math.pi / 2
        for name, source, return_ in study.CLASSES:
            with self.subTest(support=name):
                plus = study.signed_cutoff_one(source, return_, angle, 1,
                                               prepared, reference)
                minus = study.signed_cutoff_one(source, return_, angle, -1,
                                                prepared, reference)
                twirled = study.pooled_contrast((plus + minus) / 2, _weights())
                expected = (math.sin(angle / 2) ** 4 * np.asarray(
                    law[name]["transfer_squared_coefficient_D"]))
                np.testing.assert_allclose(twirled, expected, rtol=0, atol=1e-16)

    def test_flipped_return_sign_matches_aer(self):
        prepared, reference = _vectors()
        sparse = study.signed_cutoff_one(0, 3, math.pi / 2, -1, prepared, reference)
        aer = study.aer_cutoff_one(
            study.model15.template(study.N),
            lambda _angles: study.pulses(0, 3, math.pi / 2, -1),
            AerSimulator(method="statevector"))
        self.assertLess(np.max(np.abs(sparse - aer)), 1e-11)


class RepetitionTests(unittest.TestCase):
    def test_single_repetition_equals_single_access(self):
        prepared, reference = _vectors()
        for name, source, return_ in study.CLASSES:
            with self.subTest(support=name):
                np.testing.assert_allclose(
                    study.repeated_grid(source, return_, math.pi / 2, 1,
                                        prepared, reference),
                    study.signed_cutoff_one(source, return_, math.pi / 2, 1,
                                            prepared, reference),
                    rtol=0, atol=1e-15)

    def test_two_repetitions_open_self_return_paths_for_the_conjugate(self):
        prepared, reference = _vectors()
        self.assertEqual(study.path_census(0, 1, prepared),
                         {"direct_transfer_paths": 0,
                          "self_return_paths_k_ge_2": 14})
        single = study.pooled_contrast(study.repeated_grid(
            0, 1, math.pi / 2, 1, prepared, reference), _weights())
        double = study.pooled_contrast(study.repeated_grid(
            0, 1, math.pi / 2, 2, prepared, reference), _weights())
        self.assertLess(np.max(np.abs(single)), 1e-15)
        self.assertGreater(np.max(np.abs(double)), 1e-4)


class DiagonalNoiseLemmaTests(unittest.TestCase):
    def test_random_diagonal_phases_preserve_the_conjugate_null(self):
        # Independent sparse check of the lemma, not reusing the Aer path.
        prepared, reference = _vectors()
        rng = np.random.default_rng(7)
        first = np.exp(1j * rng.normal(0, 0.5, len(prepared)))
        second = np.exp(1j * rng.normal(0, 0.5, len(prepared)))
        angle = math.pi / 2
        values = np.zeros((2, study.N, study.LABELS))
        for block, axis, label, angles in study.cutoff_one_settings():
            state = study.rx_state(first * prepared, 0, angle)
            state = second * sector.phase_diagonal(study.N, angles) * state
            state = study.rx_state(state, 1, -angle)
            values[block, axis, label] = abs(np.vdot(reference, state)) ** 2
        self.assertLess(np.max(np.abs(study.pooled_contrast(values, _weights()))),
                        1e-15)


class SavedResultTests(unittest.TestCase):
    def test_all_preregistered_criteria_passed(self):
        saved = json.loads(study.RESULT.read_text(encoding="utf-8"))
        self.assertTrue(saved["all_preregistered_checks_passed"])
        self.assertEqual(len(saved["criteria_summary"]), 17)
        self.assertTrue(all(saved["criteria_summary"].values()))
        self.assertEqual(saved["A_label_aliasing"]["nonzero_by_tau_over_pi"],
                         {"0.0": 16, "0.5": 0})


if __name__ == "__main__":
    unittest.main()
