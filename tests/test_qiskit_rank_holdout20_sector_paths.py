"""Independent circuit checks of the sector-path amplitude formula."""

import math
import unittest

import numpy as np
from qiskit.quantum_info import Statevector

from relational_atlas_i01.holdout import qiskit_rank_holdout20_sector_paths as study
from relational_atlas_i01.holdout._clock_postselection import template as clock_template


class SectorPathTests(unittest.TestCase):
    def test_sector_support_creates_the_n3_same_sign_threshold(self):
        expected = {2: (3, 0, 0, 2), 3: (7, 0, 2, 2),
                    4: (19, 0, 4, 6)}
        for n, target in expected.items():
            prepared, reference = study.vectors(n, 0.45, 0.15)
            phases = study.phase_diagonal(n, np.arange(1, n + 1) * 0.31)
            paths = []
            for _, source, return_ in study.circuit_model.SUPPORTS:
                terms = study.amplitude_terms(prepared, reference, phases,
                                              source, return_)
                self.assertLess(abs(terms["source_only"]), 1e-15)
                self.assertLess(abs(terms["return_only"]), 1e-15)
                paths.append(terms["allowed_transfer_paths"])
            self.assertEqual(tuple(paths), target)

    def test_amplitude_formula_matches_raw_qiskit_for_an_unseen_setting(self):
        n, source, return_, angle = 3, 0, 2, 0.83
        angles = np.array([0.17, -0.61, 0.47])
        prepared, reference = study.vectors(n, 0.45, 0.15)
        predicted, _ = study.probability(
            n, prepared, reference, study.phase_diagonal(n, angles),
            source, return_, angle)
        base, theta, _ = study.circuit_model.template(n, transpile_circuit=False)
        circuit = base.assign_parameters({theta[i]: float(angles[i]) for i in range(n)})
        circuit = study.circuit_model.insert_paired_rx(circuit, source, return_, angle)
        actual = Statevector.from_instruction(circuit).probabilities()[0]
        self.assertTrue(math.isclose(predicted, actual, rel_tol=0, abs_tol=1e-12))

    def test_two_term_angle_law_reproduces_the_independent_aer_scan(self):
        coefficients = study.angle_law_coefficients(3, 0.45, 0.15)
        self.assertLess(study.angle_law_reproduction(coefficients), 1e-12)
        self.assertLess(max(
            entry["maximum_absolute_direct_block_difference"]
            for entry in coefficients.values()), 1e-12)

    def test_clock_phase_version_matches_raw_qiskit(self):
        n, tau, angle = 3, math.pi / 2, 0.83
        angles = np.array([0.17, -0.61, 0.47])
        prepared, reference = study.vectors(n, 0.45, 0.15, tau=tau)
        predicted, _ = study.probability(
            n, prepared, reference, study.phase_diagonal(n, angles),
            0, 2, angle)
        entry = study.circuit_model.grid.model(n, 0.45, True, "nonlinear")
        base, theta = clock_template(entry, tau, noise_markers=True)
        circuit = base.assign_parameters({theta[i]: float(angles[i]) for i in range(n)})
        circuit = study.circuit_model.insert_paired_rx(circuit, 0, 2, angle)
        actual = Statevector.from_instruction(circuit).probabilities()[0]
        self.assertTrue(math.isclose(predicted, actual, rel_tol=0, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
