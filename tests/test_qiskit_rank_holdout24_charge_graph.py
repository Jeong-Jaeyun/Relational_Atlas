"""Regression tests for preregistered cutoff-two support and amplitude rules."""

import math
import unittest

import numpy as np
from qiskit_aer import AerSimulator

from relational_atlas_i01.holdout import qiskit_rank_holdout24_charge_graph as study


class ChargeGraphTests(unittest.TestCase):
    def test_coefficient_and_encoded_graph_agree_with_frozen_table(self):
        for (cutoff, n), expected in study.FROZEN_COUNTS.items():
            direct, largest_odd = study.direct_graph_counts(n, cutoff)
            self.assertEqual(tuple(direct.values()), expected)
            self.assertEqual(study.predicted_counts(n, cutoff), expected)
            self.assertEqual(largest_odd, 0)

    def test_cutoff_two_analytic_return_matches_independent_aer(self):
        n, cutoff = 3, 2
        system = 2 * cutoff * n
        clock = study.clock_bits(n, cutoff)
        prepared, reference, indices, energies = study.weights_and_vectors(n, cutoff)
        first = np.exp(-study.BETA * energies.astype(float))
        first /= first.sum()
        joint = np.zeros(1 << (system + clock), dtype=complex)
        joint[indices | (energies << system)] = np.sqrt(first)
        angles = np.array([0.17, -0.61, 0.47])
        simulator = AerSimulator(method="statevector", max_parallel_threads=2)
        for _, source, return_ in study.CLASSES:
            predicted, _ = study.analytic_probability(
                prepared, reference, study.phase_diagonal(n, cutoff, angles),
                source, return_, study.ANGLE, clock)
            circuit = study.aer_circuit(joint, n, cutoff, source,
                                        return_, angles, study.ANGLE)
            result = simulator.run(circuit, shots=1).result()
            self.assertTrue(result.success)
            state = np.asarray(result.data(0)["statevector"].data)
            observed = abs(np.vdot(reference, state[:1 << system])) ** 2
            self.assertTrue(math.isclose(predicted, observed,
                                         rel_tol=0, abs_tol=study.TOLERANCE))


if __name__ == "__main__":
    unittest.main()
