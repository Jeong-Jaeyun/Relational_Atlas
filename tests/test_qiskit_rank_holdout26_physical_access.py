"""Focused checks for the code-preserving access experiment."""

import unittest

import numpy as np
from qiskit_aer import AerSimulator

from relational_atlas_i01.holdout import qiskit_rank_holdout24_charge_graph as graph
from relational_atlas_i01.holdout import qiskit_rank_holdout25_invalid_extension as extension
from relational_atlas_i01.holdout import qiskit_rank_holdout26_physical_access as study


class PhysicalAccessTests(unittest.TestCase):
    def test_frozen_graph_counts(self):
        for n, expected in study.FROZEN_COUNTS.items():
            observed, odd = study.enumerate_counts(n, graph.CUTOFF)
            self.assertEqual(tuple(observed.values()), expected)
            self.assertEqual(study.predicted_counts(n, graph.CUTOFF), expected)
            self.assertEqual(odd, 0)

    def test_gate_stays_in_code_and_matches_sparse_prediction(self):
        n, cutoff = graph.N, graph.CUTOFF
        system = 2 * cutoff * n
        clock = graph.clock_bits(n, cutoff)
        prepared, reference, _, _ = graph.weights_and_vectors(n, cutoff)
        joint, _ = extension.joint_preparation(n, cutoff)
        valid = study.valid_code_indices(n, cutoff)
        angles = np.array([0.17, -0.61, 0.47])
        simulator = AerSimulator(method="statevector", max_parallel_threads=2)
        for name in ("same_bit", "other_same_sign"):
            _, source, return_ = next(row for row in extension.classes(cutoff)
                                       if row[0] == name)
            for phase_extension in extension.EXTENSIONS:
                phase = extension.phase_diagonal(n, cutoff, angles,
                                                 phase_extension)
                predicted, _ = study.analytic_probability(
                    prepared, reference, phase, source, return_, cutoff, clock)
                circuit = study.aer_circuit(joint, n, cutoff, source,
                                            return_, angles, phase_extension)
                result = simulator.run(circuit, shots=1).result()
                self.assertTrue(result.success)
                data = result.data(0)
                before = np.asarray(data["after_source"].data)
                final = np.asarray(data["final"].data)
                self.assertLess(study.invalid_mass(before, valid, system), 1e-12)
                self.assertLess(study.invalid_mass(final, valid, system), 1e-12)
                actual = abs(np.vdot(reference, final[:1 << system])) ** 2
                self.assertLess(abs(predicted - actual), graph.TOLERANCE)


if __name__ == "__main__":
    unittest.main()
