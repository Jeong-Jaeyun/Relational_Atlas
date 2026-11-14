"""Regression checks for the circuit-level I01 controls."""

import unittest

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from relational_atlas_i01.phase_reference import qiskit_condition_probe as probe


class QiskitConditionProbeTests(unittest.TestCase):
    def test_two_bit_momentum_encoding_and_phase(self):
        self.assertEqual(probe.basis_index(np.array([0, 0])), 0)
        self.assertEqual(probe.basis_index(np.array([1, -1])), 9)
        self.assertEqual(probe.basis_index(np.array([-1, 1])), 6)
        angles = np.array([0.19, -0.61, 0.38, 0.12])
        for physical in (False, True):
            entry = probe.model(4, physical)
            observed = Statevector.from_instruction(
                probe.phase_circuit(entry, angles)).probabilities()[0]
            self.assertAlmostEqual(observed, probe.return_probability(entry, angles), places=11)

    def test_constraint_and_detector_twirling_are_distinct(self):
        bias = 0.8
        for n in (3, 4, 5, 6):
            unrestricted = probe.model(n, False)
            constrained = probe.model(n, True)
            full_eigs = np.linalg.eigvalsh(probe.qfi_from_qiskit_state(unrestricted))
            gauge_eigs = np.linalg.eigvalsh(probe.qfi_from_qiskit_state(constrained))
            twirled_eigs = np.linalg.eigvalsh(probe.analytic_fisher(unrestricted, bias, True))
            self.assertEqual(int(np.sum(full_eigs > 1e-8)), n)
            self.assertEqual(int(np.sum(gauge_eigs > 1e-8)), n - 1)
            self.assertEqual(int(np.sum(twirled_eigs > 1e-8)), n - 1)
            self.assertEqual(len(unrestricted["momenta"]), 3 ** n)
            self.assertLess(np.max(np.abs(probe.analytic_fisher(constrained, bias, False)
                                          @ np.ones(n))), 1e-10)

    def test_charge_twirl_is_phase_reference_loss(self):
        entry = probe.model(4, False)
        angles = np.array([0.11, 0.34, -0.27, 0.48])
        alpha = 0.39
        phases = np.arange(9) * 2 * np.pi / 9
        average = lambda theta: np.mean([
            probe.return_probability(entry, theta + phase * np.ones(4))
            for phase in phases
        ])
        self.assertAlmostEqual(average(angles), average(angles + alpha), places=11)
        self.assertGreater(abs(probe.return_probability(entry, angles) -
                               probe.return_probability(entry, angles + alpha)), 0.01)

    def test_partial_reference_loss_does_not_create_exact_constraint(self):
        entry = probe.model(4, False)
        rows = probe.phase_reference_scan(entry, 0.8)
        self.assertEqual(rows[0]["mathematical_rank"], 4)
        self.assertEqual(rows[-1]["mathematical_rank"], 3)
        self.assertTrue(all(row["mathematical_rank"] == 4 for row in rows[:-1]))
        self.assertLess(rows[-2]["common_to_relative_eigenvalue_ratio"],
                        rows[1]["common_to_relative_eigenvalue_ratio"])
        transition = probe.apparent_rank_transition(entry, 0.8)
        self.assertGreater(transition, 0.95)
        self.assertLess(transition, 0.99)

    def test_aer_return_probability(self):
        entry = probe.model(4, True)
        angles = np.array([0.14, -0.47, 0.26, -0.11])
        circuit = probe.phase_circuit(entry, angles, measure=True)
        # Avoid the Aer 0.17.2 Windows multiplexer assembler crash.
        compiled = transpile(circuit, basis_gates=["u3", "cx"], optimization_level=1)
        result = AerSimulator(method="statevector", max_parallel_threads=1).run(
            compiled, shots=8192, seed_simulator=103
        ).result()
        observed = result.get_counts().get("0" * 8, 0) / 8192
        self.assertLess(abs(observed - probe.return_probability(entry, angles)), 0.025)


if __name__ == "__main__":
    unittest.main()
