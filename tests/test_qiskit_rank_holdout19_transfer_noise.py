"""Checks for the paired-transfer density-matrix noise experiment."""

import unittest

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from relational_atlas_i01.holdout import qiskit_rank_holdout19_transfer_noise as study


class TransferNoiseTests(unittest.TestCase):
    def test_frozen_cases_and_noise_grid(self):
        self.assertEqual(len(study.CASES), 4)
        self.assertEqual(study.LEVELS, (0.001, 0.01, 0.03))
        self.assertEqual(len(study.specs()), 10)

    def test_channel_precedes_the_paired_pulse_at_its_marker(self):
        base, theta, _ = study.circuit_model.template(3)
        for kind, marker, channels in (
            ("preparation_amplitude_damping", 1, 8),
            ("phase_dephasing", 2, 6),
        ):
            with self.subTest(kind=kind):
                circuit = study.build_circuit(base, theta, kind, 0.03,
                                              0, 2, 0, 1, 3)
                ops = [item.operation.name for item in circuit.data]
                barriers = [i for i, name in enumerate(ops) if name == "barrier"]
                self.assertEqual(ops[barriers[marker - 1] + 1:
                                     barriers[marker - 1] + 1 + channels],
                                 ["quantum_channel"] * channels)
                self.assertEqual(ops[barriers[marker - 1] + 1 + channels], "rx")

    def test_ideal_density_matrix_matches_untranspiled_statevector_event(self):
        base, theta, _ = study.circuit_model.template(3)
        circuit = study.build_circuit(base, theta, "ideal", 0.0,
                                      0, 2, 1, 0, 3)
        reference = Statevector.from_instruction(circuit).probabilities()
        circuit.save_probabilities()
        result = AerSimulator(method="density_matrix").run(circuit, shots=1).result()
        actual = np.asarray(result.data(0)["probabilities"])
        self.assertLess(np.max(np.abs(actual - reference)), 1e-12)
        joint, clock = study.noise.readout_events(actual, 0.0)
        self.assertAlmostEqual(joint, reference[0], places=12)
        self.assertAlmostEqual(clock, reference[:64].sum(), places=12)


if __name__ == "__main__":
    unittest.main()
