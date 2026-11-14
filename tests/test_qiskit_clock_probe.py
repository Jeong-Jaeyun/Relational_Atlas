"""Finite-clock and postselection regression checks for I01."""

import math
import unittest

import numpy as np

from relational_atlas_i01.phase_reference import qiskit_clock_probe as clock
from relational_atlas_i01.phase_reference import qiskit_condition_probe as probe


class QiskitClockProbeTests(unittest.TestCase):
    def test_energy_matched_constraint_and_clock_support(self):
        entry = clock.clock_model()
        self.assertEqual(entry["constraint_residual"], 0.0)
        self.assertEqual(sorted(set(entry["energies"])), [0, 2, 4])
        self.assertEqual(np.count_nonzero(entry["amplitudes"]), 19)

    def test_conditional_state_and_recurrence(self):
        entry = clock.clock_model()
        rows = clock.exact_study(entry)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(abs(row["clock_postselection_probability"] - 1 / 3) < 1e-11
                            for row in rows))
        self.assertTrue(all(abs(row["conditional_state_match_fidelity"] - 1) < 1e-11
                            for row in rows))
        self.assertAlmostEqual(rows[0]["conditional_return_fidelity"], 1.0, places=11)
        self.assertAlmostEqual(rows[-1]["conditional_return_fidelity"], 1.0, places=11)
        self.assertLess(rows[2]["conditional_return_fidelity"], 0.2)

    def test_pi_recurrence_requires_even_energy_support(self):
        full = probe.model(4, False)
        energies = np.sum(full["momenta"] ** 2, axis=1)
        full_fidelity = abs(np.sum(full["weights"] * np.exp(1j * energies * math.pi))) ** 2
        self.assertLess(full_fidelity, 0.9)
        self.assertAlmostEqual(clock.expected_fidelity(clock.clock_model(), math.pi),
                               1.0, places=11)


if __name__ == "__main__":
    unittest.main()
