"""Checks that the invalid-word intervention changes only its target."""

import unittest

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout24_charge_graph as graph
from relational_atlas_i01.holdout import qiskit_rank_holdout25_invalid_extension as study


class InvalidPhaseExtensionTests(unittest.TestCase):
    def test_extensions_agree_on_physical_codes_and_at_cutoff_one(self):
        self.assertTrue(np.array_equal(study.local_charge_table(1, "additive"),
                                       study.local_charge_table(1, "neutral_invalid")))
        for value in range(-2, 3):
            word = graph.basis_index((value,), 2)
            self.assertEqual(study.local_charge_table(2, "additive")[word],
                             study.local_charge_table(2, "neutral_invalid")[word])

    def test_cutoff_two_intermediate_charge_can_differ(self):
        plus_one_and_two = 0b0011
        self.assertEqual(study.local_charge_table(2, "additive")[plus_one_and_two], 3)
        self.assertEqual(study.local_charge_table(2, "neutral_invalid")[plus_one_and_two], 0)

    def test_cross_class_is_invariant_at_unseen_phase(self):
        n, cutoff = 3, 2
        prepared, reference, _, _ = graph.weights_and_vectors(n, cutoff)
        angles = np.array([0.17, -0.61, 0.47])
        for name, source, return_ in study.classes(cutoff):
            probabilities = []
            for extension in study.EXTENSIONS:
                value, _ = graph.analytic_probability(
                    prepared, reference,
                    study.phase_diagonal(n, cutoff, angles, extension),
                    source, return_, graph.ANGLE, graph.clock_bits(n, cutoff))
                probabilities.append(value)
            if name != "same_bit":
                self.assertAlmostEqual(*probabilities, places=12)


if __name__ == "__main__":
    unittest.main()
