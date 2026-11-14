"""Qiskit checks for the second frozen holdout protocol."""

import unittest

from relational_atlas_i01.holdout import qiskit_rank_holdout2_circuit as circuit


class RankHoldout2CircuitTests(unittest.TestCase):
    def test_mismatched_projection_against_aer_density_matrix(self):
        result = circuit.check_case(circuit.CASES[0])
        self.assertLess(result["statevector_absolute_error"], 1e-10)
        self.assertLess(result["aer_density_matrix_absolute_error"], 1e-10)

    def test_charge_twirled_mixed_effect(self):
        result = circuit.check_case(circuit.CASES[3])
        self.assertLess(result["statevector_absolute_error"], 1e-10)

    def test_skewed_pooling_against_independent_circuit(self):
        result = circuit.check_skewed_pooling()
        self.assertLess(result["maximum_label_probability_error"], 1e-10)


if __name__ == "__main__":
    unittest.main()
