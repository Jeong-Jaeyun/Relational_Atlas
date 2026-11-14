"""Checks for the finite-shot, stratified phase-label protocol."""

import math
import unittest

import numpy as np
from qiskit.quantum_info import Statevector

from relational_atlas_i01.holdout import qiskit_rank_holdout2_grid as grid
from relational_atlas_i01.holdout import qiskit_rank_holdout3_shots as shots


class RankHoldout3ShotTests(unittest.TestCase):
    def test_fixed_allocations_keep_total_and_are_skewed_only_when_requested(self):
        uniform = shots.shot_allocation(32768, "uniform")
        skewed = shots.shot_allocation(32768, "skewed")
        self.assertEqual(uniform.tolist(), [32768] * 7)
        self.assertEqual(int(skewed.sum()), 7 * 32768)
        self.assertGreater(int(skewed.max()), int(skewed.min()))

    def test_stratified_erasure_null_with_equal_arms(self):
        allocation = shots.shot_allocation(100, "skewed")
        observed = np.broadcast_to(allocation[None, :] // 3, (3, 7)).copy()
        result = shots.stratified_erased_pvalue(
            observed, observed, allocation, 99, np.random.default_rng(7))
        self.assertEqual(result["g_statistic"], 0.0)
        self.assertEqual(result["permutation_pvalue"], 1.0)

    def test_parametric_circuit_matches_independent_formula(self):
        entry = grid.model(3, 0.45, False, "nonlinear")
        qc, parameters = shots.template(entry, math.pi / 2)
        angles = np.full(3, 2 * math.pi / 7)
        angles[0] -= grid.BIAS
        bound = qc.assign_parameters({parameters[k]: float(angles[k])
                                      for k in range(3)})
        probability = abs(Statevector.from_instruction(
            bound.remove_final_measurements(inplace=False)).data[0]) ** 2
        expected, _ = grid.probability_gradient(entry, "mismatched",
                                                math.pi / 2, angles)
        self.assertAlmostEqual(probability, float(expected), places=10)


if __name__ == "__main__":
    unittest.main()
