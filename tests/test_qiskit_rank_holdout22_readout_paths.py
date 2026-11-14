"""Checks for the stage-22 readout-path attribution experiment."""

import json
import unittest

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout22_readout_paths as study


def _saved() -> dict:
    return json.loads(study.RESULT.read_text(encoding="utf-8"))


def _synthetic(seed: int, identical_blocks: bool = False) -> np.ndarray:
    values = np.random.default_rng(seed).random((2, 3, 7, 256))
    values /= values.sum(axis=-1, keepdims=True)
    if identical_blocks:
        values[1] = values[0]
        return values
    # Like the conjugate-partner null, give both blocks the same all-zero
    # probability: the module's linear prediction omits that zeroth-order term.
    rest = values[1, ..., 1:]
    rest *= ((1 - values[0, ..., :1]) / rest.sum(axis=-1, keepdims=True))
    values[1, ..., 0] = values[0, ..., 0]
    return values


class ReadoutPathTests(unittest.TestCase):
    def test_identical_blocks_give_no_contrast(self):
        analysis = study.contrasts(_synthetic(1, identical_blocks=True))
        np.testing.assert_array_equal(
            analysis["readout_first_derivative_at_zero"], 0)
        for level in analysis["levels"].values():
            np.testing.assert_array_equal(
                level["axis_pooled_joint_difference"], 0)

    def test_first_derivative_leaves_a_second_order_residual(self):
        levels = study.contrasts(_synthetic(2))["levels"]
        small = levels["0.001"]["maximum_linear_approximation_error"]
        large = levels["0.01"]["maximum_linear_approximation_error"]
        self.assertGreater(large / small, 50)
        self.assertLess(large / small, 200)

    def test_saved_leakage_is_dominated_by_the_return_bit(self):
        saved = _saved()
        analysis = saved["analysis"]
        derivative = np.asarray(analysis["readout_first_derivative_at_zero"])
        return_bit = np.asarray(
            analysis["single_one_outcome_contributions_by_qubit"]["1"])
        self.assertLess(np.max(np.abs(derivative - return_bit)),
                        1e-4 * np.max(np.abs(derivative)))
        self.assertAlmostEqual(
            analysis["levels"]["0.03"]["maximum_absolute_joint_difference"]
            / 1.651e-5, 1, delta=1e-3)
        error = study._stage19_error(analysis["levels"])
        self.assertIsNotNone(error)
        self.assertLess(error, 1e-15)

    def test_recomputed_contrasts_reproduce_saved_analysis(self):
        saved = _saved()["analysis"]
        fresh = study.contrasts(study.acquire(workers=2))
        np.testing.assert_allclose(fresh["readout_first_derivative_at_zero"],
                                   saved["readout_first_derivative_at_zero"],
                                   rtol=0, atol=1e-15)
        for level in study.LEVELS:
            with self.subTest(level=level):
                np.testing.assert_allclose(
                    fresh["levels"][str(level)]["axis_pooled_joint_difference"],
                    saved["levels"][str(level)]["axis_pooled_joint_difference"],
                    rtol=0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
