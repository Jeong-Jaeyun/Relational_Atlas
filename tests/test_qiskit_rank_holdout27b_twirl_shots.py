"""Checks for the stage-27b finite-shot twirl on/off family."""

import itertools
import json
import unittest

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout4_weaknull as additive
from relational_atlas_i01.holdout import qiskit_rank_holdout27b_twirl_shots as study


class DesignTests(unittest.TestCase):
    def test_exact_noncentralities_match_the_protocol(self):
        lambdas = study.checks()["exact_lambda"]
        for test, frozen in study.FROZEN_LAMBDA.items():
            self.assertAlmostEqual(lambdas[test], frozen, delta=0.06)
        self.assertGreater(min(lambdas.values()), 40)

    def test_every_circuit_seed_is_unique(self):
        seeds = [study.seed_for(c, k, label, block, axis, sign)
                 for c, k, label, block, axis, sign in itertools.product(
                     range(4), range(3), range(study.LABELS), range(2),
                     range(study.N), range(2))]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_twirl_splits_are_reproducible_and_balanced(self):
        first, second = study.split_table(), study.split_table()
        shots = study.cell_shots()
        for name, table in first.items():
            np.testing.assert_array_equal(table, second[name])
            self.assertTrue(np.all((table >= 0) & (table <= shots)))
            self.assertLess(abs(table.sum() / shots.sum() - 0.5), 1e-3)

    def test_contrast_matches_the_stage4_statistic_for_uniform_cells(self):
        rng = np.random.default_rng(3)
        alloc = study.allocation()
        counts = rng.binomial(np.broadcast_to(alloc, (2, study.N, study.LABELS)), 0.1)
        delta, variance = study.contrast(counts, study.cell_shots(),
                                         study.design_weights())
        reference_delta, reference_variance, _ = additive.statistic(
            counts[0], counts[1], alloc)
        np.testing.assert_allclose(delta, reference_delta, rtol=1e-12)
        np.testing.assert_allclose(variance, reference_variance, rtol=1e-12)

    def test_frozen_budget(self):
        self.assertEqual(study.SHOTS_PER_LABEL, 1 << 20)
        self.assertEqual(int(study.cell_shots().sum()) * 12, 528_482_304)


class SavedResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = json.loads(study.RESULT.read_text(encoding="utf-8"))

    def test_raw_tables_follow_the_frozen_design(self):
        tables = self.saved["tables"]
        self.assertEqual(len(tables), 12)
        self.assertEqual(study.total_preparations(tables), 528_482_304)
        splits = study.split_table()
        for key, table in tables.items():
            with self.subTest(table=key):
                joint, clock, shots = study.pooled(table)
                np.testing.assert_array_equal(shots, study.cell_shots())
                self.assertTrue(np.all(joint <= clock))
                if key.endswith("/twirl"):
                    np.testing.assert_array_equal(
                        table["shots_by_sign"]["1"], splits[key.split("/")[0]])

    def test_positive_family_reproduces_from_raw_counts(self):
        tables, weights = self.saved["tables"], study.design_weights()
        for test, name, first, second in study.POSITIVE:
            with self.subTest(test=test):
                a = study.contrast(*study.pooled(tables[f"{name}/{first}"])[::2],
                                   weights)
                b = study.contrast(*study.pooled(tables[f"{name}/{second}"])[::2],
                                   weights)
                statistic, _ = study.wald(*a, *b)
                stored = self.saved["analysis"]["positive_family"][test]
                self.assertAlmostEqual(float(statistic),
                                       stored["wald_chi_square_df3"], places=9)

    def test_recorded_outcome(self):
        analysis = self.saved["analysis"]
        self.assertTrue(self.saved["calibration"]["comparison_passed"])
        self.assertTrue(self.saved["calibration"]["weak_null_passed"])
        self.assertTrue(analysis["all_positive_passed"])
        self.assertTrue(analysis["all_null_passed"])
        self.assertTrue(analysis["all_secondary_passed"])
        self.assertEqual(analysis["withheld_by_calibration"], [])


if __name__ == "__main__":
    unittest.main()
