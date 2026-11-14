"""Technical checks for post-shot calibration and label reweighting."""

import json
import unittest

from relational_atlas_i01.holdout import qiskit_rank_holdout3_audit as audit


class RankHoldout3AuditTests(unittest.TestCase):
    def test_three_degree_chi_square_survival(self):
        self.assertAlmostEqual(audit.chi_square_three_survival(0.0), 1.0)
        self.assertLess(audit.chi_square_three_survival(100.0), 1e-15)

    def test_uniform_target_exactly_removes_common_phase(self):
        source = json.loads(audit.SOURCE.read_text(encoding="utf-8"))
        result = audit.audit(source, panels=100, seed=9)
        self.assertEqual(len(result["reweighted_uniform_target"]), 8)
        self.assertLess(max(row["exact_uniform_target_max_difference"]
                            for row in result["reweighted_uniform_target"]), 1e-10)
        self.assertGreater(result["standardized_count_variance"], 0)


if __name__ == "__main__":
    unittest.main()
