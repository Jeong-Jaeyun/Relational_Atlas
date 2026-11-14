"""Regression tests for the generalized explicit-clock transfer scan."""

import unittest

from relational_atlas_i01.holdout import qiskit_rank_holdout15_dimension_transfer as study


class DimensionTransferTests(unittest.TestCase):
    def test_dimensions_and_support_classes_are_frozen(self):
        self.assertEqual(study.DIMENSIONS, (2, 3, 4))
        self.assertEqual([name for name, *_ in study.SUPPORTS], [
            "same_support", "conjugate_partner", "different_pair_same_sign",
            "different_pair_opposite_sign"])

    def test_generic_clock_width_and_barrier_placement(self):
        self.assertEqual(study.clock_qubits(2), 2)
        self.assertEqual(study.clock_qubits(3), 2)
        self.assertEqual(study.clock_qubits(4), 3)
        checks = study.circuit_checks()
        self.assertEqual(checks["4"]["total_qubits"], 11)
        self.assertEqual(len(checks["3"]["rotation_indices"]), 2)

    def test_n3_weight_rule_matches_the_frozen_stage6_allocation(self):
        expected = study.previous.shot_allocation(
            study.stage6.SHOTS_PER_LABEL, "skewed")
        self.assertTrue((study.frozen_label_weights(7) * expected.sum() == expected).all())

    def test_untranspiled_template_is_available_for_backend_crosschecks(self):
        circuit, _, _ = study.template(3, transpile_circuit=False)
        self.assertTrue(any(item.operation.name == "state_preparation"
                            for item in circuit.data))


if __name__ == "__main__":
    unittest.main()
