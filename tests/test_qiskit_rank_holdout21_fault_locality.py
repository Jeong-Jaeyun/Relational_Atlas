"""Checks for the stage-21 amplitude-damping fault-locality experiment."""

import json
import unittest

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout21_fault_locality as study


def _saved() -> dict:
    return json.loads(study.RESULT.read_text(encoding="utf-8"))


class FaultLocalityTests(unittest.TestCase):
    def test_frozen_subsets_partition_the_register(self):
        subsets = study.SUBSETS
        self.assertEqual(study.LEVELS, (0.001, 0.03))
        self.assertEqual(subsets["all_eight"], tuple(range(8)))
        self.assertEqual(sorted(subsets["source_only"] + subsets["return_only"]
                                + subsets["remote_system"]),
                         list(subsets["system_all"]))
        self.assertEqual(sorted(subsets["system_all"] + subsets["clock_only"]),
                         list(subsets["all_eight"]))

    def test_channel_is_inserted_only_on_the_requested_qubits(self):
        base, theta, _ = study.transfer.circuit_model.template(3)
        instruction, barrier = study.noise.channel(
            "preparation_amplitude_damping", 0.03)
        circuit = study.transfer.build_circuit(base, theta, "ideal", 0.0,
                                               0, 1, 0, 0, 0)
        for name, qubits in study.SUBSETS.items():
            with self.subTest(subset=name):
                noisy = study.noise.insert_channel(circuit, instruction,
                                                   barrier, qubits)
                touched = [noisy.find_bit(item.qubits[0]).index
                           for item in noisy.data
                           if item.operation.name == "quantum_channel"]
                self.assertEqual(touched, list(qubits))

    def test_saved_contrasts_follow_from_saved_rows(self):
        scenarios = _saved()["scenarios"]
        self.assertEqual(len(scenarios), len(study.LEVELS) * len(study.SUBSETS))
        weights = study.transfer.circuit_model.frozen_label_weights(7)
        for key, scenario in scenarios.items():
            with self.subTest(scenario=key):
                joint = np.asarray(scenario["rows"]["joint"])
                delta = (joint[1] - joint[0]) @ weights
                np.testing.assert_allclose(
                    delta, scenario["analysis"]["axis_pooled_joint_difference"],
                    rtol=0, atol=1e-20)

    def test_saved_values_support_the_topic_decision_claims(self):
        saved = _saved()
        low = saved["summary"]["0.001"]["maximum_null_by_subset"]
        high = saved["summary"]["0.03"]["maximum_null_by_subset"]
        # Damping the clock, the source q0 or the return q1 alone leaves the
        # conjugate-partner null at numerical zero.
        for name in ("clock_only", "source_only", "return_only"):
            with self.subTest(subset=name):
                self.assertLess(low[name], 2.5e-17)
                self.assertLess(high[name], 1e-16)
        # The leakage comes from the four remote system qubits.
        self.assertAlmostEqual(low["remote_system"] / 2.477e-7, 1, delta=1e-3)
        self.assertAlmostEqual(low["all_eight"] / 2.476e-7, 1, delta=1e-3)
        self.assertAlmostEqual(high["all_eight"] / 7.213e-6, 1, delta=1e-3)
        # The all-eight scenarios reproduce the independent stage-19 scan.
        error = study._stage19_crosscheck(saved["scenarios"])
        self.assertIsNotNone(error)
        self.assertLess(error, 1e-15)

    def test_recomputed_remote_scenario_reproduces_saved_rows(self):
        saved = _saved()["scenarios"]["remote_system_0.001"]
        fresh = study.acquire(study.SUBSETS["remote_system"], 0.001, workers=4)
        for kind in ("joint", "clock"):
            np.testing.assert_allclose(fresh["rows"][kind], saved["rows"][kind],
                                       rtol=0, atol=1e-13)
        np.testing.assert_allclose(
            fresh["analysis"]["axis_pooled_joint_difference"],
            saved["analysis"]["axis_pooled_joint_difference"],
            rtol=0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
