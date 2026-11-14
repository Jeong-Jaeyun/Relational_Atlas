"""Guard the package layout and CLI against future drift."""

import importlib
import sys
import unittest
from unittest.mock import patch

from relational_atlas_i01 import cli
from relational_atlas_i01.paths import PROJECT_ROOT, RESULTS_DIR


class PackageStructureTests(unittest.TestCase):
    def test_project_paths_do_not_depend_on_working_directory(self):
        self.assertEqual(PROJECT_ROOT / "results", RESULTS_DIR)
        self.assertTrue((PROJECT_ROOT / "notes").is_dir())
        self.assertTrue((PROJECT_ROOT / "pyproject.toml").is_file())

    def test_discovery_contains_all_migrated_experiments(self):
        available = cli.experiments()
        self.assertGreaterEqual(len(available), 49)
        self.assertEqual(available["condition_atlas"],
                         "relational_atlas_i01.geometry.condition_atlas")
        self.assertEqual(available["qiskit_rank_holdout6_clock"],
                         "relational_atlas_i01.holdout.qiskit_rank_holdout6_clock")
        self.assertEqual(available["qiskit_rank_holdout7_noise_stress"],
                         "relational_atlas_i01.holdout."
                         "qiskit_rank_holdout7_noise_stress")
        self.assertEqual(available["qiskit_rank_holdout11_paired_shots"],
                         "relational_atlas_i01.holdout."
                         "qiskit_rank_holdout11_paired_shots")
        for stage, name in ((19, "transfer_noise"), (20, "sector_paths"),
                            (21, "fault_locality"), (22, "readout_paths"),
                            (23, "remote_faults"), (24, "charge_graph"),
                            (25, "invalid_extension"),
                            (26, "physical_access")):
            module = f"qiskit_rank_holdout{stage}_{name}"
            self.assertEqual(available[module],
                             f"relational_atlas_i01.holdout.{module}")

    def test_every_discovered_experiment_has_an_entry_point(self):
        for name, module_path in cli.experiments().items():
            with self.subTest(experiment=name):
                module = importlib.import_module(module_path)
                self.assertTrue(callable(getattr(module, "main", None)))

    def test_dispatch_forwards_experiment_options(self):
        from relational_atlas_i01.holdout import qiskit_rank_holdout6_clock

        saved = sys.argv
        with patch.object(qiskit_rank_holdout6_clock, "main") as run:
            self.assertEqual(cli.main(["qiskit_rank_holdout6_clock",
                                       "--checks-only"]), 0)
            run.assert_called_once_with()
        self.assertIs(sys.argv, saved)


if __name__ == "__main__":
    unittest.main()
