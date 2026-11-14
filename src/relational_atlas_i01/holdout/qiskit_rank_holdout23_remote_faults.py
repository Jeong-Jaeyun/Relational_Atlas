"""Resolve the conjugate-null amplitude-damping response by remote qubit."""

from __future__ import annotations

import argparse
import json

import numpy as np

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout21_fault_locality as locality


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout23_remote_faults.json"
CHECKPOINT = PROJECT_ROOT / "results" / "qiskit_rank_holdout23_remote_faults_checkpoint.json"
QUBITS = (2, 3, 4, 5)
LEVELS = locality.LEVELS


def _load() -> dict:
    if not CHECKPOINT.exists():
        return {}
    saved = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if saved.get("qubits") != list(QUBITS) or saved.get("levels") != list(LEVELS):
        return {}
    return saved.get("scenarios", {})


def _save(scenarios: dict) -> None:
    CHECKPOINT.write_text(json.dumps({"qubits": QUBITS, "levels": LEVELS,
                                      "scenarios": scenarios}, indent=2),
                          encoding="utf-8")


def study(workers: int) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    prepared = locality.transfer.circuit_model.template(3)
    scenarios = _load()
    for level in LEVELS:
        for qubit in QUBITS:
            key = f"q{qubit}_{level:.3f}"
            if key in scenarios:
                print(f"resumed {key}", flush=True)
                continue
            scenarios[key] = {"qubit": qubit, "level": level,
                              **locality.acquire((qubit,), level, workers,
                                                 prepared)}
            _save(scenarios)
            print(f"completed {key}", flush=True)
    reference = None
    if locality.RESULT.exists():
        reference = json.loads(locality.RESULT.read_text(encoding="utf-8"))
    summaries = {}
    for level in LEVELS:
        rows = {qubit: scenarios[f"q{qubit}_{level:.3f}"]["analysis"]
                for qubit in QUBITS}
        total = sum((np.asarray(row["axis_pooled_joint_difference"])
                     for row in rows.values()), np.zeros(3))
        remote = None
        if reference is not None:
            remote = np.asarray(reference["scenarios"][
                f"remote_system_{level:.3f}"]["analysis"][
                    "axis_pooled_joint_difference"])
        summaries[str(level)] = {
            "per_qubit_maximum_contrast": {
                str(qubit): row["maximum_absolute_joint_difference"]
                for qubit, row in rows.items()},
            "sum_single_qubit_differences": total.tolist(),
            "maximum_difference_from_remote_all": (
                None if remote is None else float(np.max(abs(total - remote)))),
        }
    output = {"parameters": {"n": 3, "partner": [0, 1],
                             "qubits": QUBITS, "levels": LEVELS,
                             "workers": workers},
              "scenarios": scenarios, "summary": summaries}
    RESULT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(study(args.workers)["summary"], indent=2))


if __name__ == "__main__":
    main()
