"""Locate amplitude-damping-induced conjugate-partner null leakage."""

from __future__ import annotations

import argparse
import json

import numpy as np
from qiskit_aer import AerSimulator

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout7_noise_stress as noise
from relational_atlas_i01.holdout import qiskit_rank_holdout19_transfer_noise as transfer


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout21_fault_locality.json"
CHECKPOINT = PROJECT_ROOT / "results" / "qiskit_rank_holdout21_fault_locality_checkpoint.json"
LEVELS = (0.001, 0.03)
SUBSETS = {
    "clock_only": (6, 7),
    "source_only": (0,),
    "return_only": (1,),
    "remote_system": (2, 3, 4, 5),
    "system_all": (0, 1, 2, 3, 4, 5),
    "all_eight": tuple(range(8)),
}


def acquire(subset: tuple[int, ...], level: float, workers: int,
            prepared=None) -> dict:
    base, theta, _ = (transfer.circuit_model.template(3)
                      if prepared is None else prepared)
    instruction, barrier = noise.channel("preparation_amplitude_damping", level)
    simulator = AerSimulator(method="density_matrix", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    joint = np.zeros((2, 3, 7))
    clock = np.zeros_like(joint)
    for label in range(7):
        circuits = []
        settings = []
        for block in range(2):
            for axis in range(3):
                circuit = transfer.build_circuit(base, theta, "ideal", 0.0,
                                                 0, 1, block, axis, label)
                circuit = noise.insert_channel(circuit, instruction, barrier,
                                               subset)
                circuit.save_probabilities()
                circuits.append(circuit)
                settings.append((block, axis))
        result = simulator.run(circuits, shots=1).result()
        if not result.success:
            raise RuntimeError(result.status)
        for i, (block, axis) in enumerate(settings):
            values = np.asarray(result.data(i)["probabilities"], dtype=float)
            joint[block, axis, label], clock[block, axis, label] = (
                noise.readout_events(values, 0.0))
    weights = transfer.circuit_model.frozen_label_weights(7)
    delta = (joint[1] - joint[0]) @ weights
    clock_delta = (clock[1] - clock[0]) @ weights
    return {
        "rows": {"joint": joint.tolist(), "clock": clock.tolist()},
        "analysis": {
            "axis_pooled_joint_difference": delta.tolist(),
            "maximum_absolute_joint_difference": float(np.max(np.abs(delta))),
            "axis_pooled_clock_difference": clock_delta.tolist(),
            "maximum_absolute_clock_difference": float(np.max(
                np.abs(clock_delta))),
            "mean_clock_acceptance": float(np.mean(clock)),
        },
    }


def _load() -> dict:
    if not CHECKPOINT.exists():
        return {}
    saved = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if saved.get("levels") != list(LEVELS) or saved.get("subsets") != (
            {name: list(qubits) for name, qubits in SUBSETS.items()}):
        return {}
    return saved.get("scenarios", {})


def _save(scenarios: dict) -> None:
    CHECKPOINT.write_text(json.dumps({"levels": LEVELS, "subsets": SUBSETS,
                                      "scenarios": scenarios}, indent=2),
                          encoding="utf-8")


def _stage19_crosscheck(scenarios: dict) -> float | None:
    if not transfer.CHECKPOINT.exists():
        return None
    reference = json.loads(transfer.CHECKPOINT.read_text(encoding="utf-8"))[
        "scenarios"]
    error = 0.0
    for level in LEVELS:
        key = f"all_eight_{level:.3f}"
        other = transfer.scenario_key("preparation_amplitude_damping", level)
        if other not in reference:
            return None
        here = scenarios[key]["analysis"]["axis_pooled_joint_difference"]
        there = reference[other]["analysis"]["conjugate_partner"][
            "axis_pooled_joint_difference"]
        error = max(error, float(np.max(np.abs(np.asarray(here) - there))))
    return error


def study(workers: int) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    prepared = transfer.circuit_model.template(3)
    scenarios = _load()
    for level in LEVELS:
        for name, qubits in SUBSETS.items():
            key = f"{name}_{level:.3f}"
            if key in scenarios:
                print(f"resumed {key}", flush=True)
                continue
            scenarios[key] = {"qubits": qubits, "level": level,
                              **acquire(qubits, level, workers, prepared)}
            _save(scenarios)
            print(f"completed {key}", flush=True)
    scale = {}
    for level in LEVELS:
        entries = {name: scenarios[f"{name}_{level:.3f}"]["analysis"]
                   for name in SUBSETS}
        local_sum = sum((np.asarray(entries[name]["axis_pooled_joint_difference"])
                         for name in ("source_only", "return_only",
                                      "remote_system")), np.zeros(3))
        system = np.asarray(entries["system_all"]["axis_pooled_joint_difference"])
        scale[str(level)] = {
            "maximum_null_by_subset": {
                name: entry["maximum_absolute_joint_difference"]
                for name, entry in entries.items()},
            "local_sum_minus_system_all_maximum": float(np.max(
                np.abs(local_sum - system))),
            "clock_only_maximum_clock_block_difference":
                entries["clock_only"]["maximum_absolute_clock_difference"],
        }
    output = {"parameters": {"n": 3, "source": 0, "return": 1,
                             "angle_radians": transfer.ANGLE,
                             "levels": LEVELS, "subsets": SUBSETS,
                             "workers": workers},
              "scenarios": scenarios, "summary": scale,
              "maximum_difference_from_stage19_all_eight":
                  _stage19_crosscheck(scenarios)}
    RESULT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    output = study(args.workers)
    print(json.dumps({"summary": output["summary"],
                      "stage19_reproduction_error": output[
                          "maximum_difference_from_stage19_all_eight"]},
                     indent=2))


if __name__ == "__main__":
    main()
