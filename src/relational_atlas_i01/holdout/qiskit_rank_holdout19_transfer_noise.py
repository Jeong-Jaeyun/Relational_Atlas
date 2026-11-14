"""Exact density-matrix noise stress of the four paired-support classes.

The fixed n=3, tau=0, pi/2 paired-pulse circuit from stages 17--18 is used
throughout. Each local error is varied independently; no fitted correction is
applied. A checkpoint stores each complete scenario.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from qiskit_aer import AerSimulator

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout7_noise_stress as noise
from relational_atlas_i01.holdout import qiskit_rank_holdout15_dimension_transfer as circuit_model
from relational_atlas_i01.holdout import qiskit_rank_holdout17_transfer_shots as shots


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout19_transfer_noise.json"
CHECKPOINT = PROJECT_ROOT / "results" / "qiskit_rank_holdout19_transfer_noise_checkpoint.json"
N = 3
LABELS = 2 * N + 1
ANGLE = math.pi / 2
LEVELS = (0.001, 0.01, 0.03)
CASES = tuple((name, source, return_) for name, source, return_, _ in shots.CASES)
KINDS = noise.ERROR_KINDS


def specs() -> tuple[tuple[str, float], ...]:
    return (("ideal", 0.0),) + tuple(
        (kind, level) for kind in KINDS for level in LEVELS
    )


def scenario_key(kind: str, level: float) -> str:
    return f"{kind}_{level:.3f}"


def _load_checkpoint() -> dict:
    if not CHECKPOINT.exists():
        return {}
    saved = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if (saved.get("angle_radians") != ANGLE or
            saved.get("levels") != list(LEVELS) or
            saved.get("cases") != [list(row) for row in CASES]):
        return {}
    scenarios = saved.get("scenarios", {})
    return scenarios if isinstance(scenarios, dict) else {}


def _save_checkpoint(scenarios: dict) -> None:
    CHECKPOINT.write_text(json.dumps({
        "angle_radians": ANGLE,
        "levels": LEVELS,
        "cases": CASES,
        "scenarios": scenarios,
    }, indent=2), encoding="utf-8")


def _channel_qubits(kind: str) -> range:
    return range(8 if kind == "preparation_amplitude_damping" else 6)


def build_circuit(base, theta, kind: str, level: float, source: int,
                  return_: int, block: int, axis: int, label: int):
    phases = np.full(N, circuit_model.SHIFTS[block] + 2 * math.pi * label / LABELS)
    phases[axis] -= circuit_model.grid.BIAS
    bound = base.assign_parameters({theta[i]: float(phases[i]) for i in range(N)})
    paired = circuit_model.insert_paired_rx(bound, source, return_, ANGLE)
    instruction, marker = noise.channel(kind, level)
    # Noise at the first marker precedes the source Rx; at the second marker
    # it precedes the return Rx. Readout flips act only on final probabilities.
    return noise.insert_channel(paired, instruction, marker, _channel_qubits(kind))


def acquire_scenario(kind: str, level: float, workers: int, prepared=None) -> dict:
    """Evaluate all axes and labels, retaining the full event probabilities."""
    if workers < 1:
        raise ValueError("workers must be positive")
    base, theta, _ = circuit_model.template(N) if prepared is None else prepared
    simulator = AerSimulator(method="density_matrix", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    readout = level if kind == "symmetric_readout" else 0.0
    rows = {}
    for case, source, return_ in CASES:
        joint = np.zeros((2, N, LABELS))
        clock = np.zeros_like(joint)
        # Small batches keep the density-matrix memory bound, and each case
        # becomes independently inspectable in the scenario checkpoint.
        for label in range(LABELS):
            circuits = []
            settings = []
            for block in range(2):
                for axis in range(N):
                    circuit = build_circuit(base, theta, kind, level, source,
                                            return_, block, axis, label)
                    circuit.save_probabilities()
                    circuits.append(circuit)
                    settings.append((block, axis))
            result = simulator.run(circuits, shots=1).result()
            if not result.success:
                raise RuntimeError(result.status)
            for i, (block, axis) in enumerate(settings):
                values = np.asarray(result.data(i)["probabilities"], dtype=float)
                joint[block, axis, label], clock[block, axis, label] = (
                    noise.readout_events(values, readout))
        rows[case] = {"source": source, "return": return_,
                      "joint": joint.tolist(), "clock": clock.tolist()}
        print(f"{kind} p={level:.3f} case={case}", flush=True)
    return rows


def analyze_scenario(rows: dict) -> dict:
    weights = circuit_model.frozen_label_weights(LABELS)
    allocation = shots._allocation()
    analyzed = {}
    for case, _, _ in CASES:
        raw = rows[case]
        joint = np.asarray(raw["joint"], dtype=float)
        clock = np.asarray(raw["clock"], dtype=float)
        if joint.shape != (2, N, LABELS) or clock.shape != joint.shape:
            raise ValueError("invalid exact probability array")
        if np.any(joint < -1e-12) or np.any(joint > clock + 1e-12):
            raise ValueError("joint event exceeds clock event")
        delta = (joint[1] - joint[0]) @ weights
        first, second = joint[0] @ weights, joint[1] @ weights
        average = (first + second) / 2
        noncentrality = float(np.sum(delta ** 2 * allocation.sum() /
                                     (2 * average * (1 - average))))
        clock_delta = (clock[1] - clock[0]) @ weights
        analyzed[case] = {
            "axis_pooled_joint_difference": delta.tolist(),
            "pooled_joint_probability_first": first.tolist(),
            "pooled_joint_probability_second": second.tolist(),
            "stage17_budget_approx_noncentrality": noncentrality,
            "maximum_absolute_joint_difference": float(np.max(np.abs(delta))),
            "axis_pooled_clock_difference": clock_delta.tolist(),
            "maximum_absolute_clock_difference": float(np.max(np.abs(clock_delta))),
            "mean_clock_acceptance": float(np.mean(clock)),
        }
    return analyzed


def summarize(scenarios: dict) -> dict:
    ideal = scenarios[scenario_key("ideal", 0.0)]["analysis"]
    output = {}
    for kind, level in specs():
        key = scenario_key(kind, level)
        analysis = scenarios[key]["analysis"]
        null = analysis["conjugate_partner"]["maximum_absolute_joint_difference"]
        nonzero = {name: analysis[name]["maximum_absolute_joint_difference"]
                   for name, _, _ in CASES if name != "conjugate_partner"}
        output[key] = {
            "conjugate_partner_contrast": null,
            "minimum_nonzero_class_contrast": min(nonzero.values()),
            "null_to_minimum_nonzero_ratio": null / min(nonzero.values()),
            "signal_fraction_of_ideal": {
                name: value / ideal[name]["maximum_absolute_joint_difference"]
                for name, value in nonzero.items()
            },
            "maximum_clock_difference": max(
                row["maximum_absolute_clock_difference"]
                for row in analysis.values()),
            "stage17_budget_approx_noncentrality": {
                name: row["stage17_budget_approx_noncentrality"]
                for name, row in analysis.items()},
        }
    return output


def study(workers: int) -> dict:
    prepared = circuit_model.template(N)
    scenarios = _load_checkpoint()
    for kind, level in specs():
        key = scenario_key(kind, level)
        if key in scenarios:
            scenarios[key]["analysis"] = analyze_scenario(scenarios[key]["rows"])
            print(f"resumed {key}", flush=True)
            continue
        rows = acquire_scenario(kind, level, workers, prepared)
        scenarios[key] = {"kind": kind, "level": level, "rows": rows,
                          "analysis": analyze_scenario(rows)}
        _save_checkpoint(scenarios)
        print(f"completed {key}", flush=True)
    result = {
        "parameters": {
            "n": N, "angle_radians": ANGLE, "tau_over_pi": 0.0,
            "levels": LEVELS, "workers": workers,
            "phase_labels": LABELS,
            "label_distribution": "stage6_skewed_largest_remainder",
            "channel_locations": {
                "preparation_amplitude_damping": "after preparation, before source Rx, on all 8 qubits",
                "phase_dephasing": "after science Rz, before return Rx, on 6 system qubits",
                "symmetric_readout": "independent classical bit flip on all 8 output bits",
            },
        },
        "scenarios": scenarios,
        "summary": summarize(scenarios),
    }
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    result = study(args.workers)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
