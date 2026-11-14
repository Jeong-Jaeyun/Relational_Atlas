"""Exact Aer atlas for the conditions creating paired-reference leakage."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout3_shots as previous
from relational_atlas_i01.holdout import qiskit_rank_holdout6_clock as stage6
from relational_atlas_i01.holdout import qiskit_rank_holdout7_noise_stress as stage7
from relational_atlas_i01.holdout._clock_analysis import condition_rows, scenario_summary
from relational_atlas_i01.holdout._clock_postselection import CLOCK_ACCEPTANCE, bind


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout12_intervention_atlas.json"
PROTOCOL = "notes/prevalidation_protocol12_2026-10-04.tex"
ANGLE = 1.0
N, LABELS = previous.N, previous.LABELS

# name, first axis/qubit, second axis/qubit, second signed-angle multiplier.
SCENARIOS = (
    ("one_sided_rx_q0", "x", 0, None, None, None),
    ("paired_rx_inverse_same_q0", "x", 0, "x", 0, -1.0),
    ("paired_ry_inverse_same_q0", "y", 0, "y", 0, -1.0),
    ("paired_rz_inverse_same_q0", "z", 0, "z", 0, -1.0),
    ("paired_rx_inverse_q0_to_q1", "x", 0, "x", 1, -1.0),
    ("paired_rx_same_sign_q0", "x", 0, "x", 0, 1.0),
)


def scenario_specs() -> tuple[tuple, ...]:
    """Return the frozen placement, axis, support, and sign interventions."""
    return SCENARIOS


def _rotate(circuit: QuantumCircuit, axis: str, angle: float, qubit: int) -> None:
    """Apply a supported single-qubit rotation."""
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be x, y, or z")
    if not 0 <= qubit < circuit.num_qubits:
        raise ValueError("qubit is outside the circuit")
    getattr(circuit, f"r{axis}")(angle, circuit.qubits[qubit])


def insert_intervention(circuit: QuantumCircuit, spec: tuple,
                        angle: float = ANGLE) -> QuantumCircuit:
    """Insert rotations immediately after the two preserved science barriers."""
    _, first_axis, first_qubit, second_axis, second_qubit, multiplier = spec
    output = circuit.copy_empty_like()
    barriers, first_done, second_done = 0, False, False
    for item in circuit.data:
        output.append(item.operation, item.qubits, item.clbits)
        if item.operation.name != "barrier":
            continue
        barriers += 1
        if barriers == 1:
            _rotate(output, first_axis, angle, first_qubit)
            first_done = True
        elif barriers == 2 and second_axis is not None:
            _rotate(output, second_axis, multiplier * angle, second_qubit)
            second_done = True
    if barriers != 2 or not first_done or (second_axis is not None) != second_done:
        raise ValueError("intervention could not be placed at both science barriers")
    return output


def full_axis_probabilities(spec: tuple, workers: int,
                            templates: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate every block, axis, label, and physical condition exactly."""
    if workers < 1:
        raise ValueError("workers must be positive")
    templates = stage7.compiled_templates() if templates is None else templates
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    joint = np.zeros((2, 2, 2, N, LABELS))
    clock = np.zeros_like(joint)
    for physical in (0, 1):
        for tau_index in range(2):
            base, theta = templates[(physical, tau_index)]
            circuits, settings = [], []
            for shift_index, shift in enumerate(previous.SHIFTS):
                for axis in range(N):
                    for label in range(LABELS):
                        phases = np.full(N, shift + 2 * math.pi * label / LABELS)
                        phases[axis] -= stage7.grid.BIAS
                        circuit = insert_intervention(bind(base, theta, phases), spec)
                        circuit.save_probabilities()
                        circuits.append(circuit)
                        settings.append((shift_index, axis, label))
            result = simulator.run(circuits, shots=1).result()
            if not result.success:
                raise RuntimeError(result.status)
            for index, setting in enumerate(settings):
                probabilities = np.asarray(result.data(index)["probabilities"], dtype=float)
                joint[physical, tau_index][setting] = probabilities[0]
                clock[physical, tau_index][setting] = probabilities[:1 << (2 * N)].sum()
    return joint, clock


def _constraint_row(rows: list[dict]) -> dict:
    return next(row for row in rows if row["role"] == "constraint_null_tau_0")


def circuit_checks() -> dict:
    """Verify all scenario rotations occur directly after their declared barriers."""
    base, _ = stage7.compiled_templates()[(0, 0)]
    details = {}
    for spec in SCENARIOS:
        data = list(insert_intervention(base, spec, 0.1).data)
        barriers = [i for i, item in enumerate(data) if item.operation.name == "barrier"]
        rotations = [i for i, item in enumerate(data)
                     if item.operation.name in {"rx", "ry", "rz"}]
        expected = [barriers[0] + 1]
        if spec[3] is not None:
            expected.append(barriers[1] + 1)
        if len(barriers) != 2 or rotations != expected:
            raise AssertionError(f"invalid placement for {spec[0]}")
        details[spec[0]] = {"barrier_indices": barriers, "rotation_indices": rotations}
    return details


def study(workers: int) -> dict:
    """Run the frozen exact-Aer condition atlas."""
    templates = stage7.compiled_templates()
    ideal_joint = stage6.exact_joint_probabilities()
    ideal_rows = condition_rows(ideal_joint, np.full_like(ideal_joint, CLOCK_ACCEPTANCE))
    scenarios = []
    for spec in SCENARIOS:
        joint, clock = full_axis_probabilities(spec, workers, templates)
        rows = condition_rows(joint, clock)
        constraint = _constraint_row(rows)
        scenarios.append({
            "name": spec[0],
            "intervention": {"angle_radians": ANGLE, "first_axis": spec[1],
                             "first_qubit": spec[2], "second_axis": spec[3],
                             "second_qubit": spec[4], "second_angle_multiplier": spec[5]},
            "rows": rows,
            "summary": scenario_summary(rows, ideal_rows),
            "constraint_tau_0": {
                "maximum_absolute_joint_difference": constraint["maximum_absolute_joint_difference"],
                "stage6_budget_approx_noncentrality": constraint["stage6_budget_approx_noncentrality"],
            },
        })
    return {
        "protocol": PROTOCOL,
        "parameters": {"workers": workers, "angle_radians": ANGLE,
                       "all_axes_explicitly_evaluated": True,
                       "scenarios_fixed_before_scan": [spec[0] for spec in SCENARIOS]},
        "circuit_checks": circuit_checks(),
        "scenarios": scenarios,
        "caveat": "Exact statevector map of coherent interventions; not a calibrated noise model or finite-shot evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.checks_only:
        print(json.dumps(circuit_checks(), indent=2))
        return
    output = study(args.workers)
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([
        {"name": row["name"], **row["summary"], **row["constraint_tau_0"]}
        for row in output["scenarios"]
    ], indent=2))


if __name__ == "__main__":
    main()
