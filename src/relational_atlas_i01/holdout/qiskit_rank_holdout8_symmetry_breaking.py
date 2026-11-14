"""Stage-8 symmetry-breaking fault test for the explicit-clock experiment.

Stage 7 found no null leakage from phase-covariant, label-independent error
channels. This test deliberately breaks those two protection conditions one
at a time: a local coherent preparation rotation and a label-dependent
readout response.
"""

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
from relational_atlas_i01.holdout._clock_analysis import (
    CONDITIONS,
    SEPARATION_RATIO,
    condition_rows,
    scenario_summary,
)
from relational_atlas_i01.holdout._clock_postselection import (
    CLOCK_ACCEPTANCE,
    bind,
)


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout8_symmetry_breaking.json"
PROTOCOL = "notes/prevalidation_protocol8_2026-10-04.tex"
N = previous.N
LABELS = previous.LABELS
ROTATION_ANGLES = (0.01, 0.03, 0.1)
READOUT_LEVELS = (0.001, 0.01, 0.03)
LABEL_MODULATION = 0.8
FAULT_QUBIT = 0


def scenario_specs() -> tuple[tuple[str, float], ...]:
    """Return one ideal scenario and the two frozen symmetry-breaking sweeps."""
    return (("ideal", 0.0),) + tuple(
        ("coherent_preparation_rx", angle) for angle in ROTATION_ANGLES
    ) + tuple(
        ("label_dependent_readout", level) for level in READOUT_LEVELS
    )


def insert_rx_after_preparation(circuit: QuantumCircuit, angle: float,
                                qubit: int = FAULT_QUBIT) -> QuantumCircuit:
    """Insert one coherent local fault after the first preserved barrier."""
    if not 0 <= qubit < circuit.num_qubits:
        raise ValueError("fault qubit is outside the circuit")
    output = circuit.copy_empty_like()
    barriers = 0
    inserted = False
    for item in circuit.data:
        output.append(item.operation, item.qubits, item.clbits)
        if item.operation.name == "barrier":
            barriers += 1
            if barriers == 1:
                output.rx(angle, output.qubits[qubit])
                inserted = True
    if not inserted:
        raise ValueError("preparation barrier was not found")
    return output


def label_readout_error(level: float, label: int) -> float:
    """Frozen label-dependent flip profile with mean level and positive rates."""
    if not 0 <= label < LABELS or not 0 <= level < 1:
        raise ValueError("invalid label or readout level")
    error = level * (1 + LABEL_MODULATION *
                     math.cos(2 * math.pi * label / LABELS))
    if not 0 <= error < 1:
        raise ValueError("label-dependent error must lie in [0, 1)")
    return error


def full_axis_probabilities(workers: int, fault_angle: float = 0.0,
                            label_dependent_level: float = 0.0,
                            templates: dict | None = None
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate every stage-6 axis, label, and block with one selected fault."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if fault_angle and label_dependent_level:
        raise ValueError("fault mechanisms must remain one at a time")
    templates = stage7.compiled_templates() if templates is None else templates
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    joint = np.zeros((2, 2, 2, N, LABELS))
    clock = np.zeros_like(joint)
    for physical in (0, 1):
        for tau_index in range(2):
            base, theta = templates[(physical, tau_index)]
            circuits = []
            settings = []
            for shift_index, shift in enumerate(previous.SHIFTS):
                for axis in range(N):
                    for label in range(LABELS):
                        angles = np.full(
                            N, shift + 2 * math.pi * label / LABELS)
                        angles[axis] -= stage7.grid.BIAS
                        circuit = bind(base, theta, angles)
                        if fault_angle:
                            circuit = insert_rx_after_preparation(
                                circuit, fault_angle)
                        circuit.save_probabilities()
                        circuits.append(circuit)
                        settings.append((shift_index, axis, label))
            result = simulator.run(circuits, shots=1).result()
            if not result.success:
                raise RuntimeError(result.status)
            for index, (shift_index, axis, label) in enumerate(settings):
                probabilities = np.asarray(
                    result.data(index)["probabilities"], dtype=float)
                error = (label_readout_error(label_dependent_level, label)
                         if label_dependent_level else 0.0)
                joint_value, clock_value = stage7.readout_events(
                    probabilities, error)
                joint[physical, tau_index, shift_index, axis, label] = joint_value
                clock[physical, tau_index, shift_index, axis, label] = clock_value
    return joint, clock


def circuit_checks() -> dict:
    """Check frozen fault placement and label-error profile without sampling."""
    templates = stage7.compiled_templates()
    base, _ = templates[(0, 0)]
    injected = insert_rx_after_preparation(base, ROTATION_ANGLES[0])
    names = [item.operation.name for item in injected.data]
    first_barrier = names.index("barrier")
    if names[first_barrier + 1] != "rx":
        raise AssertionError("coherent fault was not placed after preparation")
    profile = np.array([label_readout_error(0.01, label)
                        for label in range(LABELS)])
    if not np.isclose(profile.mean(), 0.01):
        raise AssertionError("label profile does not have its declared mean")
    return {
        "fault_qubit": FAULT_QUBIT,
        "fault_instruction_after_first_barrier": names[first_barrier + 1],
        "label_readout_profile_at_mean_0_point_01": profile.tolist(),
        "profile_mean": float(profile.mean()),
        "profile_minimum": float(profile.min()),
        "profile_maximum": float(profile.max()),
    }


def study(workers: int) -> dict:
    """Run the predeclared one-fault-at-a-time Aer statevector sweep."""
    templates = stage7.compiled_templates()
    ideal_rows = condition_rows(stage6.exact_joint_probabilities(),
                                np.full_like(stage6.exact_joint_probabilities(),
                                             CLOCK_ACCEPTANCE))
    scenarios = []
    for kind, level in scenario_specs():
        if kind == "ideal":
            joint = stage6.exact_joint_probabilities()
            clock = np.full_like(joint, CLOCK_ACCEPTANCE)
        elif kind == "coherent_preparation_rx":
            joint, clock = full_axis_probabilities(
                workers, fault_angle=level, templates=templates)
        elif kind == "label_dependent_readout":
            joint, clock = full_axis_probabilities(
                workers, label_dependent_level=level, templates=templates)
        else:
            raise AssertionError("unexpected frozen scenario")
        rows = condition_rows(joint, clock)
        scenarios.append({
            "fault_kind": kind,
            "fault_strength": level,
            "rows": rows,
            "summary": scenario_summary(rows, ideal_rows),
        })
    return {
        "protocol": PROTOCOL,
        "parameters": {
            "n": N,
            "phase_labels": LABELS,
            "workers": workers,
            "faults_are_one_at_a_time": True,
            "coherent_preparation_rx": (
                "One Rx(angle) on system qubit 0 immediately after correlated "
                "state preparation and before the scientific phase layer."
            ),
            "label_dependent_readout": (
                "Per-label independent symmetric bit flip with error e_l = "
                "e[1 + 0.8 cos(2 pi l / 7)] on every output qubit."
            ),
            "rotation_angles_radians": list(ROTATION_ANGLES),
            "mean_label_readout_error_levels": list(READOUT_LEVELS),
            "all_axes_explicitly_evaluated": True,
            "conditions": [
                {"label_distribution": distribution,
                 "physical_P_zero": physical,
                 "tau_over_pi": previous.TAUS[tau_index] / math.pi,
                 "role": role}
                for distribution, physical, tau_index, role in CONDITIONS
            ],
            "predeclared_negligible_leakage_ratio": SEPARATION_RATIO,
        },
        "circuit_checks": circuit_checks(),
        "scenarios": scenarios,
        "caveat": (
            "Exact-probability Aer fault model. The coherent pulse and the "
            "harmonic label response are deliberately chosen counterexamples, "
            "not calibrated noise models or finite-shot p-value evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    checks = circuit_checks()
    if args.checks_only:
        print(json.dumps(checks, indent=2))
        return
    output = study(args.workers)
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps([
        {
            "fault_kind": scenario["fault_kind"],
            "fault_strength": scenario["fault_strength"],
            **scenario["summary"],
        }
        for scenario in output["scenarios"]
    ], indent=2))


if __name__ == "__main__":
    main()

