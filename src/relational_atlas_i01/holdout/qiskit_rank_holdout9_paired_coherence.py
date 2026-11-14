"""Stage-9 paired sector-mixing test for the P=0 reference projection."""

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
from relational_atlas_i01.holdout import qiskit_rank_holdout8_symmetry_breaking as stage8
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


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout9_paired_coherence.json"
PROTOCOL = "notes/prevalidation_protocol9_2026-10-04.tex"
N = previous.N
LABELS = previous.LABELS
ANGLES = (0.01, 0.03, 0.1)
FAULT_QUBIT = stage8.FAULT_QUBIT


def paired_rx_after_barriers(circuit: QuantumCircuit, angle: float,
                             qubit: int = FAULT_QUBIT) -> QuantumCircuit:
    """Mix sectors after preparation and recombine before reference projection."""
    if not 0 <= qubit < circuit.num_qubits:
        raise ValueError("fault qubit is outside the circuit")
    output = circuit.copy_empty_like()
    barriers = 0
    inserted = 0
    for item in circuit.data:
        output.append(item.operation, item.qubits, item.clbits)
        if item.operation.name == "barrier":
            barriers += 1
            if barriers == 1:
                output.rx(angle, output.qubits[qubit])
                inserted += 1
            elif barriers == 2:
                output.rx(-angle, output.qubits[qubit])
                inserted += 1
    if inserted != 2:
        raise ValueError("both marked barriers are required")
    return output


def full_axis_probabilities(angle: float, workers: int,
                            templates: dict | None = None
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Run every axis, label, and block for a paired coherent intervention."""
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
            circuits = []
            settings = []
            for shift_index, shift in enumerate(previous.SHIFTS):
                for axis in range(N):
                    for label in range(LABELS):
                        phases = np.full(
                            N, shift + 2 * math.pi * label / LABELS)
                        phases[axis] -= stage7.grid.BIAS
                        circuit = paired_rx_after_barriers(
                            bind(base, theta, phases), angle)
                        circuit.save_probabilities()
                        circuits.append(circuit)
                        settings.append((shift_index, axis, label))
            result = simulator.run(circuits, shots=1).result()
            if not result.success:
                raise RuntimeError(result.status)
            for index, setting in enumerate(settings):
                probabilities = np.asarray(
                    result.data(index)["probabilities"], dtype=float)
                joint[physical, tau_index][setting] = probabilities[0]
                clock[physical, tau_index][setting] = np.sum(
                    probabilities[:1 << (2 * N)])
    return joint, clock


def circuit_checks() -> dict:
    """Check that the opposite rotations bracket the scientific phase layer."""
    base, _ = stage7.compiled_templates()[(0, 0)]
    angle = ANGLES[0]
    circuit = paired_rx_after_barriers(base, angle)
    data = list(circuit.data)
    barriers = [index for index, item in enumerate(data)
                if item.operation.name == "barrier"]
    rotations = [(index, float(item.operation.params[0]))
                 for index, item in enumerate(data)
                 if item.operation.name == "rx"]
    if len(barriers) != 2 or rotations != [
            (barriers[0] + 1, angle), (barriers[1] + 1, -angle)]:
        raise AssertionError("paired rotations do not bracket phase layer")
    return {
        "fault_qubit": FAULT_QUBIT,
        "first_rotation": rotations[0][1],
        "second_rotation": rotations[1][1],
        "rotation_indices": [index for index, _ in rotations],
        "barrier_indices": barriers,
    }


def study(workers: int) -> dict:
    """Run the predeclared paired-coherence sweep."""
    templates = stage7.compiled_templates()
    ideal_joint = stage6.exact_joint_probabilities()
    ideal_rows = condition_rows(ideal_joint,
                                np.full_like(ideal_joint, CLOCK_ACCEPTANCE))
    scenarios = [{
        "intervention_angle": 0.0,
        "rows": ideal_rows,
        "summary": scenario_summary(ideal_rows, ideal_rows),
    }]
    for angle in ANGLES:
        joint, clock = full_axis_probabilities(angle, workers, templates)
        rows = condition_rows(joint, clock)
        scenarios.append({
            "intervention_angle": angle,
            "rows": rows,
            "summary": scenario_summary(rows, ideal_rows),
        })
    return {
        "protocol": PROTOCOL,
        "parameters": {
            "n": N,
            "phase_labels": LABELS,
            "workers": workers,
            "fault_qubit": FAULT_QUBIT,
            "paired_angles_radians": list(ANGLES),
            "all_axes_explicitly_evaluated": True,
            "operation": (
                "Rx(angle) immediately after preparation, then Rx(-angle) "
                "after the scientific phase layer and before reference "
                "projection."
            ),
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
            "This is a deliberately coherent two-sided intervention, not a "
            "single passive error channel or calibrated processor noise. It "
            "tests whether a reference can recombine amplitudes from sectors "
            "that the P=0 projector otherwise filters out."
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
            "intervention_angle": scenario["intervention_angle"],
            **scenario["summary"],
        }
        for scenario in output["scenarios"]
    ], indent=2))


if __name__ == "__main__":
    main()

