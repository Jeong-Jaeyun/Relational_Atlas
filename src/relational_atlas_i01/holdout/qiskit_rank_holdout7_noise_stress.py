"""Stage-7 localized-noise stress test for the explicit-clock intervention.

This is an exact-probability Aer density-matrix study. It asks whether the
stage-6 contrast remains separated from two distinct exact-null controls when
preparation, phase, and readout errors are introduced one at a time.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import amplitude_damping_error, phase_damping_error

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout2_grid as grid
from relational_atlas_i01.holdout import qiskit_rank_holdout3_shots as previous
from relational_atlas_i01.holdout import qiskit_rank_holdout6_clock as stage6
from relational_atlas_i01.holdout._clock_postselection import (
    CLOCK_ACCEPTANCE,
    CLOCK_QUBITS,
    bind,
    template,
)


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout7_noise_stress.json"
PROTOCOL = "notes/prevalidation_protocol7_2026-10-04.tex"
N = previous.N
SYSTEM_QUBITS = 2 * N
TOTAL_QUBITS = SYSTEM_QUBITS + CLOCK_QUBITS
LABELS = previous.LABELS
LEVELS = (0.001, 0.01, 0.03)
ERROR_KINDS = (
    "preparation_amplitude_damping",
    "phase_dephasing",
    "symmetric_readout",
)
CONDITIONS = (
    ("skewed", False, 0, "strong_signal"),
    ("skewed", False, 1, "weak_signal"),
    ("uniform", False, 0, "distribution_null_tau_0"),
    ("uniform", False, 1, "distribution_null_tau_half_pi"),
    ("skewed", True, 0, "constraint_null_tau_0"),
    ("skewed", True, 1, "constraint_null_tau_half_pi"),
)
SEPARATION_RATIO = 0.1


def scenario_specs(levels: tuple[float, ...] = LEVELS) -> tuple[tuple[str, float], ...]:
    """Return the sole ideal reference plus each one-error stress scenario."""
    if not levels or any(not 0 < level < 1 for level in levels):
        raise ValueError("noise levels must lie strictly between zero and one")
    return (("ideal", 0.0),) + tuple(
        (kind, level) for kind in ERROR_KINDS for level in levels)


def channel(kind: str, level: float):
    """Return the one-qubit channel and the barrier after which it belongs."""
    if kind == "preparation_amplitude_damping":
        return amplitude_damping_error(level).to_instruction(), 1
    if kind == "phase_dephasing":
        return phase_damping_error(level).to_instruction(), 2
    if kind in ("ideal", "symmetric_readout"):
        return None, None
    raise ValueError(f"unknown error kind: {kind}")


def insert_channel(circuit: QuantumCircuit, instruction, after_barrier: int,
                   qubits: range) -> QuantumCircuit:
    """Insert an Aer channel immediately after a preserved barrier."""
    if instruction is None:
        return circuit.copy()
    output = circuit.copy_empty_like()
    barrier = 0
    inserted = False
    for item in circuit.data:
        output.append(item.operation, item.qubits, item.clbits)
        if item.operation.name == "barrier":
            barrier += 1
            if barrier == after_barrier:
                for qubit in qubits:
                    output.append(instruction, [output.qubits[qubit]])
                inserted = True
    if not inserted:
        raise ValueError(f"barrier {after_barrier} was not found")
    return output


def readout_events(probabilities: np.ndarray, error: float) -> tuple[float, float]:
    """Apply a symmetric independent classical readout flip to exact outcomes."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (1 << TOTAL_QUBITS,):
        raise ValueError("unexpected probability-vector shape")
    if not 0 <= error < 1:
        raise ValueError("readout error must lie in [0, 1)")
    indices = np.arange(probabilities.size, dtype=np.uint16)
    all_weights = np.array([
        error ** int(value).bit_count() *
        (1 - error) ** (TOTAL_QUBITS - int(value).bit_count())
        for value in indices
    ])
    clock_bits = indices >> SYSTEM_QUBITS
    clock_weights = np.array([
        error ** int(value).bit_count() *
        (1 - error) ** (CLOCK_QUBITS - int(value).bit_count())
        for value in clock_bits
    ])
    return float(probabilities @ all_weights), float(probabilities @ clock_weights)


def compiled_templates() -> dict[tuple[int, int], tuple[QuantumCircuit, object]]:
    """Compile marked, parameterized templates once for all stress scenarios."""
    output = {}
    for physical in (0, 1):
        entry = grid.model(N, previous.BETA, bool(physical), previous.SPECTRUM)
        for tau_index, tau in enumerate(previous.TAUS):
            circuit, theta = template(entry, tau, noise_markers=True)
            output[(physical, tau_index)] = (
                transpile(circuit, basis_gates=["u3", "cx"], optimization_level=1),
                theta,
            )
    return output


def one_axis_probabilities(kind: str, level: float, workers: int,
                           templates: dict | None = None) -> dict:
    """Evaluate all labels and both shifts for one representative system axis.

    The state, channel placement, and measurement are invariant under a
    permutation of the three system axes. We prove that invariance with
    explicit axis checks below and then use it to avoid a factor-three
    repetition of equivalent density-matrix circuits.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    templates = compiled_templates() if templates is None else templates
    instruction, marker = channel(kind, level)
    affected = (range(TOTAL_QUBITS) if kind == "preparation_amplitude_damping"
                else range(SYSTEM_QUBITS))
    readout_error = level if kind == "symmetric_readout" else 0.0
    simulator = AerSimulator(method="density_matrix", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    raw = {}
    for physical in (0, 1):
        for tau_index in range(2):
            base, theta = templates[(physical, tau_index)]
            circuits = []
            settings = []
            for shift_index, shift in enumerate(previous.SHIFTS):
                for label in range(LABELS):
                    angles = np.full(N, shift + 2 * math.pi * label / LABELS)
                    angles[0] -= grid.BIAS
                    circuit = bind(base, theta, angles)
                    circuit = insert_channel(circuit, instruction, marker, affected)
                    circuit.save_probabilities()
                    circuits.append(circuit)
                    settings.append((shift_index, label))
            result = simulator.run(circuits, shots=1).result()
            if not result.success:
                raise RuntimeError(result.status)
            joint = np.zeros((2, LABELS))
            clock = np.zeros((2, LABELS))
            for index, setting in enumerate(settings):
                probabilities = np.asarray(
                    result.data(index)["probabilities"], dtype=float)
                joint[setting], clock[setting] = readout_events(
                    probabilities, readout_error)
            raw[(physical, tau_index)] = {
                "joint": joint,
                "clock": clock,
                "compiled_depth": base.depth(),
            }
    return raw


def axis_invariance_check(kind: str, level: float, workers: int,
                          templates: dict | None = None) -> float:
    """Check the only symmetry reduction used in the main sweep."""
    templates = compiled_templates() if templates is None else templates
    instruction, marker = channel(kind, level)
    affected = (range(TOTAL_QUBITS) if kind == "preparation_amplitude_damping"
                else range(SYSTEM_QUBITS))
    readout_error = level if kind == "symmetric_readout" else 0.0
    simulator = AerSimulator(method="density_matrix", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    physical, tau_index, shift_index, label = 0, 0, 1, 3
    base, theta = templates[(physical, tau_index)]
    circuits = []
    for axis in range(N):
        angles = np.full(N, previous.SHIFTS[shift_index] +
                         2 * math.pi * label / LABELS)
        angles[axis] -= grid.BIAS
        circuit = insert_channel(bind(base, theta, angles), instruction, marker,
                                  affected)
        circuit.save_probabilities()
        circuits.append(circuit)
    result = simulator.run(circuits, shots=1).result()
    events = []
    for index in range(N):
        events.append(readout_events(
            np.asarray(result.data(index)["probabilities"], dtype=float),
            readout_error,
        )[0])
    return float(np.max(np.abs(np.asarray(events) - events[0])))


def expand_axes(one_axis: dict) -> tuple[np.ndarray, np.ndarray]:
    """Restore the stage-6 [physical, tau, shift, axis, label] array shape."""
    joint = np.zeros((2, 2, 2, N, LABELS))
    clock = np.zeros_like(joint)
    for physical in (0, 1):
        for tau_index in range(2):
            joint[physical, tau_index] = one_axis[(physical, tau_index)][
                "joint"][:, None, :]
            clock[physical, tau_index] = one_axis[(physical, tau_index)][
                "clock"][:, None, :]
    return joint, clock


def full_axis_readout_probabilities(level: float, workers: int,
                                    templates: dict | None = None
                                    ) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate all three axes for the classical readout-error model.

    This is intentionally not reduced by the recovery-event symmetry check:
    the full measured bit-string distribution can differ by axis even where
    the ideal all-zero recovery probability agrees.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    templates = compiled_templates() if templates is None else templates
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
                        angles[axis] -= grid.BIAS
                        circuit = bind(base, theta, angles)
                        circuit.save_probabilities()
                        circuits.append(circuit)
                        settings.append((shift_index, axis, label))
            result = simulator.run(circuits, shots=1).result()
            if not result.success:
                raise RuntimeError(result.status)
            for index, setting in enumerate(settings):
                probabilities = np.asarray(
                    result.data(index)["probabilities"], dtype=float)
                joint[physical, tau_index][setting], clock[physical, tau_index][
                    setting] = readout_events(probabilities, level)
    return joint, clock


def condition_rows(joint: np.ndarray, clock: np.ndarray) -> list[dict]:
    """Calculate predeclared pooled contrasts and stage-6 budget proxies."""
    rows = []
    for distribution, physical, tau_index, role in CONDITIONS:
        physical_index = int(physical)
        shots = previous.shot_allocation(stage6.SHOTS_PER_LABEL, distribution)
        weights = shots / shots.sum()
        first = joint[physical_index, tau_index, 0]
        second = joint[physical_index, tau_index, 1]
        pooled_first = first @ weights
        pooled_second = second @ weights
        delta = pooled_second - pooled_first
        average = (pooled_first + pooled_second) / 2
        noncentrality = float(np.sum(
            delta ** 2 * shots.sum() / (2 * average * (1 - average))))
        clock_first = clock[physical_index, tau_index, 0] @ weights
        clock_second = clock[physical_index, tau_index, 1] @ weights
        rows.append({
            "role": role,
            "label_distribution": distribution,
            "physical_P_zero": bool(physical),
            "tau_over_pi": previous.TAUS[tau_index] / math.pi,
            "axis_pooled_joint_difference": delta.tolist(),
            "maximum_absolute_joint_difference": float(np.max(np.abs(delta))),
            "pooled_joint_probability_first": pooled_first.tolist(),
            "pooled_joint_probability_second": pooled_second.tolist(),
            "axis_pooled_clock_difference": (clock_second - clock_first).tolist(),
            "maximum_absolute_clock_difference": float(
                np.max(np.abs(clock_second - clock_first))),
            "stage6_budget_approx_noncentrality": noncentrality,
        })
    return rows


def scenario_summary(rows: list[dict], ideal_rows: list[dict]) -> dict:
    """State the predeclared false-leakage-to-signal separation diagnostic."""
    by_role = {row["role"]: row for row in rows}
    ideal_by_role = {row["role"]: row for row in ideal_rows}
    strong = by_role["strong_signal"]["maximum_absolute_joint_difference"]
    weak = by_role["weak_signal"]["maximum_absolute_joint_difference"]
    nulls = [by_role["distribution_null_tau_0"][
                 "maximum_absolute_joint_difference"],
             by_role["distribution_null_tau_half_pi"][
                 "maximum_absolute_joint_difference"],
             by_role["constraint_null_tau_0"][
                 "maximum_absolute_joint_difference"],
             by_role["constraint_null_tau_half_pi"][
                 "maximum_absolute_joint_difference"]]
    max_null = max(nulls)
    ratio = max_null / strong if strong else math.inf
    if max_null <= 1e-10:
        classification = "all_four_nulls_numerically_preserved"
    elif ratio <= SEPARATION_RATIO:
        classification = "null_leakage_below_predeclared_tenth_of_strong_signal"
    elif ratio <= 1:
        classification = "null_leakage_is_not_negligible_against_strong_signal"
    else:
        classification = "null_leakage_matches_or_exceeds_strong_signal"
    return {
        "strong_signal_scale": strong,
        "weak_signal_scale": weak,
        "maximum_null_scale": max_null,
        "maximum_null_to_strong_ratio": ratio,
        "strong_signal_fraction_of_ideal": (
            strong / ideal_by_role["strong_signal"][
                "maximum_absolute_joint_difference"]),
        "weak_signal_fraction_of_ideal": (
            weak / ideal_by_role["weak_signal"][
                "maximum_absolute_joint_difference"]),
        "classification": classification,
        "predeclared_negligible_leakage_ratio": SEPARATION_RATIO,
    }


def study(workers: int, levels: tuple[float, ...] = LEVELS) -> dict:
    """Run the entire frozen one-error-at-a-time exact Aer sweep."""
    templates = compiled_templates()
    scenarios = []
    ideal_rows = None
    for kind, level in scenario_specs(levels):
        if kind == "ideal":
            joint = stage6.exact_joint_probabilities()
            clock = np.full_like(joint, CLOCK_ACCEPTANCE)
            raw = None
            axis_reduction = False
            axis_error = 0.0
        elif kind == "symmetric_readout":
            joint, clock = full_axis_readout_probabilities(level, workers,
                                                           templates)
            raw = None
            axis_reduction = False
            axis_error = axis_invariance_check(kind, level, workers, templates)
        else:
            raw = one_axis_probabilities(kind, level, workers, templates)
            joint, clock = expand_axes(raw)
            axis_reduction = True
            axis_error = axis_invariance_check(kind, level, workers, templates)
            if axis_error > 1e-10:
                raise AssertionError("channel violated required axis symmetry")
        rows = condition_rows(joint, clock)
        if kind == "ideal":
            ideal_rows = rows
        scenario = {
            "error_kind": kind,
            "error_probability": level,
            "axis_symmetry_reduction_applied": axis_reduction,
            "axis_invariance_maximum_absolute_difference": axis_error,
            "compiled_depths": ({
                f"physical_{physical}_tau_{tau}":
                    templates[(physical, tau)][0].depth()
                for physical in (0, 1) for tau in range(2)
            }),
            "rows": rows,
        }
        if raw is not None:
            scenario["one_axis_joint_probabilities"] = {
                f"physical_{physical}_tau_{tau}":
                    raw[(physical, tau)]["joint"].tolist()
                for physical in (0, 1) for tau in range(2)
            }
            scenario["one_axis_clock_probabilities"] = {
                f"physical_{physical}_tau_{tau}":
                    raw[(physical, tau)]["clock"].tolist()
                for physical in (0, 1) for tau in range(2)
            }
        scenarios.append(scenario)
    if ideal_rows is None:
        raise AssertionError("ideal scenario missing")
    for scenario in scenarios:
        scenario["summary"] = scenario_summary(scenario["rows"], ideal_rows)
    return {
        "protocol": PROTOCOL,
        "parameters": {
            "n": N,
            "phase_labels": LABELS,
            "stage6_shots_per_label_if_uniform": stage6.SHOTS_PER_LABEL,
            "workers": workers,
            "error_models": {
                "preparation_amplitude_damping": (
                    "One amplitude-damping channel per qubit immediately after "
                    "the correlated state preparation."
                ),
                "phase_dephasing": (
                    "One phase-damping channel per system qubit immediately "
                    "after the scientific RZ phase layer."
                ),
                "symmetric_readout": (
                    "Independent classical 0<->1 readout flip on every qubit "
                    "after the ideal measurement."
                ),
            },
            "nonzero_error_probabilities": list(levels),
            "axis_symmetry_reduction": (
                "The amplitude-damping and phase-damping channels are checked "
                "for axis symmetry before expanding one explicitly simulated "
                "axis. Symmetric readout error is evaluated on every axis "
                "because the full measured-output distribution need not share "
                "the ideal recovery-event symmetry."
            ),
        },
        "scenarios": scenarios,
        "caveat": (
            "Localized one-error-at-a-time Aer model study. It is not a "
            "calibration to a physical processor, a claim that these channels "
            "are exhaustive, or a finite-shot hypothesis test."
        ),
    }


def circuit_checks(workers: int) -> dict:
    """Low-cost checks for channel placement and the ideal stage-6 limit."""
    templates = compiled_templates()
    ideal = one_axis_probabilities("ideal", 0.0, workers, templates)
    joint, clock = expand_axes(ideal)
    expected = stage6.exact_joint_probabilities()
    error = float(np.max(np.abs(joint - expected)))
    clock_error = float(np.max(np.abs(clock - CLOCK_ACCEPTANCE)))
    amplitude = axis_invariance_check("preparation_amplitude_damping", 0.01,
                                      workers, templates)
    phase = axis_invariance_check("phase_dephasing", 0.01, workers, templates)
    readout = axis_invariance_check("symmetric_readout", 0.01, workers, templates)
    if max(error, clock_error, amplitude, phase) > 1e-10:
        raise AssertionError("noise-stress circuit check failed")
    return {
        "ideal_joint_probability_maximum_absolute_error": error,
        "ideal_clock_probability_maximum_absolute_error": clock_error,
        "axis_invariance_maximum_absolute_errors": {
            "preparation_amplitude_damping": amplitude,
            "phase_dephasing": phase,
        },
        "symmetric_readout_axis_asymmetry_requires_full_axis_evaluation": readout,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    checks = circuit_checks(args.workers)
    if args.checks_only:
        print(json.dumps(checks, indent=2))
        return
    output = study(args.workers)
    output["circuit_checks"] = checks
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps([
        {
            "error_kind": scenario["error_kind"],
            "error_probability": scenario["error_probability"],
            **scenario["summary"],
        }
        for scenario in output["scenarios"]
    ], indent=2))


if __name__ == "__main__":
    main()
