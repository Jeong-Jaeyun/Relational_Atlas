"""Aer finite-shot intervention at an n=3 holdout, not the original n=4 fit."""

from __future__ import annotations

import argparse
import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np
from qiskit import transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator

from relational_atlas_i01.phase_reference import qiskit_phase_label_intervention as intervention
from relational_atlas_i01.holdout import qiskit_rank_holdout_circuit as circuit_check
from relational_atlas_i01.holdout import qiskit_rank_holdout_grid as holdout


PROJECT = PROJECT_ROOT
RESULT = PROJECT / "results" / "qiskit_rank_holdout_shots.json"
N = 3
BETA = 0.45
BIAS = 1.25
LABEL_COUNTS = (2 * N - 1, 2 * N + 1)
TAUS = (0.0, math.pi / 2)
SHIFTS = (0.0, 0.23)


def schedule():
    return [(tau_index, labels_index, shift_index, axis, label)
            for tau_index in range(len(TAUS))
            for labels_index, label_count in enumerate(LABEL_COUNTS)
            for shift_index in range(len(SHIFTS))
            for axis in range(N) for label in range(label_count)]


def run_model(physical: bool, shots: int, workers: int, seed: int) -> dict:
    entry = holdout.model(N, BETA, physical)
    tau_parameter = Parameter("tau")
    phase_parameters = ParameterVector("phase", N)
    template, system_qubits = circuit_check.circuit(
        entry, tau_parameter, phase_parameters)
    template.measure_all()
    compiled = transpile(template, basis_gates=["u3", "cx"],
                         optimization_level=1)
    settings = schedule()
    circuits = []
    for tau_index, labels_index, shift_index, axis, label in settings:
        label_count = LABEL_COUNTS[labels_index]
        angles = np.full(N, SHIFTS[shift_index] +
                         2 * math.pi * label / label_count)
        angles[axis] -= BIAS
        bindings = {tau_parameter: TAUS[tau_index]}
        bindings.update({phase_parameters[k]: float(angles[k]) for k in range(N)})
        circuits.append(compiled.assign_parameters(bindings))
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    result = simulator.run(circuits, shots=shots, seed_simulator=seed).result()
    if not result.success:
        raise RuntimeError(result.status)
    accepted = np.zeros((2, 2, 2, N, max(LABEL_COUNTS)), dtype=int)
    joint = np.zeros_like(accepted)
    errors = []
    for index, (tau_index, labels_index, shift_index, axis, label) in enumerate(settings):
        label_count = LABEL_COUNTS[labels_index]
        angles = np.full(N, SHIFTS[shift_index] +
                         2 * math.pi * label / label_count)
        angles[axis] -= BIAS
        counts = result.get_counts(index)
        clock_qubits = compiled.num_qubits - system_qubits
        accepted[tau_index, labels_index, shift_index, axis, label] = sum(
            count for bits, count in counts.items()
            if bits[:clock_qubits] == "0" * clock_qubits)
        joint[tau_index, labels_index, shift_index, axis, label] = counts.get(
            "0" * compiled.num_qubits, 0)
        p, _ = holdout.probability_gradient(entry, TAUS[tau_index], angles)
        errors.append(abs(joint[tau_index, labels_index, shift_index, axis, label]
                          / shots - p / (N + 1)))
    return {
        "physical_P_zero": physical, "circuits": len(circuits),
        "transpiled_depth": compiled.depth(),
        "clock_postselection_rate": float(accepted.sum() / (len(circuits) * shots)),
        "mean_absolute_joint_probability_error": float(np.mean(errors)),
        "clock_success_counts": accepted,
        "joint_zero_counts": joint,
    }


def study(shots: int, workers: int, seed: int, permutations: int) -> dict:
    rows = {}
    rng = np.random.default_rng(seed + 100)
    for physical in (False, True):
        raw = run_model(physical, shots, workers, seed + int(physical))
        comparisons = []
        for tau_index, tau in enumerate(TAUS):
            for labels_index, label_count in enumerate(LABEL_COUNTS):
                a = raw["joint_zero_counts"][tau_index, labels_index, 0, :, :label_count]
                b = raw["joint_zero_counts"][tau_index, labels_index, 1, :, :label_count]
                fisher = holdout.information(holdout.model(N, BETA, physical),
                                             tau, BIAS, label_count)
                comparisons.append({
                    "tau_over_pi": tau / math.pi,
                    "phase_labels": label_count,
                    "exact_labeled_rank": fisher["label_kept"]["rank_relative_1e_minus_8"],
                    "exact_erased_rank": fisher["label_erased"]["rank_relative_1e_minus_8"],
                    "labeled_block_difference": intervention.permutation_pvalue(
                        a, b, shots, permutations, rng),
                    "erased_block_difference": intervention.permutation_pvalue(
                        a.sum(axis=1), b.sum(axis=1), label_count * shots,
                        permutations, rng),
                })
        rows["P_zero" if physical else "unrestricted"] = {
            "metadata": {key: value for key, value in raw.items()
                         if not isinstance(value, np.ndarray)},
            "raw_axes": ["tau", "label_count", "shift", "axis", "label_padded"],
            "clock_success_counts": raw["clock_success_counts"].tolist(),
            "joint_zero_counts": raw["joint_zero_counts"].tolist(),
            "comparisons": comparisons,
        }
    family_tests = sum(2 * len(case["comparisons"]) for case in rows.values())
    family_threshold = 0.05 / family_tests
    adjusted = {name: {
        "labeled": sum(row["labeled_block_difference"]["permutation_pvalue"] <=
                       family_threshold for row in case["comparisons"]),
        "erased": sum(row["erased_block_difference"]["permutation_pvalue"] <=
                      family_threshold for row in case["comparisons"])}
        for name, case in rows.items()}
    return {"parameters": {"n": N, "beta": BETA, "bias": BIAS,
                           "label_counts": LABEL_COUNTS,
                           "tau_over_pi": [tau / math.pi for tau in TAUS],
                           "common_shifts": SHIFTS, "shots": shots,
                           "workers": workers, "seed": seed,
                           "permutations": permutations},
            "exploratory_familywise_audit": {
                "tests": family_tests,
                "bonferroni_threshold": family_threshold,
                "detections_by_case": adjusted,
                "caveat": "Multiplicity adjustment was added after observing raw p-values; it is a diagnostic, not a preregistered confirmatory test."},
            "cases": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()
    if min(args.shots, args.workers, args.permutations) < 1:
        parser.error("shots, workers, permutations must be positive")
    output = study(args.shots, args.workers, args.seed, args.permutations)
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({"pvalues": {
        name: [(row["tau_over_pi"], row["phase_labels"],
                row["labeled_block_difference"]["permutation_pvalue"],
                row["erased_block_difference"]["permutation_pvalue"])
               for row in case["comparisons"]]
        for name, case in output["cases"].items()},
        "result": str(RESULT)}, indent=2))


if __name__ == "__main__":
    main()
