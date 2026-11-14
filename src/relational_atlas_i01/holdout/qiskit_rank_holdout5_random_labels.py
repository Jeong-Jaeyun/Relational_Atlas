"""Aer validation of iid label-count jitter versus block distribution drift.

Frozen protocol: notes/prevalidation_protocol5_2026-10-04.tex.
"""

from __future__ import annotations

import argparse
import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from relational_atlas_i01.phase_reference import qiskit_phase_label_intervention as testing
from relational_atlas_i01.holdout import qiskit_rank_holdout2_grid as grid
from relational_atlas_i01.holdout import qiskit_rank_holdout3_shots as previous


PROJECT = PROJECT_ROOT
RESULT = PROJECT / "results" / "qiskit_rank_holdout5_random_labels.json"
BLOCKS = ("A_uniform", "B_uniform", "C_skewed")
TARGET_DISTRIBUTIONS = ("uniform", "uniform", "skewed")
CONTRASTS = ((0, 1), (0, 2))
N = 3
LABELS = 7
TOTAL_SHOTS = LABELS * 32768


def allocation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    assigned = np.zeros((3, 2, 2, N, LABELS), dtype=int)
    for block, distribution in enumerate(TARGET_DISTRIBUTIONS):
        target = grid.label_weights(LABELS, distribution)
        for physical in (0, 1):
            for tau_index in range(2):
                for axis in range(N):
                    assigned[block, physical, tau_index, axis] = rng.multinomial(
                        TOTAL_SHOTS, target)
    if np.any(assigned.sum(axis=-1) != TOTAL_SHOTS):
        raise AssertionError("multinomial totals changed")
    return assigned


def exact_probabilities() -> np.ndarray:
    result = np.zeros((2, 2, N, LABELS))
    for physical in (0, 1):
        entry = grid.model(N, 0.45, bool(physical), "nonlinear")
        for tau_index, tau in enumerate((0.0, math.pi / 2)):
            result[physical, tau_index], _ = grid.probability_gradient(
                entry, "mismatched", tau,
                grid.angles_for_grid(N, LABELS))
    return result


def uniform_standardization_pvalue(first: np.ndarray, second: np.ndarray,
                                    first_n: np.ndarray, second_n: np.ndarray,
                                    permutations: int,
                                    rng: np.random.Generator) -> dict:
    """Exact conditional test of the same-label null, with unequal n by label."""
    if first.shape != (N, LABELS) or second.shape != (N, LABELS):
        raise ValueError("unexpected count shape")
    pooled_probability = (first + second) / (first_n + second_n)
    variance = np.sum(pooled_probability * (1 - pooled_probability) * (
        1 / first_n + 1 / second_n), axis=1) / (LABELS * LABELS)
    if np.any(variance <= 0):
        raise ArithmeticError("zero standardized variance")
    observed_difference = np.mean(second / second_n - first / first_n, axis=1)
    observed = float(np.sum(observed_difference ** 2 / variance))
    combined = first + second
    exceeding = 0
    for _ in range(permutations):
        resampled_first = rng.hypergeometric(
            combined, first_n + second_n - combined, first_n)
        resampled_second = combined - resampled_first
        difference = np.mean(resampled_second / second_n -
                             resampled_first / first_n, axis=1)
        statistic = float(np.sum(difference ** 2 / variance))
        exceeding += statistic >= observed
    return {"statistic": observed,
            "permutation_pvalue": (1 + exceeding) / (permutations + 1),
            "axis_uniform_standardized_differences": observed_difference.tolist(),
            "axis_standard_errors": np.sqrt(variance).tolist(),
            "permutations": permutations,
            "null_conditioning": "axis x label successes and unequal label trials"}


def analyze(assigned: np.ndarray, counts: np.ndarray,
            exact: np.ndarray, permutations: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for first_block, second_block in CONTRASTS:
        for physical in (0, 1):
            for tau_index, tau in enumerate((0.0, math.pi / 2)):
                first = counts[first_block, physical, tau_index]
                second = counts[second_block, physical, tau_index]
                first_n = assigned[first_block, physical, tau_index]
                second_n = assigned[second_block, physical, tau_index]
                erased = testing.permutation_pvalue(
                    first.sum(axis=1), second.sum(axis=1),
                    TOTAL_SHOTS, permutations, rng)
                standardized = uniform_standardization_pvalue(
                    first, second, first_n, second_n,
                    permutations, rng)
                probability = exact[physical, tau_index]
                actual_first_weights = first_n / TOTAL_SHOTS
                actual_second_weights = second_n / TOTAL_SHOTS
                actual_expected_delta = np.sum(probability * (
                    actual_second_weights - actual_first_weights), axis=1)
                first_target = grid.label_weights(
                    LABELS, TARGET_DISTRIBUTIONS[first_block])
                second_target = grid.label_weights(
                    LABELS, TARGET_DISTRIBUTIONS[second_block])
                target_expected_delta = probability @ (
                    second_target - first_target)
                rows.append({
                    "contrast": f"{BLOCKS[first_block]}_vs_{BLOCKS[second_block]}",
                    "physical_P_zero": bool(physical),
                    "tau_over_pi": tau / math.pi,
                    "target_max_erased_probability_difference": float(np.max(
                        np.abs(target_expected_delta))),
                    "realized_allocation_max_expected_difference": float(np.max(
                        np.abs(actual_expected_delta))),
                    "erased_unconditional_label_test": erased,
                    "recorded_label_uniform_standardization": standardized,
                })
    threshold = 0.05 / len(rows)
    return {
        "tests_per_family": len(rows),
        "bonferroni_threshold_per_family": threshold,
        "erased_detections": [
            (row["contrast"], row["physical_P_zero"], row["tau_over_pi"])
            for row in rows if row["erased_unconditional_label_test"][
                "permutation_pvalue"] <= threshold],
        "recorded_label_detections": [
            (row["contrast"], row["physical_P_zero"], row["tau_over_pi"])
            for row in rows if row["recorded_label_uniform_standardization"][
                "permutation_pvalue"] <= threshold],
        "rows": rows,
    }


def run(workers: int, allocation_seed: int, aer_seed: int,
        permutations: int) -> dict:
    assigned = allocation(allocation_seed)
    exact = exact_probabilities()
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    compiled = {}
    parameters = {}
    for physical in (0, 1):
        entry = grid.model(N, 0.45, bool(physical), "nonlinear")
        for tau_index, tau in enumerate((0.0, math.pi / 2)):
            qc, angle_parameters = previous.template(entry, tau)
            compiled[(physical, tau_index)] = transpile(
                qc, basis_gates=["u3", "cx"], optimization_level=1)
            parameters[(physical, tau_index)] = angle_parameters
    counts = np.zeros_like(assigned)
    experiment = 0
    for block in range(3):
        for physical in (0, 1):
            for tau_index in range(2):
                for axis in range(N):
                    for label in range(LABELS):
                        angles = np.full(N, 2 * math.pi * label / LABELS)
                        angles[axis] -= grid.BIAS
                        binding = {
                            parameters[(physical, tau_index)][j]:
                                float(angles[j]) for j in range(N)}
                        bound = compiled[(physical, tau_index)].assign_parameters(
                            binding)
                        shots = int(assigned[block, physical, tau_index, axis, label])
                        result = simulator.run(
                            bound, shots=shots,
                            seed_simulator=aer_seed + experiment).result()
                        if not result.success:
                            raise RuntimeError(result.status)
                        counts[block, physical, tau_index, axis, label] = (
                            result.get_counts(0).get("0" * (2 * N), 0))
                        experiment += 1
    analysis = analyze(assigned, counts, exact, permutations,
                       seed=aer_seed + 1000)
    return {
        "protocol": "notes/prevalidation_protocol5_2026-10-04.tex",
        "parameters": {"n": N, "phase_labels": LABELS,
                       "shots_per_axis_block": TOTAL_SHOTS,
                       "workers": workers,
                       "allocation_seed": allocation_seed,
                       "aer_seed_base": aer_seed,
                       "permutations": permutations,
                       "circuits_executed": experiment,
                       "science_common_phase_all_blocks": 0.0},
        "block_targets": dict(zip(BLOCKS, TARGET_DISTRIBUTIONS)),
        "raw_axes": ["block", "physical", "tau", "axis", "phase_label"],
        "raw_label_counts": assigned.tolist(),
        "raw_zero_counts": counts.tolist(),
        "exact_probability_axes": ["physical", "tau", "axis", "phase_label"],
        "exact_zero_probabilities": exact.tolist(),
        "analysis": analysis,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--allocation-seed", type=int, default=20261023)
    parser.add_argument("--aer-seed", type=int, default=20261024)
    parser.add_argument("--permutations", type=int, default=10000)
    args = parser.parse_args()
    if min(args.workers, args.permutations) < 1:
        parser.error("workers and permutations must be positive")
    output = run(args.workers, args.allocation_seed, args.aer_seed,
                 args.permutations)
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({
        "circuits_executed": output["parameters"]["circuits_executed"],
        "threshold": output["analysis"]["bonferroni_threshold_per_family"],
        "erased_pvalues": [(row["contrast"], row["physical_P_zero"],
                            row["tau_over_pi"], row[
                                "erased_unconditional_label_test"][
                                    "permutation_pvalue"])
                           for row in output["analysis"]["rows"]],
        "standardized_pvalues": [(row["contrast"], row["physical_P_zero"],
                                  row["tau_over_pi"], row[
                                      "recorded_label_uniform_standardization"][
                                          "permutation_pvalue"])
                                 for row in output["analysis"]["rows"]],
        "erased_detections": output["analysis"]["erased_detections"],
        "recorded_label_detections": output["analysis"][
            "recorded_label_detections"],
        "result": str(RESULT),
    }, indent=2))


if __name__ == "__main__":
    main()
