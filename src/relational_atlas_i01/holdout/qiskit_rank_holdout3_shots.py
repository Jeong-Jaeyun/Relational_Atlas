"""Aer shot validation of uniform versus skewed phase-label erasure.

Frozen settings: notes/prevalidation_protocol3_2026-10-04.tex.
"""

from __future__ import annotations

import argparse
import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import DiagonalGate, StatePreparation
from qiskit_aer import AerSimulator

from relational_atlas_i01.phase_reference import qiskit_condition_probe as encoding
from relational_atlas_i01.phase_reference import qiskit_phase_label_intervention as testing
from relational_atlas_i01.holdout import qiskit_rank_holdout2_grid as grid


PROJECT = PROJECT_ROOT
RESULT = PROJECT / "results" / "qiskit_rank_holdout3_shots.json"
N = 3
LABELS = 7
BETA = 0.45
REFERENCE_BETA = 0.15
SPECTRUM = "nonlinear"
TAUS = (0.0, math.pi / 2)
SHIFTS = (0.0, 0.60)
DISTRIBUTIONS = ("uniform", "skewed")


def shot_allocation(shots_per_label: int, distribution: str) -> np.ndarray:
    total = LABELS * shots_per_label
    target = grid.label_weights(LABELS, distribution) * total
    allocation = np.floor(target).astype(int)
    missing = total - int(allocation.sum())
    order = np.argsort(-(target - allocation), kind="stable")
    allocation[order[:missing]] += 1
    if allocation.sum() != total or np.any(allocation < 1):
        raise AssertionError("invalid shot allocation")
    return allocation


def template(entry: dict, tau: float) -> tuple[QuantumCircuit, ParameterVector]:
    n = entry["n"]
    size = 1 << (2 * n)
    system = np.zeros(size, dtype=complex)
    reference = np.zeros(size, dtype=complex)
    diagonal = np.ones(size, dtype=complex)
    reference_weights = np.exp(-REFERENCE_BETA * entry["base_energies"])
    reference_weights /= reference_weights.sum()
    for k, weight, ref_weight, energy in zip(
        entry["momenta"], entry["weights"], reference_weights,
        entry["energies"]
    ):
        index = encoding.basis_index(k)
        system[index] = math.sqrt(weight)
        reference[index] = math.sqrt(ref_weight)
        diagonal[index] = np.exp(1j * tau * energy)
    angles = ParameterVector("theta", n)
    qc = QuantumCircuit(2 * n)
    qc.append(StatePreparation(system), range(2 * n))
    qc.append(DiagonalGate(diagonal), range(2 * n))
    for axis in range(n):
        qc.rz(angles[axis], 2 * axis)
        qc.rz(-angles[axis], 2 * axis + 1)
    qc.append(StatePreparation(reference).inverse(), range(2 * n))
    qc.measure_all()
    return qc, angles


def stratified_erased_pvalue(first: np.ndarray, second: np.ndarray,
                             shots_by_label: np.ndarray, permutations: int,
                             rng: np.random.Generator) -> dict:
    """Pool labels for the statistic, but condition on each label's null counts."""
    if first.shape != (N, LABELS) or second.shape != (N, LABELS):
        raise ValueError("unexpected count shape")
    total = first + second
    trials = int(shots_by_label.sum())
    observed = testing.deviance(first.sum(axis=1), second.sum(axis=1), trials)
    null_values = np.empty(permutations)
    for repetition in range(permutations):
        resampled = rng.hypergeometric(total, 2 * shots_by_label[None, :] - total,
                                      shots_by_label[None, :])
        null_values[repetition] = testing.deviance(
            resampled.sum(axis=1),
            (total - resampled).sum(axis=1), trials)
    return {"g_statistic": float(observed),
            "permutation_pvalue": float((1 + np.sum(null_values >= observed)) /
                                        (permutations + 1)),
            "null_95pct_statistic": float(np.percentile(null_values, 95)),
            "permutations": permutations,
            "null_conditioning": "axis x label, then pool labels for statistic"}


def exact_probabilities() -> np.ndarray:
    result = np.zeros((2, 2, 2, N, LABELS))
    for physical in (0, 1):
        entry = grid.model(N, BETA, bool(physical), SPECTRUM)
        for tau_index, tau in enumerate(TAUS):
            for shift_index, shift in enumerate(SHIFTS):
                angles = grid.angles_for_grid(N, LABELS, common=shift)
                result[physical, tau_index, shift_index], _ = (
                    grid.probability_gradient(entry, "mismatched", tau, angles))
    return result


def run(shots_per_label: int, workers: int, seed: int,
        permutations: int) -> dict:
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    compiled = {}
    parameters = {}
    for physical in (0, 1):
        entry = grid.model(N, BETA, bool(physical), SPECTRUM)
        for tau_index, tau in enumerate(TAUS):
            qc, angles = template(entry, tau)
            compiled[(physical, tau_index)] = transpile(
                qc, basis_gates=["u3", "cx"], optimization_level=1)
            parameters[(physical, tau_index)] = angles
    counts = np.zeros((2, 2, 2, 2, N, LABELS), dtype=int)
    allocations = np.stack([shot_allocation(shots_per_label, distribution)
                            for distribution in DISTRIBUTIONS])
    for dist_index in range(len(DISTRIBUTIONS)):
        for label in range(LABELS):
            circuits = []
            settings = []
            for physical in (0, 1):
                for tau_index in range(len(TAUS)):
                    for shift_index, shift in enumerate(SHIFTS):
                        for axis in range(N):
                            angles = np.full(N, shift + 2 * math.pi * label / LABELS)
                            angles[axis] -= grid.BIAS
                            binding = {parameters[(physical, tau_index)][j]:
                                       float(angles[j]) for j in range(N)}
                            circuits.append(compiled[(physical, tau_index)].assign_parameters(
                                binding))
                            settings.append((physical, tau_index, shift_index, axis))
            job = simulator.run(circuits,
                                shots=int(allocations[dist_index, label]),
                                seed_simulator=seed + 100 * dist_index + label)
            result = job.result()
            if not result.success:
                raise RuntimeError(result.status)
            for index, (physical, tau_index, shift_index, axis) in enumerate(settings):
                counts[dist_index, physical, tau_index, shift_index, axis, label] = (
                    result.get_counts(index).get("0" * (2 * N), 0))
    expected = exact_probabilities()
    tests = []
    rng = np.random.default_rng(seed + 1000)
    for dist_index, distribution in enumerate(DISTRIBUTIONS):
        weights = allocations[dist_index] / allocations[dist_index].sum()
        for physical in (0, 1):
            for tau_index, tau in enumerate(TAUS):
                first = counts[dist_index, physical, tau_index, 0]
                second = counts[dist_index, physical, tau_index, 1]
                pooled = stratified_erased_pvalue(
                    first, second, allocations[dist_index], permutations, rng)
                labeled = testing.permutation_pvalue(
                    first, second, allocations[dist_index][None, :],
                    permutations, rng)
                exact_first = expected[physical, tau_index, 0] @ weights
                exact_second = expected[physical, tau_index, 1] @ weights
                trials = allocations[dist_index].sum()
                predicted_noncentrality = float(np.sum(
                    (exact_second - exact_first) ** 2 * trials /
                    (2 * exact_first * (1 - exact_first))))
                tests.append({
                    "label_distribution": distribution,
                    "physical_P_zero": bool(physical),
                    "tau_over_pi": tau / math.pi,
                    "exact_max_erased_probability_difference": float(np.max(
                        np.abs(exact_second - exact_first))),
                    "approx_noncentrality": predicted_noncentrality,
                    "observed_max_erased_rate_difference": float(np.max(np.abs(
                        (second - first).sum(axis=1) / trials))),
                    "erased_stratified_permutation": pooled,
                    "labeled_permutation_exploratory": labeled,
                })
    threshold = 0.05 / len(tests)
    return {
        "protocol": "notes/prevalidation_protocol3_2026-10-04.tex",
        "parameters": {"n": N, "beta": BETA, "reference_beta": REFERENCE_BETA,
                       "spectrum": SPECTRUM, "phase_labels": LABELS,
                       "bias": grid.BIAS, "tau_over_pi": [t / math.pi for t in TAUS],
                       "common_shifts": SHIFTS,
                       "shots_per_label_if_uniform": shots_per_label,
                       "workers": workers, "seed": seed,
                       "permutations": permutations,
                       "circuits_executed": 2 * LABELS * 2 * 2 * 2 * N},
        "shot_allocations": {name: allocations[index].tolist()
                             for index, name in enumerate(DISTRIBUTIONS)},
        "raw_axes": ["label_distribution", "physical", "tau", "shift",
                     "axis", "phase_label"],
        "raw_zero_counts": counts.tolist(),
        "exact_probabilities_axes": ["physical", "tau", "shift", "axis",
                                     "phase_label"],
        "exact_zero_probabilities": expected.tolist(),
        "familywise_erased_tests": len(tests),
        "familywise_bonferroni_threshold": threshold,
        "tests": tests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots-per-label", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()
    if min(args.shots_per_label, args.workers, args.permutations) < 1:
        parser.error("shot, worker, and permutation counts must be positive")
    output = run(args.shots_per_label, args.workers, args.seed,
                 args.permutations)
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({
        "shots": output["shot_allocations"],
        "bonferroni_threshold": output["familywise_bonferroni_threshold"],
        "pvalues": [(row["label_distribution"], row["physical_P_zero"],
                     row["tau_over_pi"], row["erased_stratified_permutation"][
                         "permutation_pvalue"])
                    for row in output["tests"]],
        "result": str(RESULT),
    }, indent=2))


if __name__ == "__main__":
    main()
