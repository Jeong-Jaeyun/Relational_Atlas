"""I01 intervention: keep or erase the phase labels on the *same* Aer shots.

For each reference detector i, phase sample t, and common displacement alpha,
we run the same return-measurement circuit. Two analyses then consume exactly
the same success counts: (1) pool over t (label erased), (2) stratify by t
(label kept). A P=0 state is the physical-constraint control.
"""

from __future__ import annotations

import argparse
import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from relational_atlas_i01.phase_reference import qiskit_condition_probe as probe


PROJECT = PROJECT_ROOT
RESULT = PROJECT / "results" / "qiskit_phase_label_intervention.json"
N = 4
L = 2 * N + 1
BIAS = 0.8
COMMON_DISPLACEMENT = 0.3


def angle(i: int, t: int, alpha: float) -> np.ndarray:
    phases = np.full(N, alpha + 2 * np.pi * t / L)
    phases[i] -= BIAS
    return phases


def fisher(entry: dict, use_labels: bool) -> np.ndarray:
    information = np.zeros((N, N))
    for i in range(N):
        probabilities = []
        gradients = []
        for t in range(L):
            phases = angle(i, t, 0.0)
            probabilities.append(probe.return_probability(entry, phases))
            gradients.append(probe.return_gradient(entry, phases))
        if use_labels:
            for p, g in zip(probabilities, gradients):
                information += np.outer(g, g) / (p * (1 - p)) / (N * L)
        else:
            p = float(np.mean(probabilities))
            g = np.mean(gradients, axis=0)
            information += np.outer(g, g) / (p * (1 - p)) / N
    return information


def eigen_summary(matrix: np.ndarray) -> dict:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return {
        "eigenvalues": eigenvalues.tolist(),
        "rank_tol_1e_minus_8": int(np.sum(eigenvalues > 1e-8)),
        "smallest_mode_common_alignment_abs": float(abs(
            eigenvectors[:, 0] @ (np.ones(N) / np.sqrt(N)))),
        "common_direction_information": float(
            np.ones(N) @ matrix @ np.ones(N) / N),
    }


def exact_study() -> dict:
    rows = {}
    for name, physical in (("unrestricted", False), ("P_zero", True)):
        entry = probe.model(N, physical)
        kept = fisher(entry, True)
        erased = fisher(entry, False)
        difference = np.linalg.eigvalsh(kept - erased)
        marginal_error = max(abs(np.mean([
            probe.return_probability(entry, angle(i, t, COMMON_DISPLACEMENT))
            for t in range(L)]) - np.mean([
            probe.return_probability(entry, angle(i, t, 0.0))
            for t in range(L)])) for i in range(N))
        assert np.min(difference) > -1e-10
        assert marginal_error < 1e-12
        rows[name] = {
            "label_kept_joint_cfi": eigen_summary(kept),
            "label_erased_marginal_cfi": eigen_summary(erased),
            "data_processing_difference_min_eigenvalue": float(np.min(difference)),
            "erased_probability_max_change_under_common_displacement": marginal_error,
            "minimum_labeled_return_probability": float(min(
                probe.return_probability(entry, angle(i, t, 0.0))
                for i in range(N) for t in range(L))),
        }
    return rows


def deviance(success_zero: np.ndarray, success_shift: np.ndarray, trials: int) -> float:
    """Two-sample Bernoulli G statistic summed over observable channels."""
    a = np.asarray(success_zero, dtype=float)
    b = np.asarray(success_shift, dtype=float)
    pooled_success = (a + b) / 2
    pooled_failure = trials - pooled_success

    def term(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
        contribution = np.zeros_like(observed)
        nonzero = observed > 0
        contribution[nonzero] = observed[nonzero] * np.log(
            observed[nonzero] / expected[nonzero])
        return contribution

    return float(2 * np.sum(
        term(a, pooled_success) + term(trials - a, pooled_failure)
        + term(b, pooled_success) + term(trials - b, pooled_failure)))


def permutation_pvalue(a: np.ndarray, b: np.ndarray, trials: int,
                       repetitions: int, rng: np.random.Generator) -> dict:
    observed = deviance(a, b, trials)
    total = a + b
    null_values = np.empty(repetitions)
    for repetition in range(repetitions):
        resampled_a = rng.hypergeometric(total, 2 * trials - total, trials)
        null_values[repetition] = deviance(resampled_a, total - resampled_a, trials)
    return {
        "g_statistic": observed,
        "permutation_pvalue": float((1 + np.sum(null_values >= observed)) /
                                    (repetitions + 1)),
        "null_95pct_statistic": float(np.percentile(null_values, 95)),
        "permutations": repetitions,
    }


def shot_study(shots: int, repetitions: int, seed: int, workers: int) -> dict:
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    rng = np.random.default_rng(seed + 1000)
    rows = {}
    for case_index, (name, physical) in enumerate((("unrestricted", False),
                                                     ("P_zero", True))):
        entry = probe.model(N, physical)
        template, parameters = probe.template(entry)
        # Aer 0.17.2 on this Windows installation crashes on multiplexer.
        compiled = transpile(template, basis_gates=["u3", "cx"], optimization_level=1)
        schedule = [(alpha_index, i, t, angle(i, t, alpha))
                    for alpha_index, alpha in enumerate((0.0, COMMON_DISPLACEMENT))
                    for i in range(N) for t in range(L)]
        circuits = [compiled.assign_parameters({parameters[j]: float(phases[j])
                                                for j in range(N)})
                    for _, _, _, phases in schedule]
        result = simulator.run(circuits, shots=shots,
                               seed_simulator=seed + case_index).result()
        if not result.success:
            raise RuntimeError(result.status)
        successes = np.zeros((2, N, L), dtype=int)
        probability_errors = []
        for index, (alpha_index, i, t, phases) in enumerate(schedule):
            observed = result.get_counts(index).get("0" * (2 * N), 0)
            successes[alpha_index, i, t] = observed
            probability_errors.append(abs(observed / shots -
                                          probe.return_probability(entry, phases)))
        a, b = successes
        kept = permutation_pvalue(a, b, shots, repetitions, rng)
        # The stratified comparison keeps detector i and phase t. The erased
        # comparison keeps detector i but pools all nine phase labels t.
        erased = permutation_pvalue(a.sum(axis=1), b.sum(axis=1),
                                     L * shots, repetitions, rng)
        rows[name] = {
            "same_raw_success_counts_for_both_analyses": successes.tolist(),
            "count_array_axes": ["common_displacement_index", "reference_axis_i", "phase_label_t"],
            "labeled_test": kept,
            "erased_test": erased,
            "shots_per_phase_circuit": shots,
            "circuits": len(circuits),
            "mean_absolute_probability_error": float(np.mean(probability_errors)),
            "transpiled_depth": compiled.depth(),
        }
    return rows


def loss_scaling() -> dict:
    """Contrast unflagged quadratic loss with flagged linear information."""
    entry = probe.model(N, False)
    referenced = probe.analytic_fisher(entry, BIAS, False)
    erased = probe.analytic_fisher(entry, BIAS, True)
    epsilons = (0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
    rows = []
    for epsilon in epsilons:
        eta = 1 - epsilon
        unflagged = probe.phase_reference_fisher(entry, BIAS, eta)
        flagged = epsilon * referenced + eta * erased
        rows.append({
            "retained_reference_fraction_epsilon": epsilon,
            "unflagged_common_eigenvalue": float(np.linalg.eigvalsh(unflagged)[0]),
            "flagged_common_eigenvalue": float(np.linalg.eigvalsh(flagged)[0]),
        })
    log_epsilon = np.log([r["retained_reference_fraction_epsilon"] for r in rows[-4:]])
    quadratic_slope = float(np.polyfit(log_epsilon,
                                      np.log([r["unflagged_common_eigenvalue"]
                                              for r in rows[-4:]]), 1)[0])
    linear_slope = float(np.polyfit(log_epsilon,
                                   np.log([r["flagged_common_eigenvalue"]
                                           for r in rows[-4:]]), 1)[0])
    return {
        "resource_models": {
            "unflagged": "Detector type is not recorded: p_eta=(1-eta)*p_reference+eta*p_twirl",
            "flagged": "Availability flag is recorded: F=(1-eta)*F_reference+eta*F_twirl",
        },
        "rows": rows,
        "last_four_loglog_slope_unflagged": quadratic_slope,
        "last_four_loglog_slope_flagged": linear_slope,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    if min(args.shots, args.permutations, args.workers) < 1:
        parser.error("shots, permutations and workers must be positive")
    output = {
        "design": "Balanced nine-phase intervention on the same return-count dataset, with label-used versus label-erased analysis",
        "assumption": "An external, calibrated classical record identifies each applied phase t; no phase estimation from the quantum system itself is assumed.",
        "parameters": {"n": N, "bias": BIAS, "common_displacement": COMMON_DISPLACEMENT,
                       "phase_samples": [float(2 * math.pi * t / L) for t in range(L)],
                       "shots_per_phase_circuit": args.shots, "seed": args.seed},
        "exact": exact_study(),
        "shot_intervention": shot_study(args.shots, args.permutations,
                                         args.seed, args.workers),
        "partial_loss_scaling": loss_scaling(),
    }
    RESULT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "exact_ranks": {case: {label: row[label]["rank_tol_1e_minus_8"]
                               for label in ("label_kept_joint_cfi", "label_erased_marginal_cfi")}
                        for case, row in output["exact"].items()},
        "permutation_pvalues": {case: {label: row[label]["permutation_pvalue"]
                                      for label in ("labeled_test", "erased_test")}
                                for case, row in output["shot_intervention"].items()},
        "loss_slopes": [output["partial_loss_scaling"]["last_four_loglog_slope_unflagged"],
                        output["partial_loss_scaling"]["last_four_loglog_slope_flagged"]],
        "result": str(RESULT),
    }, indent=2))


if __name__ == "__main__":
    main()
