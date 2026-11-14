"""I01 phase-reference audit: nuisance offsets, block drift, and calibration.

The phase-label intervention only identifies a common system phase when its
phase origin is externally anchored. Here two acquisition blocks have unknown
offsets. Science counts alone identify alpha+delta_b, not alpha; separate
known-phase reference runs estimate each delta_b. All runs use Qiskit Aer.
"""

from __future__ import annotations

import argparse
import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np
from numpy.polynomial.hermite import hermgauss
from qiskit import transpile
from qiskit_aer import AerSimulator

from relational_atlas_i01.phase_reference import qiskit_condition_probe as probe
from relational_atlas_i01.phase_reference import qiskit_phase_label_intervention as intervention


PROJECT = PROJECT_ROOT
RESULT = PROJECT / "results" / "qiskit_phase_anchor_audit.json"
ALPHA_TRUE = 0.30
OFFSETS = (-0.15, 0.35)
CALIBRATION_RATIOS = (0.0, 0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0)
FIT_GRID = np.linspace(-0.8, 1.0, 3601)


def labeled_fisher_at_offset(entry: dict, offset: float, jitter_sigma: float = 0.0) -> np.ndarray:
    """Per interrogation CFI with t known and Gaussian unrecorded phase jitter."""
    n, length = intervention.N, intervention.L
    fisher = np.zeros((n, n))
    if jitter_sigma:
        nodes, weights = hermgauss(31)
        jitters = np.sqrt(2) * jitter_sigma * nodes
        quadrature = weights / np.sqrt(np.pi)
    else:
        jitters, quadrature = (0.0,), (1.0,)
    for i in range(n):
        for t in range(length):
            p = 0.0
            gradient = np.zeros(n)
            for jitter, weight in zip(jitters, quadrature):
                phases = intervention.angle(i, t, offset + jitter)
                p += weight * probe.return_probability(entry, phases)
                gradient += weight * probe.return_gradient(entry, phases)
            fisher += np.outer(gradient, gradient) / (p * (1 - p)) / (n * length)
    return fisher


def profile_nuisance(physical: bool, calibration_ratio: float) -> dict:
    """Schur-complement CFI for theta after two unknown block offsets."""
    n, blocks = intervention.N, len(OFFSETS)
    science = probe.model(n, physical)
    anchor = probe.model(n, False)
    joint = np.zeros((n + blocks, n + blocks))
    ones = np.ones(n)
    for block, offset in enumerate(OFFSETS):
        science_fisher = labeled_fisher_at_offset(science, offset)
        anchor_fisher = labeled_fisher_at_offset(anchor, offset)
        anchor_information = float(ones @ anchor_fisher @ ones)
        weight = 1 / blocks
        joint[:n, :n] += weight * science_fisher
        cross = science_fisher @ ones
        joint[:n, n + block] += weight * cross
        joint[n + block, :n] += weight * cross
        joint[n + block, n + block] += weight * (
            float(ones @ science_fisher @ ones) + calibration_ratio * anchor_information)
    theta_theta = joint[:n, :n]
    theta_delta = joint[:n, n:]
    delta_delta = joint[n:, n:]
    effective = theta_theta - theta_delta @ np.linalg.pinv(delta_delta,
                                                              rcond=1e-12) @ theta_delta.T
    effective = (effective + effective.T) / 2
    eigenvalues = np.linalg.eigvalsh(effective)
    unanchored_null = np.concatenate((ones, -np.ones(blocks)))
    return {
        "calibration_to_science_shot_ratio_per_setting": calibration_ratio,
        "profiled_theta_eigenvalues": eigenvalues.tolist(),
        "profiled_rank_tol_1e_minus_8": int(np.sum(eigenvalues > 1e-8)),
        "rank_at_1pct_of_max": int(np.sum(eigenvalues > 0.01 * eigenvalues[-1])),
        "smallest_to_max_eigenvalue_ratio": float(eigenvalues[0] / eigenvalues[-1]),
        "common_direction_information": float(ones @ effective @ ones / n),
        "joint_null_residual_for_theta_plus_offset_trade": float(
            np.linalg.norm(joint @ unanchored_null)),
    }


def exact_study() -> dict:
    unanchored = [profile_nuisance(False, ratio) for ratio in CALIBRATION_RATIOS]
    physical = [profile_nuisance(True, ratio) for ratio in CALIBRATION_RATIOS]
    assert unanchored[0]["profiled_rank_tol_1e_minus_8"] == 3
    assert unanchored[-1]["profiled_rank_tol_1e_minus_8"] == 4
    assert all(row["profiled_rank_tol_1e_minus_8"] == 3 for row in physical)
    assert unanchored[0]["joint_null_residual_for_theta_plus_offset_trade"] < 1e-10

    low, high = 0.0, 1.0
    for _ in range(40):
        middle = (low + high) / 2
        if profile_nuisance(False, middle)["smallest_to_max_eigenvalue_ratio"] < 0.01:
            low = middle
        else:
            high = middle
    apparent_transition = (low + high) / 2

    full = probe.model(intervention.N, False)
    jitter = []
    for sigma in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):
        matrix = labeled_fisher_at_offset(full, 0.0, sigma)
        eigs = np.linalg.eigvalsh(matrix)
        unit_common = np.ones(intervention.N) / np.sqrt(intervention.N)
        common = float(unit_common @ matrix @ unit_common)
        relative = float((np.trace(matrix) - common) / (intervention.N - 1))
        jitter.append({
            "unrecorded_phase_jitter_sigma_radians": sigma,
            "labeled_cfi_eigenvalues": eigs.tolist(),
            "common_direction_information": common,
            "relative_direction_information": relative,
            "common_to_relative_ratio": common / relative,
        })
    return {"unrestricted_profiled_scan": unanchored,
            "P_zero_profiled_scan": physical,
            "calibration_ratio_for_apparent_rank4_at_1pct": apparent_transition,
            "unrecorded_within_block_jitter_scan": jitter}


def full_return_probabilities(parameter: np.ndarray) -> np.ndarray:
    """Factorized unrestricted cutoff-1 return p(t | common phase)."""
    value = np.asarray(parameter, dtype=float)[..., None]
    phases = value + 2 * np.pi * np.arange(intervention.L) / intervention.L
    a = np.exp(-probe.BETA)
    characteristic = lambda x: (1 + 2 * a * np.cos(x)) / (1 + 2 * a)
    p = characteristic(phases - intervention.BIAS) ** 2
    p *= characteristic(phases) ** (2 * (intervention.N - 1))
    return p


def fit_known_phase_reference(successes: np.ndarray, shots: int) -> dict:
    """Grid likelihood for one calibrated or science block, retaining t."""
    totals = np.asarray(successes, dtype=int).sum(axis=0)
    trials_per_phase = intervention.N * shots
    p_grid = np.clip(full_return_probabilities(FIT_GRID), 1e-14, 1 - 1e-14)
    log_likelihood = (totals * np.log(p_grid) +
                      (trials_per_phase - totals) * np.log1p(-p_grid)).sum(axis=1)
    maximum = int(np.argmax(log_likelihood))
    estimate = float(FIT_GRID[maximum])
    included = FIT_GRID[2 * (log_likelihood[maximum] - log_likelihood) <= 3.841459]
    h = 1e-5
    p = full_return_probabilities(np.array([estimate]))[0]
    derivative = (full_return_probabilities(np.array([estimate + h]))[0]
                  - full_return_probabilities(np.array([estimate - h]))[0]) / (2 * h)
    information = float(trials_per_phase * np.sum(derivative ** 2 / (p * (1 - p))))
    return {
        "estimated_common_phase_radians": estimate,
        "profile_likelihood_95pct_interval_radians": [float(included[0]),
                                                        float(included[-1])],
        "fisher_standard_error_radians": float(1 / np.sqrt(information)),
        "successes_by_phase_label": totals.tolist(),
        "trials_per_phase_label": trials_per_phase,
    }


def run_case(simulator: AerSimulator, entry: dict, schedule: list[tuple],
             shots: int, seed: int) -> tuple[np.ndarray, dict]:
    template, parameters = probe.template(entry)
    # Aer 0.17.2 on this Windows installation crashes on multiplexer.
    compiled = transpile(template, basis_gates=["u3", "cx"], optimization_level=1)
    circuits = []
    for _, _, i, t, common_phase in schedule:
        angles = intervention.angle(i, t, common_phase)
        circuits.append(compiled.assign_parameters({parameters[j]: float(angles[j])
                                                    for j in range(intervention.N)}))
    result = simulator.run(circuits, shots=shots, seed_simulator=seed).result()
    if not result.success:
        raise RuntimeError(result.status)
    counts = np.zeros((2, 2, intervention.N, intervention.L), dtype=int)
    errors = []
    for index, (block, run_type, i, t, common_phase) in enumerate(schedule):
        success = result.get_counts(index).get("0" * (2 * intervention.N), 0)
        counts[block, run_type, i, t] = success
        expected = probe.return_probability(entry, intervention.angle(i, t, common_phase))
        errors.append(abs(success / shots - expected))
    return counts, {"circuits": len(circuits),
                    "transpiled_depth": compiled.depth(),
                    "mean_absolute_probability_error": float(np.mean(errors))}


def shot_study(shots: int, workers: int, seed: int, permutations: int) -> dict:
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    schedule_full = [(block, run_type, i, t,
                      (ALPHA_TRUE if run_type == 0 else 0.0) + offset)
                     for block, offset in enumerate(OFFSETS)
                     for run_type in (0, 1)  # science, known-phase anchor
                     for i in range(intervention.N) for t in range(intervention.L)]
    full_counts, full_meta = run_case(simulator, probe.model(4, False),
                                      schedule_full, shots, seed)
    fits = []
    for block in range(len(OFFSETS)):
        science = fit_known_phase_reference(full_counts[block, 0], shots)
        anchor = fit_known_phase_reference(full_counts[block, 1], shots)
        alpha_estimate = (science["estimated_common_phase_radians"]
                          - anchor["estimated_common_phase_radians"])
        alpha_se = math.hypot(science["fisher_standard_error_radians"],
                              anchor["fisher_standard_error_radians"])
        fits.append({
            "block": block,
            "true_offset_radians": OFFSETS[block],
            "science_sum_alpha_plus_offset": science,
            "known_phase_anchor_offset": anchor,
            "naive_alpha_if_offset_set_to_zero": science["estimated_common_phase_radians"],
            "calibrated_alpha_estimate_radians": alpha_estimate,
            "calibrated_alpha_standard_error_radians": alpha_se,
        })
    precisions = np.array([1 / row["calibrated_alpha_standard_error_radians"] ** 2
                           for row in fits])
    estimates = np.array([row["calibrated_alpha_estimate_radians"] for row in fits])
    combined = float(np.dot(precisions, estimates) / precisions.sum())
    combined_se = float(1 / np.sqrt(precisions.sum()))

    schedule_physical = [(block, 0, i, t, ALPHA_TRUE + offset)
                         for block, offset in enumerate(OFFSETS)
                         for i in range(intervention.N) for t in range(intervention.L)]
    physical_counts, physical_meta = run_case(simulator, probe.model(4, True),
                                              schedule_physical, shots, seed + 1)
    p0 = physical_counts[0, 0]
    p1 = physical_counts[1, 0]
    rng = np.random.default_rng(seed + 100)
    physical_tests = {
        "labeled": intervention.permutation_pvalue(p0, p1, shots, permutations, rng),
        "erased": intervention.permutation_pvalue(p0.sum(axis=1), p1.sum(axis=1),
                                                    intervention.L * shots, permutations, rng),
    }
    return {
        "shots_per_setting": shots,
        "full_state_circuit_metadata": full_meta,
        "P_zero_circuit_metadata": physical_meta,
        "full_state_raw_counts_axes": ["block", "run_type_0_science_1_anchor",
                                       "reference_axis", "phase_label"],
        "full_state_raw_success_counts": full_counts.tolist(),
        "P_zero_science_raw_success_counts": physical_counts[:, 0].tolist(),
        "block_fits": fits,
        "combined_calibrated_alpha_estimate_radians": combined,
        "combined_alpha_standard_error_radians": combined_se,
        "combined_approximate_95pct_interval_radians": [combined - 1.96 * combined_se,
                                                           combined + 1.96 * combined_se],
        "P_zero_block_change_tests": physical_tests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--replicates", type=int, default=5)
    args = parser.parse_args()
    if min(args.shots, args.permutations, args.workers, args.replicates) < 1:
        parser.error("shots, permutations, workers, replicates must be positive")
    primary = shot_study(args.shots, args.workers, args.seed, args.permutations)
    replication_rows = []
    for replication in range(args.replicates):
        seed = args.seed + replication
        run = primary if replication == 0 else shot_study(
            args.shots, args.workers, seed, args.permutations)
        lo, hi = run["combined_approximate_95pct_interval_radians"]
        replication_rows.append({
            "seed": seed,
            "combined_calibrated_alpha": run["combined_calibrated_alpha_estimate_radians"],
            "approximate_95pct_interval": [lo, hi],
            "true_alpha_inside_interval": bool(lo <= ALPHA_TRUE <= hi),
            "naive_block_estimates": [row["naive_alpha_if_offset_set_to_zero"]
                                      for row in run["block_fits"]],
            "calibrated_block_estimates": [row["calibrated_alpha_estimate_radians"]
                                           for row in run["block_fits"]],
            "P_zero_labeled_pvalue": run["P_zero_block_change_tests"]["labeled"][
                "permutation_pvalue"],
        })
    output = {
        "question": "Does a recorded phase label still recover the common direction when its global zero drifts?",
        "resource_assumption": "An independent known-phase anchor run shares the detector phase frame within each block.",
        "parameters": {"alpha_true": ALPHA_TRUE, "block_offsets": OFFSETS,
                       "bias": intervention.BIAS, "phase_count": intervention.L,
                       "shots_per_setting": args.shots, "seed": args.seed,
                       "workers": args.workers, "replicates": args.replicates},
        "exact": exact_study(),
        "shots": primary,
        "replication_scan": replication_rows,
    }
    RESULT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "unanchored_rank": output["exact"]["unrestricted_profiled_scan"][0][
            "profiled_rank_tol_1e_minus_8"],
        "anchored_rank": output["exact"]["unrestricted_profiled_scan"][5][
            "profiled_rank_tol_1e_minus_8"],
        "P_zero_rank_with_anchor": output["exact"]["P_zero_profiled_scan"][5][
            "profiled_rank_tol_1e_minus_8"],
        "naive_block_alpha": [row["naive_alpha_if_offset_set_to_zero"]
                              for row in output["shots"]["block_fits"]],
        "calibrated_block_alpha": [row["calibrated_alpha_estimate_radians"]
                                   for row in output["shots"]["block_fits"]],
        "combined_alpha": output["shots"]["combined_calibrated_alpha_estimate_radians"],
        "combined_alpha_interval": output["shots"]["combined_approximate_95pct_interval_radians"],
        "replication_combined_alpha": [row["combined_calibrated_alpha"]
                                       for row in replication_rows],
        "replication_coverage_count": sum(row["true_alpha_inside_interval"]
                                          for row in replication_rows),
        "result": str(RESULT),
    }, indent=2))


if __name__ == "__main__":
    main()
