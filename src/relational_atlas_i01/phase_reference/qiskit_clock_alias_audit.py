"""I01 clock-conditioned audit of temporal aliasing in phase anchors.

A sinusoidal drift mode is zero on a five-point uniform anchor grid but nonzero
at an intervening science time. The analysis does not fit the sinusoid: it
compares local interpolation with a held-out finer grid. An exact Fisher audit
uses the hidden mode only as a counterexample to identifiability guarantees.
"""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np
from qiskit import transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator

from relational_atlas_i01.phase_reference import qiskit_clock_anchor_intervention as clock
from relational_atlas_i01.phase_reference import qiskit_phase_label_intervention as labels


PROJECT = PROJECT_ROOT
RESULT = PROJECT / "results" / "qiskit_clock_alias_audit.json"
ALPHA = 0.30
BETA_0 = -0.15
BETA_1 = 0.50
HIDDEN_AMPLITUDE = 0.18
SCIENCE_TIME = 3 / 8
ANCHOR_TIMES = np.arange(9, dtype=float) / 8
COARSE_INDICES = (0, 2, 4, 6, 8)
HOLDOUT_INDICES = (1, 3, 5, 7)
SHOT_TAUS = clock.SHOT_TAUS


@lru_cache(maxsize=2)
def model(physical: bool) -> dict:
    return clock.clock_model(physical)


def hidden_mode(time: float) -> float:
    return math.sin(4 * math.pi * time)


def drift(time: float, amplitude: float = HIDDEN_AMPLITUDE) -> float:
    return BETA_0 + BETA_1 * time + amplitude * hidden_mode(time)


def segment_phases() -> dict:
    return {
        "science": ALPHA + drift(SCIENCE_TIME),
        "anchors": [drift(float(u)) for u in ANCHOR_TIMES],
        "P_zero_science_true": ALPHA + drift(SCIENCE_TIME),
        "P_zero_science_no_hidden_mode": ALPHA + drift(SCIENCE_TIME, 0.0),
    }


def profiled_information(physical: bool, tau: float,
                         anchor_indices: tuple[int, ...]) -> dict:
    """Profile unknown offset, slope, and hidden-mode amplitude."""
    science = model(physical)
    reference = model(False)
    phases = segment_phases()
    n = labels.N
    joint = np.zeros((n + 3, n + 3))
    ones = np.ones(n)

    def basis(time: float) -> np.ndarray:
        return np.array([1.0, time, hidden_mode(time)])

    jacobian = np.concatenate((np.eye(n), np.outer(ones, basis(SCIENCE_TIME))),
                              axis=1)
    science_f = clock.conditional_fisher(science, tau, phases["science"], True)
    joint += jacobian.T @ science_f @ jacobian
    for index in anchor_indices:
        u = float(ANCHOR_TIMES[index])
        jacobian = np.concatenate((np.zeros((n, n)), np.outer(ones, basis(u))),
                                  axis=1)
        anchor_f = clock.conditional_fisher(reference, tau, phases["anchors"][index],
                                           True)
        joint += jacobian.T @ anchor_f @ jacobian
    nuisance = joint[n:, n:]
    effective = joint[:n, :n] - joint[:n, n:] @ np.linalg.pinv(
        nuisance, rcond=1e-12) @ joint[n:, :n]
    effective = (effective + effective.T) / 2
    return {"profiled": clock.rank_summary(effective),
            "joint_rank_tol_1e_minus_8": int(np.sum(np.linalg.eigvalsh(joint) > 1e-8))}


def exact_study() -> dict:
    rows = {}
    for name, physical in (("unrestricted", False), ("P_zero", True)):
        rows[name] = []
        for tau in clock.EXACT_TAUS:
            rows[name].append({
                "tau_over_pi": tau / math.pi,
                "coarse_uniform_5": profiled_information(
                    physical, tau, COARSE_INDICES),
                "coarse_plus_one_holdout_at_1_over_8": profiled_information(
                    physical, tau, COARSE_INDICES + (1,)),
                "fine_uniform_9": profiled_information(
                    physical, tau, tuple(range(9))),
            })
    return rows


def schedule(physical: bool) -> list[tuple]:
    phases = segment_phases()
    common_phases = ([phases["P_zero_science_true"],
                      phases["P_zero_science_no_hidden_mode"]] if physical else
                     [phases["science"]] + phases["anchors"])
    return [(tau_index, segment_index, i, t, common)
            for tau_index in range(len(SHOT_TAUS))
            for segment_index, common in enumerate(common_phases)
            for i in range(labels.N) for t in range(labels.L)]


def run_segments(simulator: AerSimulator, physical: bool,
                 shots: int, seed: int) -> dict:
    entry = model(physical)
    tau_parameter = Parameter("tau")
    phase_parameters = ParameterVector("phase", labels.N)
    template = clock.circuit(entry, tau_parameter, phase_parameters, measure=True)
    compiled = transpile(template, basis_gates=["u3", "cx"], optimization_level=1)
    settings = schedule(physical)
    circuits = []
    for tau_index, segment_index, i, t, common in settings:
        angles = labels.angle(i, t, common)
        values = {tau_parameter: SHOT_TAUS[tau_index]}
        values.update({phase_parameters[axis]: float(angles[axis])
                       for axis in range(labels.N)})
        circuits.append(compiled.assign_parameters(values))
    result = simulator.run(circuits, shots=shots, seed_simulator=seed).result()
    if not result.success:
        raise RuntimeError(result.status)
    segments = 2 if physical else 10
    shape = (len(SHOT_TAUS), segments, labels.N, labels.L)
    accepted = np.zeros(shape, dtype=int)
    joint = np.zeros(shape, dtype=int)
    errors = []
    for index, (tau_index, segment_index, i, t, common) in enumerate(settings):
        counts = result.get_counts(index)
        accepted[tau_index, segment_index, i, t] = sum(
            count for bits, count in counts.items()
            if bits[:clock.CLOCK_QUBITS] == "0" * clock.CLOCK_QUBITS)
        joint[tau_index, segment_index, i, t] = counts.get(
            "0" * clock.TOTAL_QUBITS, 0)
        expected, _ = clock.conditional_probability_gradient(
            entry, SHOT_TAUS[tau_index], labels.angle(i, t, common))
        errors.append(abs(joint[tau_index, segment_index, i, t] / shots -
                          expected / clock.CLOCK_LEVELS))
    return {"clock_success_counts": accepted, "joint_zero_counts": joint,
            "circuits": len(circuits), "transpiled_depth": compiled.depth(),
            "transpiled_size": compiled.size(),
            "overall_clock_postselection_rate": float(accepted.sum() /
                                                        (len(circuits) * shots)),
            "mean_absolute_joint_probability_error": float(np.mean(errors))}


def fit_segment(raw: dict, tau_index: int, segment_index: int) -> dict:
    return clock.fit_full_counts(raw["joint_zero_counts"][tau_index, segment_index],
                                 raw["clock_success_counts"][tau_index, segment_index],
                                 SHOT_TAUS[tau_index])


def contrast(fits: dict[str, dict], coefficients: dict[str, float]) -> dict:
    value = sum(weight * fits[name]["estimate_radians"]
                for name, weight in coefficients.items())
    variance = sum(weight ** 2 * fits[name]["fisher_standard_error_radians"] ** 2
                   for name, weight in coefficients.items())
    se = math.sqrt(variance)
    return {"estimate_radians": value,
            "fisher_standard_error_radians": se,
            "approximate_95pct_interval_radians": [value - 1.96 * se,
                                                    value + 1.96 * se]}


def shot_study(shots: int, workers: int, seed: int, permutations: int) -> dict:
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    full = run_segments(simulator, False, shots, seed)
    physical = run_segments(simulator, True, shots, seed + 1)
    rng = np.random.default_rng(seed + 100)
    rows = []
    for tau_index, tau in enumerate(SHOT_TAUS):
        fits = {"science": fit_segment(full, tau_index, 0)}
        fits.update({f"anchor_{index}": fit_segment(full, tau_index, index + 1)
                     for index in range(9)})
        coarse = contrast(fits, {"science": 1.0, "anchor_2": -0.5,
                                 "anchor_4": -0.5})
        direct = contrast(fits, {"science": 1.0, "anchor_3": -1.0})
        holdout = {}
        for index in HOLDOUT_INDICES:
            residual = contrast(fits, {f"anchor_{index}": 1.0,
                                       f"anchor_{index-1}": -0.5,
                                       f"anchor_{index+1}": -0.5})
            z = residual["estimate_radians"] / residual["fisher_standard_error_radians"]
            residual["gaussian_two_sided_pvalue_vs_zero"] = math.erfc(
                abs(z) / math.sqrt(2))
            holdout[str(index)] = residual
        rows.append({
            "tau_over_pi": tau / math.pi,
            "segment_fits": fits,
            "coarse_5_interpolated_alpha": coarse,
            "fine_9_direct_alpha": direct,
            "holdout_residuals_vs_coarse_interpolation": holdout,
            "P_zero_hidden_mode_vs_absent_labeled": labels.permutation_pvalue(
                physical["joint_zero_counts"][tau_index, 0],
                physical["joint_zero_counts"][tau_index, 1],
                shots, permutations, rng),
        })
    return {
        "shots_per_circuit": shots,
        "full_metadata": {key: value for key, value in full.items()
                          if not isinstance(value, np.ndarray)},
        "P_zero_metadata": {key: value for key, value in physical.items()
                            if not isinstance(value, np.ndarray)},
        "raw_axes": ["tau_index", "segment_0_science_1_to_9_anchor", "axis", "label"],
        "full_clock_success_counts": full["clock_success_counts"].tolist(),
        "full_joint_zero_counts": full["joint_zero_counts"].tolist(),
        "P_zero_clock_success_counts": physical["clock_success_counts"].tolist(),
        "P_zero_joint_zero_counts": physical["joint_zero_counts"].tolist(),
        "tau_results": rows,
    }


def P_zero_control_repeats(shots: int, workers: int, seed: int,
                           permutations: int, repeats: int) -> list[dict]:
    """Audit finite-shot false positives under the exact P=0 common-phase null."""
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    rows = []
    for repeat in range(repeats):
        repeat_seed = seed + 1000 + repeat
        physical = run_segments(simulator, True, shots, repeat_seed)
        rng = np.random.default_rng(repeat_seed + 100)
        pvalues = []
        for tau_index in range(len(SHOT_TAUS)):
            pvalues.append(labels.permutation_pvalue(
                physical["joint_zero_counts"][tau_index, 0],
                physical["joint_zero_counts"][tau_index, 1],
                shots, permutations, rng)["permutation_pvalue"])
        rows.append({
            "seed": repeat_seed,
            "pvalues_by_tau": pvalues,
            "clock_postselection_rate": physical["overall_clock_postselection_rate"],
            "clock_success_counts": physical["clock_success_counts"].tolist(),
            "joint_zero_counts": physical["joint_zero_counts"].tolist(),
        })
    return rows


def binomial_null_calibration(shots: int, seed: int, replicates: int,
                              permutations: int) -> dict:
    """Check false-positive frequency of the stratified permutation statistic."""
    entry = model(True)
    common = segment_phases()["P_zero_science_true"]
    rng = np.random.default_rng(seed + 2000)
    rows = []
    for tau in SHOT_TAUS:
        probabilities = np.array([
            [clock.conditional_probability_gradient(
                entry, tau, labels.angle(i, t, common))[0] / clock.CLOCK_LEVELS
             for t in range(labels.L)] for i in range(labels.N)])
        pvalues = []
        for _ in range(replicates):
            first = rng.binomial(shots, probabilities)
            second = rng.binomial(shots, probabilities)
            pvalues.append(labels.permutation_pvalue(
                first, second, shots, permutations, rng)["permutation_pvalue"])
        rows.append({"tau_over_pi": tau / math.pi,
                     "rejections_at_0_point_05": int(np.sum(np.array(pvalues) <= 0.05)),
                     "rejection_fraction": float(np.mean(np.array(pvalues) <= 0.05)),
                     "pvalues": pvalues})
    return {"replicates_per_tau": replicates,
            "permutations_per_test": permutations,
            "simulated_from_exact_P_zero_joint_probabilities": True,
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--control-repeats", type=int, default=5)
    parser.add_argument("--null-only", action="store_true")
    parser.add_argument("--null-replicates", type=int, default=200)
    parser.add_argument("--null-permutations", type=int, default=199)
    args = parser.parse_args()
    if min(args.shots, args.workers, args.permutations, args.control_repeats,
           args.null_replicates, args.null_permutations) < 1:
        parser.error("all count and worker arguments must be positive")
    if args.control_only and args.null_only:
        parser.error("control-only and null-only are mutually exclusive")
    if args.control_only:
        if not RESULT.exists():
            parser.error("control-only requires an existing full result JSON")
        output = json.loads(RESULT.read_text(encoding="utf-8"))
        if "shots" not in output:
            parser.error("control-only requires finite-shot results in the JSON")
        if args.shots != output["parameters"]["shots_per_circuit"]:
            parser.error("control-only shots must match existing result JSON")
        new_repeats = P_zero_control_repeats(
            args.shots, args.workers, args.seed, args.permutations,
            args.control_repeats)
        old_repeats = output.get("P_zero_control_repeats", [])
        old_seeds = {row["seed"] for row in old_repeats}
        output["P_zero_control_repeats"] = old_repeats + [
            row for row in new_repeats if row["seed"] not in old_seeds]
        RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        print(json.dumps({"P_zero_control_pvalues": [
            row["pvalues_by_tau"] for row in output["P_zero_control_repeats"]],
            "result": str(RESULT)}, indent=2))
        return
    if args.null_only:
        if not RESULT.exists():
            parser.error("null-only requires an existing full result JSON")
        output = json.loads(RESULT.read_text(encoding="utf-8"))
        if args.shots != output["parameters"]["shots_per_circuit"]:
            parser.error("null-only shots must match existing result JSON")
        output["binomial_null_calibration"] = binomial_null_calibration(
            args.shots, args.seed, args.null_replicates, args.null_permutations)
        RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        print(json.dumps({"null_rejection_fractions": [
            row["rejection_fraction"] for row in
            output["binomial_null_calibration"]["rows"]],
            "result": str(RESULT)}, indent=2))
        return
    output = {
        "question": "Can uniformly spaced anchors miss an unmodeled oscillatory phase drift?",
        "physical_limit": "The hidden sinusoidal mode is a constructed counterexample; clock, constraint and known-phase anchors are prepared resources.",
        "parameters": {"alpha_true": ALPHA, "beta_0": BETA_0,
                       "beta_1": BETA_1, "hidden_amplitude": HIDDEN_AMPLITUDE,
                       "science_time": SCIENCE_TIME,
                       "anchor_times": ANCHOR_TIMES.tolist(),
                       "coarse_indices": COARSE_INDICES,
                       "holdout_indices": HOLDOUT_INDICES,
                       "phase_values": segment_phases(),
                       "shots_per_circuit": args.shots,
                       "workers": args.workers, "seed": args.seed},
        "exact": exact_study(),
    }
    if not args.exact_only:
        output["shots"] = shot_study(args.shots, args.workers, args.seed,
                                     args.permutations)
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "exact_ranks": {name: [(
            row["tau_over_pi"],
            row["coarse_uniform_5"]["profiled"]["rank_tol_1e_minus_8"],
            row["coarse_plus_one_holdout_at_1_over_8"]["profiled"]["rank_tol_1e_minus_8"],
            row["fine_uniform_9"]["profiled"]["rank_tol_1e_minus_8"])
            for row in rows] for name, rows in output["exact"].items()},
        "coarse_alpha": [row["coarse_5_interpolated_alpha"]["estimate_radians"]
                         for row in output.get("shots", {}).get("tau_results", [])],
        "direct_alpha": [row["fine_9_direct_alpha"]["estimate_radians"]
                         for row in output.get("shots", {}).get("tau_results", [])],
        "result": str(RESULT),
    }, indent=2))


if __name__ == "__main__":
    main()
