"""I01: stochastic phase drift between bracketed anchors and science runs.

An OU covariance is an explicitly assumed toy prior. Exact Gaussian bridge
variance gives an optimistic reference-noise floor even if endpoint anchors
are measured perfectly. Aer samples finite-shot clock-conditioned circuits
for selected independent blocks; a larger Gaussian Monte Carlo maps coverage.
"""

from __future__ import annotations

import argparse
import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np
from qiskit import transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator

from relational_atlas_i01.phase_reference import qiskit_clock_anchor_intervention as clock
from relational_atlas_i01.phase_reference import qiskit_phase_label_intervention as labels


PROJECT = PROJECT_ROOT
RESULT = PROJECT / "results" / "qiskit_clock_stochastic_bridge.json"
ALPHA = 0.30
SIGMA = 0.12
GAP_OVER_CORRELATION = (0.05, 1.0)
BLOCKS_PER_GAP = 8
SEGMENTS = ("anchor_before", "science", "anchor_after")
TAUS = (0.0, math.pi / 2)


def bridge_parameters(gap_over_correlation: float,
                      sigma: float = SIGMA) -> tuple[float, float]:
    """Return the endpoint coefficient and residual variance at the midpoint."""
    if gap_over_correlation < 0 or sigma < 0:
        raise ValueError("gap and sigma must be nonnegative")
    rho = math.exp(-gap_over_correlation)
    coefficient = rho / (1 + rho * rho)
    residual_variance = sigma * sigma * math.tanh(gap_over_correlation)
    return coefficient, residual_variance


def bridge_draws(seed: int, blocks: int = BLOCKS_PER_GAP) -> dict:
    rng = np.random.default_rng(seed)
    rows = {}
    for gap in GAP_OVER_CORRELATION:
        coefficient, residual_variance = bridge_parameters(gap)
        rho = math.exp(-gap)
        covariance = SIGMA ** 2 * np.array([[1.0, rho * rho],
                                              [rho * rho, 1.0]])
        anchors = rng.multivariate_normal(np.zeros(2), covariance, size=blocks)
        conditional_means = coefficient * anchors.sum(axis=1)
        science_offsets = conditional_means + rng.normal(
            scale=math.sqrt(residual_variance), size=blocks)
        rows[str(gap)] = [
            {"anchor_before": float(anchors[index, 0]),
             "science_offset": float(science_offsets[index]),
             "anchor_after": float(anchors[index, 1]),
             "conditional_mean": float(conditional_means[index]),
             "unobserved_bridge_residual": float(
                 science_offsets[index] - conditional_means[index])}
            for index in range(blocks)]
    return rows


def conditional_phase_fisher(tau: float, phase: float, shots: int) -> dict:
    """Finite-shot information for one phase run, including clock failures."""
    h = 1e-5
    p = clock.full_conditional_probabilities(np.array([phase]), tau)[0]
    plus = clock.full_conditional_probabilities(np.array([phase + h]), tau)[0]
    minus = clock.full_conditional_probabilities(np.array([phase - h]), tau)[0]
    derivative = (plus - minus) / (2 * h)
    accepted = shots / clock.CLOCK_LEVELS
    information = float(labels.N * accepted * np.sum(
        derivative ** 2 / (p * (1 - p))))
    return {"information_per_36_circuits": information,
            "phase_standard_error_radians": float(1 / math.sqrt(information))}


def uncertainty_map(shots: int, seed: int, replicates: int = 20000,
                    blocks: int = BLOCKS_PER_GAP) -> dict:
    """Coverage under the declared OU prior and Gaussian shot approximation."""
    rng = np.random.default_rng(seed + 2000)
    rows = []
    for tau in TAUS:
        science_variance = conditional_phase_fisher(tau, ALPHA, shots)[
            "phase_standard_error_radians"] ** 2
        anchor_variance = conditional_phase_fisher(tau, 0.0, shots)[
            "phase_standard_error_radians"] ** 2
        for gap in GAP_OVER_CORRELATION:
            coefficient, residual_variance = bridge_parameters(gap)
            finite_shot_variance = science_variance + 2 * coefficient ** 2 * anchor_variance
            noise = rng.normal(scale=math.sqrt(finite_shot_variance / blocks),
                               size=replicates)
            bridge = rng.normal(scale=math.sqrt(residual_variance / blocks),
                                size=replicates)
            errors = noise + bridge
            naive_halfwidth = 1.96 * math.sqrt(finite_shot_variance / blocks)
            aware_halfwidth = 1.96 * math.sqrt(
                (finite_shot_variance + residual_variance) / blocks)
            target_halfwidth = 0.02
            rows.append({
                "tau_over_pi": tau / math.pi,
                "gap_over_correlation": gap,
                "bridge_coefficient": coefficient,
                "irreducible_bridge_sd_per_block": math.sqrt(residual_variance),
                "finite_shot_sd_per_block_approx": math.sqrt(finite_shot_variance),
                "blocks_in_coverage_study": blocks,
                "replicates": replicates,
                "shot_only_95pct_coverage": float(np.mean(
                    np.abs(errors) <= naive_halfwidth)),
                "bridge_aware_95pct_coverage": float(np.mean(
                    np.abs(errors) <= aware_halfwidth)),
                "shot_only_halfwidth": naive_halfwidth,
                "bridge_aware_halfwidth": aware_halfwidth,
                "blocks_for_0_point_02_rad_halfwidth": math.ceil(
                    1.96 ** 2 * (finite_shot_variance + residual_variance) /
                    target_halfwidth ** 2),
            })
    return {"assumption": "Known zero-mean OU covariance, independent blocks, Gaussian phase-fit noise",
            "rows": rows}


def free_innovation_profile_rank(tau: float) -> int:
    """Without a drift prior, science-only innovation absorbs common phase."""
    entry = clock.clock_model(False)
    science_f = clock.conditional_fisher(entry, tau, ALPHA, True)
    ones = np.ones(labels.N)
    cross = science_f @ ones
    effective = science_f - np.outer(cross, cross) / (ones @ science_f @ ones)
    return clock.rank_summary((effective + effective.T) / 2)["rank_tol_1e_minus_8"]


def compiled_template(physical: bool = False):
    entry = clock.clock_model(physical)
    tau_parameter = Parameter("tau")
    phase_parameters = ParameterVector("phase", labels.N)
    template = clock.circuit(entry, tau_parameter, phase_parameters, measure=True)
    compiled = transpile(template, basis_gates=["u3", "cx"], optimization_level=1)
    return compiled, tau_parameter, phase_parameters, entry


def run_block(simulator: AerSimulator, compiled, tau_parameter, phase_parameters,
              entry: dict, tau: float, phases: tuple[float, float, float],
              shots: int, seed: int) -> dict:
    settings = [(segment, i, t, phase)
                for segment, phase in enumerate(phases)
                for i in range(labels.N) for t in range(labels.L)]
    circuits = []
    for segment, i, t, phase in settings:
        angles = labels.angle(i, t, phase)
        bindings = {tau_parameter: tau}
        bindings.update({phase_parameters[axis]: float(angles[axis])
                         for axis in range(labels.N)})
        circuits.append(compiled.assign_parameters(bindings))
    result = simulator.run(circuits, shots=shots, seed_simulator=seed).result()
    if not result.success:
        raise RuntimeError(result.status)
    accepted = np.zeros((3, labels.N, labels.L), dtype=int)
    joint = np.zeros_like(accepted)
    errors = []
    for index, (segment, i, t, phase) in enumerate(settings):
        counts = result.get_counts(index)
        accepted[segment, i, t] = sum(
            count for bits, count in counts.items()
            if bits[:clock.CLOCK_QUBITS] == "0" * clock.CLOCK_QUBITS)
        joint[segment, i, t] = counts.get("0" * clock.TOTAL_QUBITS, 0)
        expected, _ = clock.conditional_probability_gradient(
            entry, tau, labels.angle(i, t, phase))
        errors.append(abs(joint[segment, i, t] / shots -
                          expected / clock.CLOCK_LEVELS))
    return {"clock_success_counts": accepted, "joint_zero_counts": joint,
            "mean_absolute_joint_probability_error": float(np.mean(errors)),
            "clock_postselection_rate": float(accepted.sum() /
                                               (len(settings) * shots))}


def aer_study(shots: int, workers: int, seed: int,
              draws: dict) -> dict:
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    compiled, tau_parameter, phase_parameters, entry = compiled_template()
    rows = []
    run_index = 0
    for gap in GAP_OVER_CORRELATION:
        coefficient, residual_variance = bridge_parameters(gap)
        for block, triple in enumerate(draws[str(gap)]):
            phases = (triple["anchor_before"], ALPHA + triple["science_offset"],
                      triple["anchor_after"])
            raw = run_block(simulator, compiled, tau_parameter, phase_parameters,
                            entry, TAUS[0], phases, shots, seed + 100 + run_index)
            run_index += 1
            fits = [clock.fit_full_counts(raw["joint_zero_counts"][segment],
                                          raw["clock_success_counts"][segment], TAUS[0])
                    for segment in range(3)]
            estimate = (fits[1]["estimate_radians"] - coefficient *
                        (fits[0]["estimate_radians"] + fits[2]["estimate_radians"]))
            shot_variance = (fits[1]["fisher_standard_error_radians"] ** 2 +
                             coefficient ** 2 * (
                                 fits[0]["fisher_standard_error_radians"] ** 2 +
                                 fits[2]["fisher_standard_error_radians"] ** 2))
            rows.append({
                "gap_over_correlation": gap, "block": block,
                "true_phases": dict(zip(SEGMENTS, phases)),
                "true_bridge_residual": triple["unobserved_bridge_residual"],
                "fitted_phases": dict(zip(SEGMENTS, fits)),
                "alpha_estimate": estimate,
                "shot_only_standard_error": math.sqrt(shot_variance),
                "bridge_aware_standard_error": math.sqrt(
                    shot_variance + residual_variance),
                "clock_postselection_rate": raw["clock_postselection_rate"],
                "mean_absolute_joint_probability_error": raw[
                    "mean_absolute_joint_probability_error"],
                "clock_success_counts": raw["clock_success_counts"].tolist(),
                "joint_zero_counts": raw["joint_zero_counts"].tolist(),
            })
    summaries = []
    for gap in GAP_OVER_CORRELATION:
        selected = [row for row in rows if row["gap_over_correlation"] == gap]
        estimates = np.array([row["alpha_estimate"] for row in selected])
        shot_variance = sum(row["shot_only_standard_error"] ** 2 for row in selected)
        shot_variance /= len(selected) ** 2
        _, residual_variance = bridge_parameters(gap)
        total_variance = shot_variance + residual_variance / len(selected)
        summaries.append({
            "gap_over_correlation": gap,
            "blocks": len(selected),
            "combined_alpha_mean": float(np.mean(estimates)),
            "shot_only_95pct_interval": [float(np.mean(estimates) -
                                               1.96 * math.sqrt(shot_variance)),
                                          float(np.mean(estimates) +
                                               1.96 * math.sqrt(shot_variance))],
            "bridge_aware_95pct_interval": [float(np.mean(estimates) -
                                                  1.96 * math.sqrt(total_variance)),
                                             float(np.mean(estimates) +
                                                  1.96 * math.sqrt(total_variance))],
        })
    return {"shots_per_circuit": shots, "circuits": len(rows) * 3 * labels.N * labels.L,
            "tau_over_pi": 0.0,
            "transpiled_depth": compiled.depth(),
            "raw_count_axes": ["segment_before_science_after", "reference_axis", "phase_label"],
            "block_rows": rows, "summaries": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--monte-carlo", type=int, default=20000)
    parser.add_argument("--exact-only", action="store_true")
    args = parser.parse_args()
    if min(args.shots, args.workers, args.monte_carlo) < 1:
        parser.error("shots, workers, and monte-carlo must be positive")
    draws = bridge_draws(args.seed)
    output = {
        "question": "How does nonzero anchor-science time separation limit common-phase recovery?",
        "physical_limit": "Known zero-mean OU phase covariance is a toy prior; clock and anchor are prepared resources.",
        "parameters": {"alpha_true": ALPHA, "sigma_radians": SIGMA,
                       "gap_over_correlation": GAP_OVER_CORRELATION,
                       "blocks_per_gap": BLOCKS_PER_GAP,
                       "seed": args.seed, "shots_per_circuit": args.shots,
                       "workers": args.workers},
        "unobserved_phase_draws_for_reproducibility": draws,
        "exact": {"bridge_parameters": [
            {"gap_over_correlation": gap,
             "anchor_coefficient": bridge_parameters(gap)[0],
             "residual_variance": bridge_parameters(gap)[1]}
            for gap in GAP_OVER_CORRELATION],
                  "free_innovation_profiled_ranks": [
                      {"tau_over_pi": tau / math.pi,
                       "rank": free_innovation_profile_rank(tau)} for tau in TAUS],
                  "uncertainty_map": uncertainty_map(
                      args.shots, args.seed, args.monte_carlo)},
    }
    if not args.exact_only:
        output["aer"] = aer_study(args.shots, args.workers, args.seed, draws)
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "free_innovation_ranks": output["exact"]["free_innovation_profiled_ranks"],
        "coverage": [(row["tau_over_pi"], row["gap_over_correlation"],
                      row["shot_only_95pct_coverage"],
                      row["bridge_aware_95pct_coverage"])
                     for row in output["exact"]["uncertainty_map"]["rows"]],
        "aer_summaries": output.get("aer", {}).get("summaries", []),
        "result": str(RESULT),
    }, indent=2))


if __name__ == "__main__":
    main()
