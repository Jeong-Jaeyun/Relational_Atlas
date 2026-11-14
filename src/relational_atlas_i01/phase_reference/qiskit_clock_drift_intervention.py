"""I01: test whether clock-conditioned phase anchors track intra-block drift.

Science is taken halfway between two endpoint anchors. A curvature component
vanishes at both endpoints but changes science's phase. Endpoint-only
calibration therefore confounds the common science phase and curvature; a
midpoint anchor measures the missing nuisance direction. The known-phase
anchor and clock-matched states remain prepared resources, not dynamics.
"""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np
from qiskit_aer import AerSimulator

from relational_atlas_i01.phase_reference import qiskit_clock_anchor_intervention as clock
from relational_atlas_i01.phase_reference import qiskit_phase_anchor_audit as anchor
from relational_atlas_i01.phase_reference import qiskit_phase_label_intervention as labels


PROJECT = PROJECT_ROOT
RESULT = PROJECT / "results" / "qiskit_clock_drift_intervention.json"
ALPHA = 0.30
BETA_0 = -0.15
BETA_1 = 0.50
CURVATURE = 0.22
TAUS = clock.SHOT_TAUS
SEGMENTS = ("science_mid", "anchor_start", "anchor_end", "anchor_mid")


@lru_cache(maxsize=2)
def model(physical: bool) -> dict:
    return clock.clock_model(physical)


def drift(time: float, curvature: float = CURVATURE) -> float:
    return BETA_0 + BETA_1 * time + 4 * curvature * time * (1 - time)


def segment_phases() -> dict[str, float]:
    return {
        "science_mid": ALPHA + drift(0.5),
        "anchor_start": drift(0.0),
        "anchor_end": drift(1.0),
        "anchor_mid": drift(0.5),
        "P_zero_science_curved": ALPHA + drift(0.5),
        "P_zero_science_affine": ALPHA + drift(0.5, 0.0),
    }


def profiled_information(physical: bool, tau: float,
                         include_mid_anchor: bool, allow_curvature: bool) -> dict:
    """Profile phase-drift nuisance parameters from the joint Fisher matrix."""
    science = model(physical)
    reference = model(False)
    phases = segment_phases()
    nuisance = 3 if allow_curvature else 2
    dimension = labels.N + nuisance
    joint = np.zeros((dimension, dimension))
    ones = np.ones(labels.N)

    def basis(time: float) -> np.ndarray:
        coefficients = [1.0, time]
        if allow_curvature:
            coefficients.append(4 * time * (1 - time))
        return np.asarray(coefficients)

    science_jacobian = np.concatenate(
        (np.eye(labels.N), np.outer(ones, basis(0.5))), axis=1)
    local = clock.conditional_fisher(science, tau, phases["science_mid"], True)
    joint += science_jacobian.T @ local @ science_jacobian

    for time, name in ((0.0, "anchor_start"), (1.0, "anchor_end"),
                       (0.5, "anchor_mid")):
        if name == "anchor_mid" and not include_mid_anchor:
            continue
        anchor_jacobian = np.concatenate(
            (np.zeros((labels.N, labels.N)), np.outer(ones, basis(time))), axis=1)
        local = clock.conditional_fisher(reference, tau, phases[name], True)
        joint += anchor_jacobian.T @ local @ anchor_jacobian

    effective = joint[:labels.N, :labels.N] - joint[:labels.N, labels.N:] @ np.linalg.pinv(
        joint[labels.N:, labels.N:], rcond=1e-12) @ joint[labels.N:, :labels.N]
    effective = (effective + effective.T) / 2
    common = np.ones(labels.N) / np.sqrt(labels.N)
    return {
        "profiled": clock.rank_summary(effective),
        "joint_rank_tol_1e_minus_8": int(np.sum(np.linalg.eigvalsh(joint) > 1e-8)),
        "common_direction_information": float(common @ effective @ common),
        "nuisance_parameters": nuisance,
    }


def exact_study() -> dict:
    rows = {}
    for name, physical in (("unrestricted", False), ("P_zero", True)):
        rows[name] = []
        for tau in clock.EXACT_TAUS:
            rows[name].append({
                "tau_over_pi": tau / math.pi,
                "affine_assumption_endpoint_anchors": profiled_information(
                    physical, tau, False, False),
                "curvature_allowed_endpoint_anchors": profiled_information(
                    physical, tau, False, True),
                "curvature_allowed_midpoint_anchor": profiled_information(
                    physical, tau, True, True),
            })
    return rows


def paired_schedule(phases: dict[str, float], physical: bool) -> list[tuple]:
    schedule = []
    for tau_index in range(len(TAUS)):
        for block in range(2):
            for run_type in range(2):
                if physical:
                    if run_type != 0:
                        continue
                    name = ("P_zero_science_curved" if block == 0
                            else "P_zero_science_affine")
                else:
                    name = SEGMENTS[2 * block + run_type]
                for i in range(labels.N):
                    for t in range(labels.L):
                        schedule.append((tau_index, block, run_type, i, t, phases[name]))
    return schedule


def fit_segment(raw: dict, tau_index: int, segment_index: int) -> dict:
    block, run_type = divmod(segment_index, 2)
    return clock.fit_full_counts(raw["joint_zero_counts"][tau_index, block, run_type],
                                 raw["clock_success_counts"][tau_index, block, run_type],
                                 TAUS[tau_index])


def independent_contrast(fits: dict, coefficients: dict[str, float]) -> dict:
    value = sum(weight * fits[name]["estimate_radians"]
                for name, weight in coefficients.items())
    variance = sum(weight ** 2 * fits[name]["fisher_standard_error_radians"] ** 2
                   for name, weight in coefficients.items())
    se = math.sqrt(variance)
    return {"estimate_radians": value, "fisher_standard_error_radians": se,
            "approximate_95pct_interval_radians": [value - 1.96 * se,
                                                    value + 1.96 * se]}


def shot_study(shots: int, workers: int, seed: int, permutations: int) -> dict:
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    phases = segment_phases()
    full = clock.run_circuits(simulator, model(False),
                              paired_schedule(phases, False), shots, seed)
    physical = clock.run_circuits(simulator, model(True),
                                  paired_schedule(phases, True), shots, seed + 1)
    rng = np.random.default_rng(seed + 100)
    rows = []
    for tau_index, tau in enumerate(TAUS):
        fits = {name: fit_segment(full, tau_index, index)
                for index, name in enumerate(SEGMENTS)}
        naive = independent_contrast(
            fits, {"science_mid": 1.0, "anchor_start": -0.5, "anchor_end": -0.5})
        corrected = independent_contrast(
            fits, {"science_mid": 1.0, "anchor_mid": -1.0})
        curvature = independent_contrast(
            fits, {"anchor_mid": 1.0, "anchor_start": -0.5, "anchor_end": -0.5})
        z = curvature["estimate_radians"] / curvature["fisher_standard_error_radians"]
        curvature["gaussian_two_sided_pvalue_vs_zero"] = math.erfc(abs(z) / math.sqrt(2))
        rows.append({
            "tau_over_pi": tau / math.pi,
            "segment_fits": fits,
            "endpoint_affine_alpha_estimate": naive,
            "midpoint_corrected_alpha_estimate": corrected,
            "measured_curvature": curvature,
            "P_zero_curved_vs_affine_labeled": labels.permutation_pvalue(
                physical["joint_zero_counts"][tau_index, 0, 0],
                physical["joint_zero_counts"][tau_index, 1, 0],
                shots, permutations, rng),
        })
    return {
        "shots_per_circuit": shots,
        "full_metadata": {key: value for key, value in full.items()
                          if not isinstance(value, np.ndarray)},
        "P_zero_metadata": {key: value for key, value in physical.items()
                            if not isinstance(value, np.ndarray)},
        "full_segment_map": {str(index): name for index, name in enumerate(SEGMENTS)},
        "full_clock_success_counts": full["clock_success_counts"].tolist(),
        "full_joint_zero_counts": full["joint_zero_counts"].tolist(),
        "P_zero_clock_success_counts": physical["clock_success_counts"].tolist(),
        "P_zero_joint_zero_counts": physical["joint_zero_counts"].tolist(),
        "tau_results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--exact-only", action="store_true")
    args = parser.parse_args()
    if min(args.shots, args.workers, args.permutations) < 1:
        parser.error("shots, workers, permutations must be positive")
    output = {
        "question": "Can midpoint science phase be distinguished from untracked intra-block curvature?",
        "physical_limit": "Energy-matched clock, P=0 state and known-phase anchor are prepared by construction.",
        "parameters": {"alpha_true": ALPHA, "beta_0": BETA_0, "beta_1": BETA_1,
                       "curvature_true": CURVATURE, "segment_phases": segment_phases(),
                       "shots_per_circuit": args.shots, "workers": args.workers,
                       "seed": args.seed},
        "exact": exact_study(),
    }
    if not args.exact_only:
        output["shots"] = shot_study(args.shots, args.workers, args.seed,
                                     args.permutations)
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "exact_ranks": {name: [(
            row["tau_over_pi"],
            row["affine_assumption_endpoint_anchors"]["profiled"]["rank_tol_1e_minus_8"],
            row["curvature_allowed_endpoint_anchors"]["profiled"]["rank_tol_1e_minus_8"],
            row["curvature_allowed_midpoint_anchor"]["profiled"]["rank_tol_1e_minus_8"])
            for row in rows] for name, rows in output["exact"].items()},
        "shot_alpha_affine": [row["endpoint_affine_alpha_estimate"]["estimate_radians"]
                              for row in output.get("shots", {}).get("tau_results", [])],
        "shot_alpha_corrected": [row["midpoint_corrected_alpha_estimate"]["estimate_radians"]
                                 for row in output.get("shots", {}).get("tau_results", [])],
        "result": str(RESULT),
    }, indent=2))


if __name__ == "__main__":
    main()
