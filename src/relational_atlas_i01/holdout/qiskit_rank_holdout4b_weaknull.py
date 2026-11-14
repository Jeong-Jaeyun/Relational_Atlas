"""Boundary-preserving weak-pooled-null test and independent Aer rerun."""

from __future__ import annotations

import argparse
import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout3_audit as audit
from relational_atlas_i01.holdout import qiskit_rank_holdout3_shots as acquisition
from relational_atlas_i01.holdout import qiskit_rank_holdout4_weaknull as additive


PROJECT = PROJECT_ROOT
RESULT = PROJECT / "results" / "qiskit_rank_holdout4b_weaknull.json"
DIAGNOSTIC = PROJECT / "results" / "qiskit_rank_holdout3_weaknull_logistic_diagnostic.json"
PROTOCOL = "notes/prevalidation_protocol4b_2026-10-04.tex"


def logistic(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def logit(value: np.ndarray) -> np.ndarray:
    return np.log(value / (1 - value))


def fitted_weak_null(first: np.ndarray, second: np.ndarray,
                     shots_by_label: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                           np.ndarray]:
    weights = shots_by_label / shots_by_label.sum()
    first_rate = (first + 0.5) / (shots_by_label + 1)
    second_rate = (second + 0.5) / (shots_by_label + 1)
    log_first = logit(first_rate)
    log_second = logit(second_rate)
    fitted_first = np.empty_like(first_rate)
    fitted_second = np.empty_like(second_rate)
    shifts = np.empty(first.shape[0])
    for axis in range(first.shape[0]):
        lower, upper = -40.0, 40.0
        for _ in range(100):
            middle = (lower + upper) / 2
            difference = np.sum(weights * (
                logistic(log_second[axis] - middle) -
                logistic(log_first[axis] + middle)))
            if difference > 0:
                lower = middle
            else:
                upper = middle
        shifts[axis] = (lower + upper) / 2
        fitted_first[axis] = logistic(log_first[axis] + shifts[axis])
        fitted_second[axis] = logistic(log_second[axis] - shifts[axis])
    if np.max(np.abs((fitted_second - fitted_first) @ weights)) > 1e-12:
        raise AssertionError("bounded projection failed weighted equality")
    if (np.min(fitted_first) <= 0 or np.max(fitted_first) >= 1 or
            np.min(fitted_second) <= 0 or np.max(fitted_second) >= 1):
        raise AssertionError("bounded projection left probability range")
    return fitted_first, fitted_second, shifts


def weak_null_test(first: np.ndarray, second: np.ndarray,
                   shots_by_label: np.ndarray, bootstraps: int,
                   rng: np.random.Generator) -> dict:
    if first.shape != (3, 7) or second.shape != (3, 7):
        raise ValueError("expected axis x label counts")
    difference, variance, observed = additive.statistic(
        first, second, shots_by_label)
    fitted_first, fitted_second, shifts = fitted_weak_null(
        first, second, shots_by_label)
    weights = shots_by_label / shots_by_label.sum()
    exceeding = 0
    for start in range(0, bootstraps, 500):
        batch = min(500, bootstraps - start)
        sampled_first = rng.binomial(
            shots_by_label[None, None, :], fitted_first[None, :, :],
            size=(batch, 3, 7))
        sampled_second = rng.binomial(
            shots_by_label[None, None, :], fitted_second[None, :, :],
            size=(batch, 3, 7))
        rates_first = sampled_first / shots_by_label
        rates_second = sampled_second / shots_by_label
        delta = np.sum(weights * (rates_second - rates_first), axis=-1)
        var = np.sum(weights ** 2 * (
            rates_first * (1 - rates_first) +
            rates_second * (1 - rates_second)) / shots_by_label, axis=-1)
        stat = np.sum(delta * delta / var, axis=-1)
        exceeding += int(np.sum(stat >= observed))
    return {"axis_differences": difference.tolist(),
            "axis_standard_errors": np.sqrt(variance).tolist(),
            "wald_chi_square_df3": observed,
            "logit_null_shifts": shifts.tolist(),
            "fitted_null_max_weighted_difference": float(np.max(
                np.abs((fitted_second - fitted_first) @ weights))),
            "weak_null_bootstraps": bootstraps,
            "weak_null_bootstrap_pvalue": (1 + exceeding) / (bootstraps + 1),
            "chi_square_df3_pvalue_crosscheck":
                audit.chi_square_three_survival(observed)}


def analyze_raw(raw: dict, bootstraps: int, seed: int) -> dict:
    counts = np.asarray(raw["raw_zero_counts"], dtype=int)
    exact = np.asarray(raw["exact_zero_probabilities"], dtype=float)
    if counts.shape != (2, 2, 2, 2, 3, 7):
        raise ValueError("unexpected raw count shape")
    rng = np.random.default_rng(seed)
    rows = []
    for dist_index, distribution in enumerate(acquisition.DISTRIBUTIONS):
        shots_by_label = np.asarray(raw["shot_allocations"][distribution],
                                    dtype=int)
        weights = shots_by_label / shots_by_label.sum()
        for physical in (0, 1):
            for tau_index, tau in enumerate(acquisition.TAUS):
                first = counts[dist_index, physical, tau_index, 0]
                second = counts[dist_index, physical, tau_index, 1]
                exact_delta = (exact[physical, tau_index, 1] -
                               exact[physical, tau_index, 0]) @ weights
                test = weak_null_test(first, second, shots_by_label,
                                      bootstraps, rng)
                rows.append({"label_distribution": distribution,
                             "physical_P_zero": bool(physical),
                             "tau_over_pi": tau / math.pi,
                             "exact_axis_pooled_differences": exact_delta.tolist(),
                             "exact_max_pooled_probability_difference":
                                 float(np.max(np.abs(exact_delta))),
                             "test": test})
    threshold = 0.05 / len(rows)
    return {"protocol": PROTOCOL,
            "familywise_tests": len(rows),
            "familywise_bonferroni_threshold": threshold,
            "rows": rows,
            "detections": [(row["label_distribution"],
                            row["physical_P_zero"], row["tau_over_pi"])
                           for row in rows if row["test"][
                               "weak_null_bootstrap_pvalue"] <= threshold]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--bootstraps", type=int, default=10000)
    args = parser.parse_args()
    if min(args.workers, args.bootstraps) < 1:
        parser.error("workers and bootstraps must be positive")
    if args.diagnostic_only:
        source = json.loads(acquisition.RESULT.read_text(encoding="utf-8"))
        output = analyze_raw(source, args.bootstraps, args.seed + 1000)
        output["caveat"] = (
            "Post-hoc analysis of previously observed Aer counts; diagnostic only.")
        destination = DIAGNOSTIC
    else:
        fresh = acquisition.run(32768, args.workers, args.seed, 1)
        output = analyze_raw(fresh, args.bootstraps, args.seed + 1000)
        output["parameters"] = fresh["parameters"]
        output["parameters"].pop("permutations", None)
        output["parameters"]["weak_null_bootstraps"] = args.bootstraps
        output["shot_allocations"] = fresh["shot_allocations"]
        output["raw_axes"] = fresh["raw_axes"]
        output["raw_zero_counts"] = fresh["raw_zero_counts"]
        output["exact_probabilities_axes"] = fresh["exact_probabilities_axes"]
        output["exact_zero_probabilities"] = fresh["exact_zero_probabilities"]
        output["caveat"] = (
            "Independent Aer seed and bounded weak-null procedure frozen before rerun.")
        destination = RESULT
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({"diagnostic_only": args.diagnostic_only,
                      "threshold": output["familywise_bonferroni_threshold"],
                      "pvalues": [(row["label_distribution"],
                                   row["physical_P_zero"], row["tau_over_pi"],
                                   row["test"]["weak_null_bootstrap_pvalue"])
                                  for row in output["rows"]],
                      "detections": output["detections"],
                      "result": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
