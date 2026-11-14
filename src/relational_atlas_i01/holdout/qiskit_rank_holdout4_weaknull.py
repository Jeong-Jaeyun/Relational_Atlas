"""Independent Aer rerun with a test calibrated for the weak pooled null."""

from __future__ import annotations

import argparse
import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout3_audit as audit
from relational_atlas_i01.holdout import qiskit_rank_holdout3_shots as acquisition


PROJECT = PROJECT_ROOT
RESULT = PROJECT / "results" / "qiskit_rank_holdout4_weaknull.json"
DIAGNOSTIC = PROJECT / "results" / "qiskit_rank_holdout3_weaknull_diagnostic.json"
PROTOCOL = "notes/prevalidation_protocol4_2026-10-04.tex"


def statistic(first: np.ndarray, second: np.ndarray,
              shots_by_label: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    trials = int(shots_by_label.sum())
    weights = shots_by_label / trials
    first_rate = first / shots_by_label
    second_rate = second / shots_by_label
    difference = np.sum(weights * (second_rate - first_rate), axis=-1)
    variance = np.sum(weights ** 2 * (
        first_rate * (1 - first_rate) +
        second_rate * (1 - second_rate)) / shots_by_label, axis=-1)
    if np.any(variance <= 0):
        raise ArithmeticError("zero estimated variance")
    value = float(np.sum(difference * difference / variance))
    return difference, variance, value


def weak_null_test(first: np.ndarray, second: np.ndarray,
                   shots_by_label: np.ndarray, bootstraps: int,
                   rng: np.random.Generator) -> dict:
    if first.shape != (3, 7) or second.shape != (3, 7):
        raise ValueError("expected axis x label counts")
    if bootstraps < 1:
        raise ValueError("bootstraps must be positive")
    difference, variance, observed = statistic(first, second, shots_by_label)
    first_rate = first / shots_by_label
    second_rate = second / shots_by_label
    fitted_first = first_rate + difference[:, None] / 2
    fitted_second = second_rate - difference[:, None] / 2
    if (np.min(fitted_first) < 0 or np.max(fitted_first) > 1 or
            np.min(fitted_second) < 0 or np.max(fitted_second) > 1):
        raise ArithmeticError("fitted weak-null probabilities outside [0,1]")
    weights = shots_by_label / shots_by_label.sum()
    if np.max(np.abs((fitted_second - fitted_first) @ weights)) > 1e-12:
        raise AssertionError("weak-null projection did not satisfy constraint")
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
            "weak_null_bootstraps": bootstraps,
            "weak_null_bootstrap_pvalue": (1 + exceeding) / (bootstraps + 1),
            "chi_square_df3_pvalue_crosscheck":
                audit.chi_square_three_survival(observed),
            "fitted_null_max_weighted_difference": float(np.max(
                np.abs((fitted_second - fitted_first) @ weights)))}


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
    parser.add_argument("--seed", type=int, default=20261019)
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
        output["parameters"]["weak_null_bootstraps"] = args.bootstraps
        output["shot_allocations"] = fresh["shot_allocations"]
        output["raw_axes"] = fresh["raw_axes"]
        output["raw_zero_counts"] = fresh["raw_zero_counts"]
        output["exact_probabilities_axes"] = fresh["exact_probabilities_axes"]
        output["exact_zero_probabilities"] = fresh["exact_zero_probabilities"]
        output["caveat"] = (
            "Independent seed and weak-null procedure frozen before this rerun.")
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
