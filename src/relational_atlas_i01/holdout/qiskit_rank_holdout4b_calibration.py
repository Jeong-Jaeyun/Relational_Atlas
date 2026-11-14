"""Post-hoc method calibration against exact-binomial model panels.

This is a statistical quality audit, not a new physical holdout. It checks a
weak pooled null where individual phase-label probabilities differ by arm.
"""

from __future__ import annotations

import argparse
import json
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout4b_weaknull as corrected


PROJECT = PROJECT_ROOT
RESULT = PROJECT / "results" / "qiskit_rank_holdout4b_calibration.json"
CASES = (
    ("weak_null_uniform_full_tau0", 0, 0, 0, True),
    ("weak_null_uniform_full_half", 0, 0, 1, True),
    ("strong_null_skew_P_zero_tau0", 1, 1, 0, True),
    ("positive_skew_full_tau0", 1, 0, 0, False),
    ("weak_positive_skew_full_half", 1, 0, 1, False),
)


def study(panels: int = 200, bootstraps: int = 499,
          seed: int = 20261022) -> dict:
    source = json.loads(corrected.RESULT.read_text(encoding="utf-8"))
    exact = np.asarray(source["exact_zero_probabilities"], dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for name, dist_index, physical, tau_index, exact_null in CASES:
        distribution = ("uniform", "skewed")[dist_index]
        allocation = np.asarray(source["shot_allocations"][distribution],
                                dtype=int)
        first_p = exact[physical, tau_index, 0]
        second_p = exact[physical, tau_index, 1]
        exact_difference = (second_p - first_p) @ (allocation / allocation.sum())
        first_counts = rng.binomial(allocation[None, None, :],
                                    first_p[None, :, :],
                                    size=(panels, 3, 7))
        second_counts = rng.binomial(allocation[None, None, :],
                                     second_p[None, :, :],
                                     size=(panels, 3, 7))
        bootstrap_p = []
        chi_square_p = []
        for panel in range(panels):
            result = corrected.weak_null_test(
                first_counts[panel], second_counts[panel], allocation,
                bootstraps, rng)
            bootstrap_p.append(result["weak_null_bootstrap_pvalue"])
            chi_square_p.append(result["chi_square_df3_pvalue_crosscheck"])
        bootstrap_p = np.asarray(bootstrap_p)
        chi_square_p = np.asarray(chi_square_p)
        rows.append({
            "name": name, "exact_null": exact_null,
            "exact_max_pooled_probability_difference": float(np.max(
                np.abs(exact_difference))),
            "max_labelwise_probability_difference": float(np.max(
                np.abs(second_p - first_p))),
            "panels": panels, "bootstrap_repetitions_per_panel": bootstraps,
            "bootstrap_rejections_at_0_point_05": int(np.sum(
                bootstrap_p <= 0.05)),
            "bootstrap_rejections_at_0_point_00625": int(np.sum(
                bootstrap_p <= 0.00625)),
            "chi_square_rejections_at_0_point_05": int(np.sum(
                chi_square_p <= 0.05)),
            "chi_square_rejections_at_0_point_00625": int(np.sum(
                chi_square_p <= 0.00625)),
            "median_bootstrap_pvalue": float(np.median(bootstrap_p)),
        })
    return {"caveat": "Calibration added after observing the independent Aer rerun; it checks inference behavior under the declared exact toy model, not general physical validity.",
            "panels_per_case": panels,
            "bootstraps_per_panel": bootstraps,
            "seed": seed, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=int, default=200)
    parser.add_argument("--bootstraps", type=int, default=499)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    if min(args.panels, args.bootstraps) < 1:
        parser.error("panels and bootstraps must be positive")
    result = study(args.panels, args.bootstraps, args.seed)
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
