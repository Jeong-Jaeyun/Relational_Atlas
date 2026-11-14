"""Post-hoc exact-binomial calibration of explicit-clock weak-null tests.

This audits statistical operating characteristics under the stipulated toy
model, not whether the toy model captures a physical clock.
"""

from __future__ import annotations

import argparse
import json
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout6_clock as clock
from relational_atlas_i01.holdout import qiskit_rank_holdout4b_weaknull as weak


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout6_calibration.json"
SEED = 20261032
THRESHOLD = 0.00625


def study(panels: int, bootstraps: int, seed: int) -> dict:
    if min(panels, bootstraps) < 1:
        raise ValueError("panels and bootstraps must be positive")
    rng = np.random.default_rng(seed)
    exact = clock.exact_joint_probabilities()
    budgets = (1, 2)
    output = []
    for multiple in budgets:
        rows = []
        all_pvalues = np.zeros((panels, 8))
        all_null = []
        row_index = 0
        for distribution in clock.previous.DISTRIBUTIONS:
            allocation = clock.previous.shot_allocation(
                clock.SHOTS_PER_LABEL * multiple, distribution)
            weights = allocation / allocation.sum()
            for physical in (0, 1):
                for tau_index, tau in enumerate(clock.previous.TAUS):
                    first_p = exact[physical, tau_index, 0]
                    second_p = exact[physical, tau_index, 1]
                    exact_delta = (second_p - first_p) @ weights
                    exact_null = bool(np.max(np.abs(exact_delta)) < 1e-12)
                    all_null.append(exact_null)
                    first_counts = rng.binomial(allocation[None, None, :],
                                                first_p[None, :, :],
                                                size=(panels, 3, 7))
                    second_counts = rng.binomial(allocation[None, None, :],
                                                 second_p[None, :, :],
                                                 size=(panels, 3, 7))
                    for panel in range(panels):
                        test = weak.weak_null_test(
                            first_counts[panel], second_counts[panel],
                            allocation, bootstraps, rng)
                        all_pvalues[panel, row_index] = test[
                            "weak_null_bootstrap_pvalue"]
                    p = all_pvalues[:, row_index]
                    rows.append({
                        "label_distribution": distribution,
                        "physical_P_zero": bool(physical),
                        "tau_over_pi": tau / np.pi,
                        "exact_null": exact_null,
                        "exact_max_pooled_joint_difference": float(
                            np.max(np.abs(exact_delta))),
                        "rejections_at_family_threshold": int(np.sum(p <= THRESHOLD)),
                        "rejections_at_nominal_0_point_05": int(np.sum(p <= 0.05)),
                        "median_pvalue": float(np.median(p)),
                    })
                    row_index += 1
        null_columns = np.asarray(all_null, dtype=bool)
        output.append({"budget_multiplier": multiple,
                       "shots_per_label_if_uniform":
                           clock.SHOTS_PER_LABEL * multiple,
                       "rows": rows,
                       "null_familywise_false_rejections": int(np.sum(
                           np.any(all_pvalues[:, null_columns] <= THRESHOLD,
                                  axis=1))),
                       "panels": panels})
    return {"seed": seed, "panels_per_budget": panels,
            "bootstraps_per_test": bootstraps,
            "familywise_threshold": THRESHOLD,
            "results": output,
            "caveat": "Post-hoc exact-binomial method calibration; not an independent circuit experiment or a test of the physical model. Monte Carlo error is substantial at small rejection rates."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=int, default=200)
    parser.add_argument("--bootstraps", type=int, default=999)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = study(args.panels, args.bootstraps, args.seed)
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
