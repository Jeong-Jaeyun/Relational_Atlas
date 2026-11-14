"""Exact pre-Aer pilot for random phase-label allocation versus block drift."""

from __future__ import annotations

import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout2_grid as grid


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout5_power.json"
N = 3
LABELS = 7
TOTAL_SHOTS_PER_AXIS_BLOCK = LABELS * 32768


def study() -> dict:
    rows = []
    uniform = grid.label_weights(LABELS, "uniform")
    skewed = grid.label_weights(LABELS, "skewed")
    angles = grid.angles_for_grid(N, LABELS)
    for physical in (False, True):
        entry = grid.model(N, 0.45, physical, "nonlinear")
        for tau in (0.0, math.pi / 2):
            label_probability, _ = grid.probability_gradient(
                entry, "mismatched", tau, angles)
            mean_uniform = label_probability @ uniform
            mean_skewed = label_probability @ skewed
            drift_difference = mean_skewed - mean_uniform
            fixed_label_variance = np.sum(
                uniform[None, :] * label_probability *
                (1 - label_probability), axis=1) / TOTAL_SHOTS_PER_AXIS_BLOCK
            iid_random_label_variance = (
                mean_uniform * (1 - mean_uniform) / TOTAL_SHOTS_PER_AXIS_BLOCK)
            variance_ratio = iid_random_label_variance / fixed_label_variance
            drift_noncentrality = float(np.sum(
                drift_difference ** 2 /
                ((mean_uniform * (1 - mean_uniform) +
                  mean_skewed * (1 - mean_skewed)) /
                 TOTAL_SHOTS_PER_AXIS_BLOCK)))
            rows.append({
                "physical_P_zero": physical, "tau_over_pi": tau / math.pi,
                "uniform_to_skewed_max_pooled_probability_difference":
                    float(np.max(np.abs(drift_difference))),
                "approx_drift_noncentrality_at_229376_shots_per_axis_block":
                    drift_noncentrality,
                "iid_label_to_fixed_label_variance_ratio_range":
                    [float(np.min(variance_ratio)),
                     float(np.max(variance_ratio))],
                "max_label_probability_range": float(np.max(
                    np.max(label_probability, axis=1) -
                    np.min(label_probability, axis=1))),
            })
    return {"caveat": "Exact-model pilot only; no new Aer counts observed.",
            "n": N, "labels": LABELS,
            "total_shots_per_axis_block": TOTAL_SHOTS_PER_AXIS_BLOCK,
            "rows": rows}


def main() -> None:
    result = study()
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
