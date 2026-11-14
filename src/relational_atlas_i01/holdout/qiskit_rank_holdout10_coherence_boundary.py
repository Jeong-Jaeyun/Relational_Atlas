"""Stage-10 paired-coherence boundary scan at the frozen stage-6 budget."""

from __future__ import annotations

import argparse
import json

import numpy as np

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout6_clock as stage6
from relational_atlas_i01.holdout import qiskit_rank_holdout7_noise_stress as stage7
from relational_atlas_i01.holdout import qiskit_rank_holdout9_paired_coherence as paired
from relational_atlas_i01.holdout._clock_analysis import (
    condition_rows,
    scenario_summary,
)
from relational_atlas_i01.holdout._clock_postselection import CLOCK_ACCEPTANCE


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout10_coherence_boundary.json"
PROTOCOL = "notes/prevalidation_protocol10_2026-10-04.tex"
ANGLES = (0.1, 0.2, 0.3, 0.5, 0.8, 1.0)
LEAKAGE_FRACTION_THRESHOLD = 0.1
APPROXIMATE_NONCENTRALITY_SCREEN = 10.0


def boundary_summary(rows: list[dict], ideal_rows: list[dict]) -> dict:
    """Extract the P=0 tau=0 boundary diagnostics from a full contrast table."""
    base = scenario_summary(rows, ideal_rows)
    constraint = next(row for row in rows
                      if row["role"] == "constraint_null_tau_0")
    return {
        **base,
        "constraint_tau_0_scale": constraint[
            "maximum_absolute_joint_difference"],
        "constraint_tau_0_stage6_budget_approx_noncentrality": constraint[
            "stage6_budget_approx_noncentrality"],
        "leakage_fraction_at_least_0_point_1": (
            base["maximum_null_to_strong_ratio"] >=
            LEAKAGE_FRACTION_THRESHOLD),
        "constraint_stage6_budget_screen_at_lambda_10": (
            constraint["stage6_budget_approx_noncentrality"] >=
            APPROXIMATE_NONCENTRALITY_SCREEN),
    }


def first_crossing(scenarios: list[dict], key: str) -> float | None:
    """Return the first frozen angle satisfying a predeclared Boolean screen."""
    for scenario in scenarios:
        if scenario["summary"].get(key, False):
            return scenario["intervention_angle"]
    return None


def study(workers: int) -> dict:
    """Map the exact paired-coherence leakage boundary on the frozen grid."""
    templates = stage7.compiled_templates()
    ideal_joint = stage6.exact_joint_probabilities()
    ideal_rows = condition_rows(ideal_joint,
                                np.full_like(ideal_joint, CLOCK_ACCEPTANCE))
    scenarios = []
    for angle in ANGLES:
        joint, clock = paired.full_axis_probabilities(angle, workers, templates)
        rows = condition_rows(joint, clock)
        scenarios.append({
            "intervention_angle": angle,
            "rows": rows,
            "summary": boundary_summary(rows, ideal_rows),
        })
    return {
        "protocol": PROTOCOL,
        "parameters": {
            "workers": workers,
            "angles_radians": list(ANGLES),
            "paired_operation": (
                "Rx(angle) after preparation and Rx(-angle) before reference "
                "projection on system qubit 0."
            ),
            "all_axes_explicitly_evaluated": True,
            "leakage_fraction_threshold": LEAKAGE_FRACTION_THRESHOLD,
            "approximate_noncentrality_screen": (
                APPROXIMATE_NONCENTRALITY_SCREEN),
        },
        "circuit_checks": paired.circuit_checks(),
        "scenarios": scenarios,
        "first_angle_at_leakage_fraction_0_point_1": first_crossing(
            scenarios, "leakage_fraction_at_least_0_point_1"),
        "first_angle_at_stage6_budget_lambda_10_screen": first_crossing(
            scenarios, "constraint_stage6_budget_screen_at_lambda_10"),
        "caveat": (
            "This is an exact Aer boundary map for a deliberately coherent "
            "intervention. Lambda=10 is a prespecified design screen, not an "
            "exact bootstrap threshold or a hardware error tolerance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    checks = paired.circuit_checks()
    if args.checks_only:
        print(json.dumps(checks, indent=2))
        return
    output = study(args.workers)
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({
        "first_angle_at_leakage_fraction_0_point_1":
            output["first_angle_at_leakage_fraction_0_point_1"],
        "first_angle_at_stage6_budget_lambda_10_screen":
            output["first_angle_at_stage6_budget_lambda_10_screen"],
        "rows": [
            {"angle": scenario["intervention_angle"], **scenario["summary"]}
            for scenario in output["scenarios"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
