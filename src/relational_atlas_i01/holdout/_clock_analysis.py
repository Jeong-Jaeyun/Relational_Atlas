"""Shared contrast metrics for explicit-clock prevalidation experiments."""

from __future__ import annotations

import math

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout3_shots as previous
from relational_atlas_i01.holdout import qiskit_rank_holdout6_clock as stage6


CONDITIONS = (
    ("skewed", False, 0, "strong_signal"),
    ("skewed", False, 1, "weak_signal"),
    ("uniform", False, 0, "distribution_null_tau_0"),
    ("uniform", False, 1, "distribution_null_tau_half_pi"),
    ("skewed", True, 0, "constraint_null_tau_0"),
    ("skewed", True, 1, "constraint_null_tau_half_pi"),
)
SEPARATION_RATIO = 0.1


def condition_rows(joint: np.ndarray, clock: np.ndarray) -> list[dict]:
    """Calculate pooled contrasts and the frozen stage-6 budget proxy."""
    rows = []
    for distribution, physical, tau_index, role in CONDITIONS:
        physical_index = int(physical)
        shots = previous.shot_allocation(stage6.SHOTS_PER_LABEL, distribution)
        weights = shots / shots.sum()
        first = joint[physical_index, tau_index, 0]
        second = joint[physical_index, tau_index, 1]
        pooled_first = first @ weights
        pooled_second = second @ weights
        delta = pooled_second - pooled_first
        average = (pooled_first + pooled_second) / 2
        noncentrality = float(np.sum(
            delta ** 2 * shots.sum() / (2 * average * (1 - average))))
        clock_first = clock[physical_index, tau_index, 0] @ weights
        clock_second = clock[physical_index, tau_index, 1] @ weights
        rows.append({
            "role": role,
            "label_distribution": distribution,
            "physical_P_zero": bool(physical),
            "tau_over_pi": previous.TAUS[tau_index] / math.pi,
            "axis_pooled_joint_difference": delta.tolist(),
            "maximum_absolute_joint_difference": float(np.max(np.abs(delta))),
            "pooled_joint_probability_first": pooled_first.tolist(),
            "pooled_joint_probability_second": pooled_second.tolist(),
            "axis_pooled_clock_difference": (clock_second - clock_first).tolist(),
            "maximum_absolute_clock_difference": float(
                np.max(np.abs(clock_second - clock_first))),
            "stage6_budget_approx_noncentrality": noncentrality,
        })
    return rows


def scenario_summary(rows: list[dict], ideal_rows: list[dict]) -> dict:
    """Classify null leakage against the predeclared strong-signal scale."""
    by_role = {row["role"]: row for row in rows}
    ideal_by_role = {row["role"]: row for row in ideal_rows}
    strong = by_role["strong_signal"]["maximum_absolute_joint_difference"]
    weak = by_role["weak_signal"]["maximum_absolute_joint_difference"]
    nulls = [by_role["distribution_null_tau_0"][
                 "maximum_absolute_joint_difference"],
             by_role["distribution_null_tau_half_pi"][
                 "maximum_absolute_joint_difference"],
             by_role["constraint_null_tau_0"][
                 "maximum_absolute_joint_difference"],
             by_role["constraint_null_tau_half_pi"][
                 "maximum_absolute_joint_difference"]]
    max_null = max(nulls)
    ratio = max_null / strong if strong else math.inf
    if max_null <= 1e-10:
        classification = "all_four_nulls_numerically_preserved"
    elif ratio <= SEPARATION_RATIO:
        classification = "null_leakage_below_predeclared_tenth_of_strong_signal"
    elif ratio <= 1:
        classification = "null_leakage_is_not_negligible_against_strong_signal"
    else:
        classification = "null_leakage_matches_or_exceeds_strong_signal"
    return {
        "strong_signal_scale": strong,
        "weak_signal_scale": weak,
        "maximum_null_scale": max_null,
        "maximum_null_to_strong_ratio": ratio,
        "strong_signal_fraction_of_ideal": (
            strong / ideal_by_role["strong_signal"][
                "maximum_absolute_joint_difference"]),
        "weak_signal_fraction_of_ideal": (
            weak / ideal_by_role["weak_signal"][
                "maximum_absolute_joint_difference"]),
        "classification": classification,
        "predeclared_negligible_leakage_ratio": SEPARATION_RATIO,
    }

