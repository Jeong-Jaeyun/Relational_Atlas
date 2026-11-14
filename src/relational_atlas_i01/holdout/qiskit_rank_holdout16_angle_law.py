"""Exact Aer angle-response test for the four paired-support transfer classes."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout15_dimension_transfer as stage15


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout16_angle_law.json"
CHECKPOINT = PROJECT_ROOT / "results" / "qiskit_rank_holdout16_angle_law_checkpoint.json"
PROTOCOL = "notes/prevalidation_protocol16_2026-10-04.tex"
N = 3
ANGLES = (0.1, 0.2, 0.5, 0.8, 1.0, 1.2, math.pi / 2, 2.0, math.pi)
SUPPORTS = stage15.SUPPORTS


def scenario_specs() -> tuple[tuple[str, float, int, int], ...]:
    """Freeze angle-major ordering of all support-class responses."""
    return tuple((name, angle, source, return_)
                 for angle in ANGLES for name, source, return_ in SUPPORTS)


SCENARIOS = scenario_specs()


def _load_checkpoint() -> list[dict]:
    if not CHECKPOINT.exists():
        return []
    saved = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    expected = [[name, angle, source, return_] for name, angle, source, return_ in SCENARIOS]
    rows = saved.get("rows", [])
    if saved.get("protocol") != PROTOCOL or saved.get("scenarios") != expected:
        return []
    return rows if isinstance(rows, list) else []


def _save_checkpoint(rows: list[dict]) -> None:
    CHECKPOINT.write_text(json.dumps({
        "protocol": PROTOCOL,
        "scenarios": [[name, angle, source, return_]
                      for name, angle, source, return_ in SCENARIOS],
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def circuit_checks() -> dict:
    """Reuse generalized-clock placement checks and validate angular endpoints."""
    if not np.isclose(ANGLES[-1], math.pi) or len(SCENARIOS) != 36:
        raise AssertionError("frozen angular grid changed")
    generic = stage15.circuit_checks()
    return {"scenario_count": len(SCENARIOS), "generic_clock_checks": generic}


def _summaries(rows: list[dict]) -> list[dict]:
    output = []
    for name, _, _ in SUPPORTS:
        selected = [row for row in rows if row["support_class"] == name]
        scaled = [row["maximum_absolute_joint_difference"] / row["sin_squared"]
                  for row in selected if row["sin_squared"] > 1e-12]
        endpoint = next(row for row in selected if np.isclose(row["angle_radians"], math.pi))
        if max(scaled, default=0) <= 1e-12:
            spread = None
        else:
            scale = float(np.median(scaled))
            spread = float(max(abs(value - scale) for value in scaled) / scale)
        output.append({
            "support_class": name,
            "nonendpoint_sin_squared_scale_range": [float(min(scaled)), float(max(scaled))],
            "relative_spread_about_median_scale": spread,
            "pi_endpoint_maximum_absolute_joint_difference": endpoint[
                "maximum_absolute_joint_difference"],
        })
    return output


def study(workers: int) -> dict:
    """Evaluate and checkpoint the complete angle-by-support exact-Aer grid."""
    if workers < 1:
        raise ValueError("workers must be positive")
    rows = _load_checkpoint()
    completed = {(row["support_class"], row["angle_radians"]) for row in rows}
    prepared = stage15.template(N)
    weights = stage15.frozen_label_weights(2 * N + 1)
    for name, angle, source, return_ in SCENARIOS:
        if (name, angle) in completed:
            print(f"resumed row={name}, angle={angle:.12g}", flush=True)
            continue
        joint, acceptance = stage15.probabilities(
            N, source, return_, workers, prepared, angle)
        delta = (joint[1] - joint[0]) @ weights
        rows.append({
            "support_class": name,
            "angle_radians": angle,
            "source_qubit": source,
            "return_qubit": return_,
            "sin_squared": float(math.sin(angle) ** 2),
            "axis_pooled_joint_difference": delta.tolist(),
            "maximum_absolute_joint_difference": float(np.max(np.abs(delta))),
            "maximum_clock_acceptance_error": float(np.max(np.abs(
                acceptance - 1 / (1 << prepared[2])))),
        })
        _save_checkpoint(rows)
        print(f"completed row={name}, angle={angle:.12g}", flush=True)
    return {
        "protocol": PROTOCOL,
        "parameters": {"workers": workers, "n": N, "angles_radians": list(ANGLES),
                       "support_classes": [name for name, _, _ in SUPPORTS],
                       "all_axes_explicitly_evaluated": True},
        "circuit_checks": circuit_checks(),
        "rows": rows,
        "support_class_summaries": _summaries(rows),
        "caveat": "Exact Aer angle map for a fixed n=3 toy circuit; scale diagnostics are descriptive, not fitted hardware parameters.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.checks_only:
        print(json.dumps(circuit_checks(), indent=2))
        return
    output = study(args.workers)
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["support_class_summaries"], indent=2))


if __name__ == "__main__":
    main()
