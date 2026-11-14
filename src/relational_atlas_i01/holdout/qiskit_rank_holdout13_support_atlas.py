"""Exact Aer support atlas for the paired transverse leakage mechanism."""

from __future__ import annotations

import argparse
import json

import numpy as np

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout6_clock as stage6
from relational_atlas_i01.holdout import qiskit_rank_holdout7_noise_stress as stage7
from relational_atlas_i01.holdout import qiskit_rank_holdout12_intervention_atlas as atlas
from relational_atlas_i01.holdout._clock_analysis import condition_rows, scenario_summary
from relational_atlas_i01.holdout._clock_postselection import CLOCK_ACCEPTANCE


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout13_support_atlas.json"
PROTOCOL = "notes/prevalidation_protocol13_2026-10-04.tex"
SYSTEM_QUBITS = 6


def scenario_specs() -> tuple[tuple, ...]:
    """Return all same-support rows and all q0-to-other-support controls."""
    same = tuple((f"same_q{qubit}", "x", qubit, "x", qubit, -1.0)
                 for qubit in range(SYSTEM_QUBITS))
    cross = tuple((f"q0_to_q{qubit}", "x", 0, "x", qubit, -1.0)
                  for qubit in range(1, SYSTEM_QUBITS))
    return same + cross


SCENARIOS = scenario_specs()


def circuit_checks() -> dict:
    """Verify exact barrier placement and qubit support for every row."""
    base, _ = stage7.compiled_templates()[(0, 0)]
    details = {}
    for spec in SCENARIOS:
        circuit = atlas.insert_intervention(base, spec, 0.1)
        data = list(circuit.data)
        barriers = [i for i, item in enumerate(data) if item.operation.name == "barrier"]
        rotations = [i for i, item in enumerate(data) if item.operation.name == "rx"]
        if rotations != [barriers[0] + 1, barriers[1] + 1]:
            raise AssertionError(f"invalid placement for {spec[0]}")
        supports = [circuit.find_bit(data[index].qubits[0]).index for index in rotations]
        if supports != [spec[2], spec[4]]:
            raise AssertionError(f"invalid support for {spec[0]}")
        details[spec[0]] = {"rotation_indices": rotations, "supports": supports}
    return details


def _constraint(rows: list[dict]) -> dict:
    return next(row for row in rows if row["role"] == "constraint_null_tau_0")


def study(workers: int) -> dict:
    """Evaluate the frozen same- and cross-support paired-intervention atlas."""
    templates = stage7.compiled_templates()
    ideal_joint = stage6.exact_joint_probabilities()
    ideal_rows = condition_rows(ideal_joint, np.full_like(ideal_joint, CLOCK_ACCEPTANCE))
    scenarios = []
    for spec in SCENARIOS:
        joint, clock = atlas.full_axis_probabilities(spec, workers, templates)
        rows = condition_rows(joint, clock)
        constraint = _constraint(rows)
        scenarios.append({
            "name": spec[0],
            "first_qubit": spec[2],
            "second_qubit": spec[4],
            "rows": rows,
            "summary": scenario_summary(rows, ideal_rows),
            "constraint_tau_0": {
                "maximum_absolute_joint_difference": constraint["maximum_absolute_joint_difference"],
                "axis_pooled_joint_difference": constraint["axis_pooled_joint_difference"],
            },
        })
    same_values = [row["constraint_tau_0"]["maximum_absolute_joint_difference"]
                   for row in scenarios if row["first_qubit"] == row["second_qubit"]]
    cross_values = [row["constraint_tau_0"]["maximum_absolute_joint_difference"]
                    for row in scenarios if row["first_qubit"] != row["second_qubit"]]
    return {
        "protocol": PROTOCOL,
        "parameters": {"workers": workers, "angle_radians": atlas.ANGLE,
                       "system_qubits": SYSTEM_QUBITS,
                       "all_axes_explicitly_evaluated": True},
        "circuit_checks": circuit_checks(),
        "scenarios": scenarios,
        "same_support_minimum_constraint_contrast": min(same_values),
        "same_support_maximum_constraint_contrast": max(same_values),
        "cross_support_maximum_constraint_contrast": max(cross_values),
        "caveat": "Exact statevector support map of coherent interventions; not finite-shot evidence.",
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
    print(json.dumps({
        "same_support_range": [output["same_support_minimum_constraint_contrast"],
                               output["same_support_maximum_constraint_contrast"]],
        "cross_support_maximum": output["cross_support_maximum_constraint_contrast"],
        "rows": [{"name": row["name"], **row["constraint_tau_0"]}
                 for row in output["scenarios"]],
    }, indent=2))


if __name__ == "__main__":
    main()
