"""Finite-shot Aer confirmation of the paired-support transfer classes."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from qiskit_aer import AerSimulator

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout3_shots as previous
from relational_atlas_i01.holdout import qiskit_rank_holdout4b_weaknull as weak
from relational_atlas_i01.holdout import qiskit_rank_holdout6_clock as stage6
from relational_atlas_i01.holdout import qiskit_rank_holdout15_dimension_transfer as stage15


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout17_transfer_shots.json"
CHECKPOINT = PROJECT_ROOT / "results" / "qiskit_rank_holdout17_transfer_shots_checkpoint.json"
PROTOCOL = "notes/prevalidation_protocol17_2026-10-04.tex"
N, LABELS = 3, 7
ANGLE = math.pi / 2
SHOTS_PER_LABEL = 131072
SEED = 20261117
BOOTSTRAP_SEED = 20261118
BOOTSTRAPS = 10000
CASES = (
    ("same_support", 0, 0, "positive"),
    ("conjugate_partner", 0, 1, "null"),
    ("different_pair_same_sign", 0, 2, "positive"),
    ("different_pair_opposite_sign", 0, 3, "positive"),
)


def case_specs() -> tuple[tuple, ...]:
    return CASES


def _allocation() -> np.ndarray:
    return previous.shot_allocation(SHOTS_PER_LABEL, "skewed")


def _load_checkpoint(seed: int) -> dict:
    if not CHECKPOINT.exists():
        return {}
    saved = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if (saved.get("protocol") != PROTOCOL or saved.get("aer_seed") != seed or
            saved.get("shots_per_label") != SHOTS_PER_LABEL):
        return {}
    cases = saved.get("cases", {})
    return cases if isinstance(cases, dict) else {}


def _save_checkpoint(cases: dict, seed: int) -> None:
    CHECKPOINT.write_text(json.dumps({
        "protocol": PROTOCOL,
        "aer_seed": seed,
        "shots_per_label": SHOTS_PER_LABEL,
        "cases": cases,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def exact_cases(workers: int) -> dict:
    """Compute independent exact target contrasts before reading shot counts."""
    weights = stage15.frozen_label_weights(LABELS)
    allocation = _allocation()
    output = {}
    for name, source, return_, expected in CASES:
        joint, clock = stage15.probabilities(N, source, return_, workers, angle=ANGLE)
        first, second = joint[0] @ weights, joint[1] @ weights
        delta = second - first
        average = (first + second) / 2
        noncentrality = float(np.sum(
            delta ** 2 * allocation.sum() / (2 * average * (1 - average))))
        output[name] = {
            "expected_type": expected,
            "exact_axis_pooled_joint_difference": delta.tolist(),
            "exact_maximum_absolute_joint_difference": float(np.max(np.abs(delta))),
            "stage6_budget_approx_noncentrality": noncentrality,
            "maximum_clock_acceptance_error": float(np.max(np.abs(clock - 0.25))),
        }
    return output


def acquire(workers: int, seed: int) -> dict:
    """Acquire each predeclared source-return raw table with resumable storage."""
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    base, theta, _ = stage15.template(N)
    allocation = _allocation()
    cases = _load_checkpoint(seed)
    for case_index, (name, source, return_, expected) in enumerate(CASES):
        if name in cases:
            print(f"resumed case={name}", flush=True)
            continue
        joint_counts = np.zeros((2, N, LABELS), dtype=int)
        clock_counts = np.zeros_like(joint_counts)
        for label, shots in enumerate(allocation):
            circuits, settings = [], []
            for block, shift in enumerate(previous.SHIFTS):
                for axis in range(N):
                    phases = np.full(N, shift + 2 * math.pi * label / LABELS)
                    phases[axis] -= stage15.grid.BIAS
                    circuit = base.assign_parameters({theta[i]: float(phases[i]) for i in range(N)})
                    circuit = stage15.insert_paired_rx(circuit, source, return_, ANGLE)
                    circuit.measure_all()
                    circuits.append(circuit)
                    settings.append((block, axis))
            result = simulator.run(circuits, shots=int(shots),
                                   seed_simulator=seed + 1000 * case_index + label).result()
            if not result.success:
                raise RuntimeError(result.status)
            for index, (block, axis) in enumerate(settings):
                counts = result.get_counts(index)
                joint_counts[block, axis, label] = counts.get("00000000", 0)
                clock_counts[block, axis, label] = sum(
                    value for bits, value in counts.items() if bits[:2] == "00")
        cases[name] = {
            "source_qubit": source,
            "return_qubit": return_,
            "expected_type": expected,
            "shot_allocation": allocation.tolist(),
            "raw_joint_zero_counts": joint_counts.tolist(),
            "raw_clock_accept_counts": clock_counts.tolist(),
        }
        _save_checkpoint(cases, seed)
        print(f"completed case={name}", flush=True)
    return cases


def total_preparations(cases: dict) -> int:
    return int(sum(2 * N * sum(cases[name]["shot_allocation"])
                   for name, *_ in CASES))


def analyze(cases: dict, bootstraps: int, bootstrap_seed: int) -> dict:
    """Apply the bounded weak-null test to the four frozen transfer rows."""
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    for name, *_ in CASES:
        raw = cases[name]
        joint = np.asarray(raw["raw_joint_zero_counts"], dtype=int)
        clock = np.asarray(raw["raw_clock_accept_counts"], dtype=int)
        if joint.shape != (2, N, LABELS) or np.any(joint > clock):
            raise AssertionError("invalid raw count table")
        test = weak.weak_null_test(joint[0], joint[1],
                                   np.asarray(raw["shot_allocation"], dtype=int),
                                   bootstraps, rng)
        rows.append({"case": name, "expected_type": raw["expected_type"],
                     "test": test,
                     "observed_axis_pooled_joint_differences": test["axis_differences"]})
    threshold = 0.05 / len(rows)
    return {
        "familywise_tests": len(rows),
        "familywise_bonferroni_threshold": threshold,
        "rows": rows,
        "detections": [row["case"] for row in rows
                       if row["test"]["weak_null_bootstrap_pvalue"] <= threshold],
        "total_preparations": total_preparations(cases),
    }


def circuit_checks() -> dict:
    checks = stage15.circuit_checks()
    if len(CASES) != 4 or SHOTS_PER_LABEL != 131072:
        raise AssertionError("frozen finite-shot family changed")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.bootstraps) < 1:
        parser.error("workers and bootstraps must be positive")
    if args.checks_only:
        print(json.dumps(circuit_checks(), indent=2))
        return
    exact = exact_cases(args.workers)
    cases = acquire(args.workers, args.seed)
    analysis = analyze(cases, args.bootstraps, BOOTSTRAP_SEED)
    output = {
        "protocol": PROTOCOL,
        "parameters": {"workers": args.workers, "aer_seed": args.seed,
                       "bootstrap_seed": BOOTSTRAP_SEED, "bootstraps": args.bootstraps,
                       "angle_radians": ANGLE, "shots_per_label_if_uniform": SHOTS_PER_LABEL},
        "circuit_checks": circuit_checks(), "exact_cases": exact,
        "cases": cases, "analysis": analysis,
    }
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pvalues": [(row["case"], row["test"]["weak_null_bootstrap_pvalue"])
                                  for row in analysis["rows"]],
                      "detections": analysis["detections"],
                      "total_preparations": analysis["total_preparations"]}, indent=2))


if __name__ == "__main__":
    main()
