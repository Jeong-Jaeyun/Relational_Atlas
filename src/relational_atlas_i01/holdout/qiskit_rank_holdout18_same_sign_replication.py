"""Independent high-budget Aer replication of the borderline same-sign transfer."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from qiskit_aer import AerSimulator

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout3_shots as previous
from relational_atlas_i01.holdout import qiskit_rank_holdout4b_weaknull as weak
from relational_atlas_i01.holdout import qiskit_rank_holdout15_dimension_transfer as stage15


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout18_same_sign_replication.json"
CHECKPOINT = PROJECT_ROOT / "results" / "qiskit_rank_holdout18_same_sign_replication_checkpoint.json"
PROTOCOL = "notes/prevalidation_protocol18_2026-10-04.tex"
N, LABELS = 3, 7
SOURCE, RETURN = 0, 2
ANGLE = math.pi / 2
SHOTS_PER_LABEL = 262144
SEED = 20261119
BOOTSTRAP_SEED = 20261120
BOOTSTRAPS = 10000


def allocation() -> np.ndarray:
    total = LABELS * SHOTS_PER_LABEL
    target = stage15.grid.label_weights(LABELS, "skewed") * total
    counts = np.floor(target).astype(int)
    counts[np.argsort(-(target - counts), kind="stable")[:total - int(counts.sum())]] += 1
    return counts


def circuit_checks() -> dict:
    if SHOTS_PER_LABEL != 2 * 131072:
        raise AssertionError("replication budget changed")
    return stage15.circuit_checks()


def _load_checkpoint(seed: int) -> dict | None:
    if not CHECKPOINT.exists():
        return None
    saved = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if (saved.get("protocol") != PROTOCOL or saved.get("aer_seed") != seed or
            saved.get("shots_per_label") != SHOTS_PER_LABEL):
        return None
    return saved.get("raw") if isinstance(saved.get("raw"), dict) else None


def _save_checkpoint(raw: dict, seed: int) -> None:
    CHECKPOINT.write_text(json.dumps({
        "protocol": PROTOCOL, "aer_seed": seed, "shots_per_label": SHOTS_PER_LABEL,
        "raw": raw,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def exact_row(workers: int) -> dict:
    weights = allocation() / allocation().sum()
    joint, clock = stage15.probabilities(N, SOURCE, RETURN, workers, angle=ANGLE)
    first, second = joint[0] @ weights, joint[1] @ weights
    delta = second - first
    average = (first + second) / 2
    return {
        "exact_axis_pooled_joint_difference": delta.tolist(),
        "exact_maximum_absolute_joint_difference": float(np.max(np.abs(delta))),
        "approximate_noncentrality": float(np.sum(
            delta ** 2 * allocation().sum() / (2 * average * (1 - average)))),
        "maximum_clock_acceptance_error": float(np.max(np.abs(clock - 0.25))),
    }


def acquire(workers: int, seed: int) -> dict:
    """Acquire the one predeclared replication table, or reuse its checkpoint."""
    existing = _load_checkpoint(seed)
    if existing is not None:
        print("resumed replication", flush=True)
        return existing
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    base, theta, _ = stage15.template(N)
    shots_by_label = allocation()
    joint_counts = np.zeros((2, N, LABELS), dtype=int)
    clock_counts = np.zeros_like(joint_counts)
    for label, shots in enumerate(shots_by_label):
        circuits, settings = [], []
        for block, shift in enumerate(previous.SHIFTS):
            for axis in range(N):
                phases = np.full(N, shift + 2 * math.pi * label / LABELS)
                phases[axis] -= stage15.grid.BIAS
                circuit = base.assign_parameters({theta[i]: float(phases[i]) for i in range(N)})
                circuit = stage15.insert_paired_rx(circuit, SOURCE, RETURN, ANGLE)
                circuit.measure_all()
                circuits.append(circuit)
                settings.append((block, axis))
        result = simulator.run(circuits, shots=int(shots),
                               seed_simulator=seed + label).result()
        if not result.success:
            raise RuntimeError(result.status)
        for index, (block, axis) in enumerate(settings):
            counts = result.get_counts(index)
            joint_counts[block, axis, label] = counts.get("00000000", 0)
            clock_counts[block, axis, label] = sum(
                value for bits, value in counts.items() if bits[:2] == "00")
    raw = {"source_qubit": SOURCE, "return_qubit": RETURN,
           "shot_allocation": shots_by_label.tolist(),
           "raw_joint_zero_counts": joint_counts.tolist(),
           "raw_clock_accept_counts": clock_counts.tolist()}
    _save_checkpoint(raw, seed)
    print("completed replication", flush=True)
    return raw


def analyze(raw: dict, bootstraps: int) -> dict:
    joint = np.asarray(raw["raw_joint_zero_counts"], dtype=int)
    clock = np.asarray(raw["raw_clock_accept_counts"], dtype=int)
    if joint.shape != (2, N, LABELS) or np.any(joint > clock):
        raise AssertionError("invalid raw count table")
    test = weak.weak_null_test(joint[0], joint[1],
                               np.asarray(raw["shot_allocation"], dtype=int),
                               bootstraps, np.random.default_rng(BOOTSTRAP_SEED))
    return {"single_test_threshold": 0.05, "test": test,
            "detected": test["weak_null_bootstrap_pvalue"] <= 0.05,
            "total_preparations": int(2 * N * sum(raw["shot_allocation"]))}


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
    exact = exact_row(args.workers)
    raw = acquire(args.workers, args.seed)
    analysis = analyze(raw, args.bootstraps)
    output = {"protocol": PROTOCOL,
              "parameters": {"workers": args.workers, "aer_seed": args.seed,
                             "bootstrap_seed": BOOTSTRAP_SEED, "bootstraps": args.bootstraps,
                             "angle_radians": ANGLE, "shots_per_label_if_uniform": SHOTS_PER_LABEL},
              "circuit_checks": circuit_checks(), "exact": exact, "raw": raw,
              "analysis": analysis}
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pvalue": analysis["test"]["weak_null_bootstrap_pvalue"],
                      "detected": analysis["detected"],
                      "total_preparations": analysis["total_preparations"]}, indent=2))


if __name__ == "__main__":
    main()
