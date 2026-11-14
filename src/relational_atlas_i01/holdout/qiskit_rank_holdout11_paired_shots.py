"""Finite-shot Aer confirmation of the stage-10 paired-coherence boundary."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from qiskit_aer import AerSimulator

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout3_shots as previous
from relational_atlas_i01.holdout import qiskit_rank_holdout4b_weaknull as weak
from relational_atlas_i01.holdout import qiskit_rank_holdout7_noise_stress as stage7
from relational_atlas_i01.holdout import qiskit_rank_holdout9_paired_coherence as paired


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout11_paired_shots.json"
CHECKPOINT = PROJECT_ROOT / "results" / "qiskit_rank_holdout11_paired_shots_checkpoint.json"
PROTOCOL = "notes/prevalidation_protocol11_2026-10-04.tex"
SHOTS_PER_LABEL = 131072
SEED = 20261104
BOOTSTRAP_SEED = 20261105
BOOTSTRAPS = 10000
CASES = (
    ("candidate_constraint", 1.0, "skewed", True, 0, "positive"),
    ("candidate_distribution_control", 1.0, "uniform", False, 0, "null"),
    ("candidate_strong_control", 1.0, "skewed", False, 0, "positive"),
    ("subthreshold_constraint", 0.2, "skewed", True, 0, "positive_but_low_power"),
)


def case_specs() -> tuple[tuple, ...]:
    """Frozen condition ordering for the four-test family."""
    return CASES


def exact_case_rows(workers: int) -> dict[str, dict]:
    """Recompute exact Aer probabilities for the two frozen paired angles."""
    output = {}
    templates = stage7.compiled_templates()
    for angle in sorted({case[1] for case in CASES}):
        joint, clock = paired.full_axis_probabilities(angle, workers, templates)
        for name, selected_angle, distribution, physical, tau_index, expected in CASES:
            if selected_angle != angle:
                continue
            allocation = previous.shot_allocation(SHOTS_PER_LABEL, distribution)
            weights = allocation / allocation.sum()
            first = joint[int(physical), tau_index, 0]
            second = joint[int(physical), tau_index, 1]
            delta = (second - first) @ weights
            output[name] = {
                "angle": angle,
                "expected_type": expected,
                "exact_axis_pooled_joint_difference": delta.tolist(),
                "exact_maximum_absolute_joint_difference": float(
                    np.max(np.abs(delta))),
                "exact_pooled_clock_acceptance_first": (
                    clock[int(physical), tau_index, 0] @ weights).tolist(),
                "exact_pooled_clock_acceptance_second": (
                    clock[int(physical), tau_index, 1] @ weights).tolist(),
            }
    return output


def _checkpoint_cases(raw_cases: dict, seed: int) -> None:
    """Persist completed sampling cases so an analysis failure costs no shots."""
    CHECKPOINT.write_text(json.dumps({
        "protocol": PROTOCOL,
        "aer_seed": seed,
        "shots_per_label_if_uniform": SHOTS_PER_LABEL,
        "cases": raw_cases,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _resumable_cases(seed: int) -> dict:
    """Load only a checkpoint compatible with this frozen sampling design."""
    if not CHECKPOINT.exists():
        return {}
    saved = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if (saved.get("protocol") != PROTOCOL or
            saved.get("aer_seed") != seed or
            saved.get("shots_per_label_if_uniform") != SHOTS_PER_LABEL):
        return {}
    cases = saved.get("cases", {})
    return cases if isinstance(cases, dict) else {}


def acquire(workers: int, seed: int) -> dict:
    """Acquire four predeclared raw-count tables with frozen label allocations."""
    templates = stage7.compiled_templates()
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    raw_cases = _resumable_cases(seed)
    for case_index, (name, angle, distribution, physical, tau_index,
                     expected) in enumerate(CASES):
        if name in raw_cases:
            print(f"resumed case={name}", flush=True)
            continue
        base, theta = templates[(int(physical), tau_index)]
        allocation = previous.shot_allocation(SHOTS_PER_LABEL, distribution)
        joint_counts = np.zeros((2, previous.N, previous.LABELS), dtype=int)
        clock_counts = np.zeros_like(joint_counts)
        for label, shots in enumerate(allocation):
            circuits = []
            settings = []
            for shift_index, shift in enumerate(previous.SHIFTS):
                for axis in range(previous.N):
                    phases = np.full(
                        previous.N, shift + 2 * math.pi * label / previous.LABELS)
                    phases[axis] -= stage7.grid.BIAS
                    circuit = paired.paired_rx_after_barriers(
                        stage7.bind(base, theta, phases), angle)
                    circuit.measure_all()
                    circuits.append(circuit)
                    settings.append((shift_index, axis))
            result = simulator.run(
                circuits, shots=int(shots),
                seed_simulator=seed + 1000 * case_index + label,
            ).result()
            if not result.success:
                raise RuntimeError(result.status)
            for index, (shift_index, axis) in enumerate(settings):
                counts = result.get_counts(index)
                joint_counts[shift_index, axis, label] = counts.get("00000000", 0)
                clock_counts[shift_index, axis, label] = sum(
                    value for bits, value in counts.items() if bits[:2] == "00")
        raw_cases[name] = {
            "angle": angle,
            "label_distribution": distribution,
            "physical_P_zero": physical,
            "tau_over_pi": previous.TAUS[tau_index] / math.pi,
            "expected_type": expected,
            "shot_allocation": allocation.tolist(),
            "raw_joint_zero_counts": joint_counts.tolist(),
            "raw_clock_accept_counts": clock_counts.tolist(),
        }
        _checkpoint_cases(raw_cases, seed)
        print(f"completed case={name}", flush=True)
    return raw_cases


def analyze(raw_cases: dict, bootstraps: int, bootstrap_seed: int) -> dict:
    """Apply the stage-6 weak pooled-null statistic to the four-test family."""
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    for name, *_ in CASES:
        raw = raw_cases[name]
        joint = np.asarray(raw["raw_joint_zero_counts"], dtype=int)
        clock = np.asarray(raw["raw_clock_accept_counts"], dtype=int)
        if joint.shape != (2, previous.N, previous.LABELS):
            raise ValueError("unexpected raw shape")
        if np.any(joint > clock):
            raise AssertionError("joint count exceeds clock count")
        test = weak.weak_null_test(
            joint[0], joint[1], np.asarray(raw["shot_allocation"], dtype=int),
            bootstraps, rng)
        rows.append({
            "case": name,
            "expected_type": raw["expected_type"],
            "test": test,
            "observed_axis_pooled_joint_differences": test[
                "axis_differences"],
        })
    threshold = 0.05 / len(rows)
    return {
        "familywise_tests": len(rows),
        "familywise_bonferroni_threshold": threshold,
        "rows": rows,
        "detections": [row["case"] for row in rows if row["test"][
            "weak_null_bootstrap_pvalue"] <= threshold],
        "total_preparations": total_preparations(raw_cases),
    }


def total_preparations(raw_cases: dict) -> int:
    """Count all two-block, three-axis circuit preparations in a raw table."""
    return int(sum(2 * previous.N * sum(raw_cases[name]["shot_allocation"])
                   for name, *_ in CASES))


def circuit_checks() -> dict:
    """Reuse the paired-placement check and verify the frozen resources."""
    checks = paired.circuit_checks()
    if len(CASES) != 4 or SHOTS_PER_LABEL != 131072:
        raise AssertionError("frozen finite-shot design changed")
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
    checks = circuit_checks()
    if args.checks_only:
        print(json.dumps(checks, indent=2))
        return
    exact = exact_case_rows(args.workers)
    raw = acquire(args.workers, args.seed)
    analysis = analyze(raw, args.bootstraps, BOOTSTRAP_SEED)
    output = {
        "protocol": PROTOCOL,
        "parameters": {
            "workers": args.workers,
            "aer_seed": args.seed,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstraps": args.bootstraps,
            "shots_per_label_if_uniform": SHOTS_PER_LABEL,
        },
        "circuit_checks": checks,
        "exact_case_rows": exact,
        "cases": raw,
        "analysis": analysis,
    }
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({
        "pvalues": [(row["case"], row["test"]["weak_null_bootstrap_pvalue"])
                    for row in analysis["rows"]],
        "detections": analysis["detections"],
        "total_preparations": analysis["total_preparations"],
    }, indent=2))


if __name__ == "__main__":
    main()
