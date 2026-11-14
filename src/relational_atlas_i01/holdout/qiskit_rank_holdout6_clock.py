"""Predeclared explicit-clock postselection test for I01 stage 6."""

from __future__ import annotations

import argparse
import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from relational_atlas_i01.holdout._clock_postselection import (
    CLOCK_ACCEPTANCE, CLOCK_QUBITS, bind, exact_joint_probabilities, template,
)
from relational_atlas_i01.holdout import qiskit_rank_holdout2_grid as grid
from relational_atlas_i01.holdout import qiskit_rank_holdout3_shots as previous
from relational_atlas_i01.holdout import qiskit_rank_holdout4b_weaknull as weak


PROJECT = PROJECT_ROOT
RESULT = PROJECT / "results" / "qiskit_rank_holdout6_clock.json"
PROTOCOL = "notes/prevalidation_protocol6_2026-10-04.tex"
N = previous.N
LABELS = previous.LABELS
SHOTS_PER_LABEL = 131072
SEED = 20261027
BOOTSTRAP_SEED = 20261028
BOOTSTRAPS = 10000


def structural_pilot() -> dict:
    rows = []
    expected = exact_joint_probabilities()
    for distribution in previous.DISTRIBUTIONS:
        allocation = previous.shot_allocation(SHOTS_PER_LABEL, distribution)
        weights = allocation / allocation.sum()
        for physical in (0, 1):
            entry = grid.model(N, previous.BETA, bool(physical), previous.SPECTRUM)
            for tau_index, tau in enumerate(previous.TAUS):
                angles = grid.angles_for_grid(N, LABELS)
                probability, gradient = grid.probability_gradient(
                    entry, "mismatched", tau, angles)
                probability *= CLOCK_ACCEPTANCE
                gradient *= CLOCK_ACCEPTANCE
                kept = np.einsum(
                    "itj,itk,it->jk", gradient, gradient,
                    weights[None, :] / (N * probability * (1 - probability)))
                pooled_probability = probability @ weights
                pooled_gradient = np.einsum("itj,t->ij", gradient, weights)
                erased = np.einsum(
                    "ij,ik,i->jk", pooled_gradient, pooled_gradient,
                    1 / (N * pooled_probability * (1 - pooled_probability)))
                first = expected[physical, tau_index, 0] @ weights
                second = expected[physical, tau_index, 1] @ weights
                noncentrality = float(np.sum(
                    (second - first) ** 2 * allocation.sum() /
                    (2 * first * (1 - first))))
                rows.append({
                    "label_distribution": distribution,
                    "physical_P_zero": bool(physical),
                    "tau_over_pi": tau / math.pi,
                    "label_kept_fisher": grid.fisher_summary(kept),
                    "label_erased_fisher": grid.fisher_summary(erased),
                    "exact_max_pooled_joint_probability_difference": float(
                        np.max(np.abs(second - first))),
                    "approx_noncentrality": noncentrality,
                })
    return {"rows": rows, "clock_acceptance_exact": CLOCK_ACCEPTANCE,
            "caveat": "Exact-model pre-run pilot; noncentrality is approximate."}


def circuit_checks() -> dict:
    rows = []
    specs = ((False, 0, 0, 0, 1), (False, 1, 1, 2, 3),
             (True, 0, 1, 1, 5), (True, 1, 0, 2, 6))
    density = AerSimulator(method="density_matrix", max_parallel_threads=2,
                           max_parallel_experiments=1)
    for physical, tau_index, shift_index, axis, label in specs:
        entry = grid.model(N, previous.BETA, physical, previous.SPECTRUM)
        tau = previous.TAUS[tau_index]
        shift = previous.SHIFTS[shift_index]
        angles = np.full(N, shift + 2 * math.pi * label / LABELS)
        angles[axis] -= grid.BIAS
        circuit, theta = template(entry, tau)
        circuit = bind(circuit, theta, angles)
        conditional, _ = grid.probability_gradient(
            entry, "mismatched", tau, angles)
        predicted_joint = float(conditional * CLOCK_ACCEPTANCE)
        vector = Statevector.from_instruction(circuit).data
        vector_clock = float(np.sum(np.abs(vector[:1 << (2 * N)]) ** 2))
        vector_joint = float(abs(vector[0]) ** 2)
        compiled = transpile(circuit, basis_gates=["u3", "cx"],
                             optimization_level=1)
        compiled.save_probabilities()
        result = density.run(compiled, shots=1, seed_simulator=SEED).result()
        if not result.success:
            raise RuntimeError(result.status)
        probabilities = np.asarray(result.data(0)["probabilities"], dtype=float)
        density_clock = float(np.sum(probabilities[:1 << (2 * N)]))
        density_joint = float(probabilities[0])
        error = max(abs(vector_clock - CLOCK_ACCEPTANCE),
                    abs(density_clock - CLOCK_ACCEPTANCE),
                    abs(vector_joint - predicted_joint),
                    abs(density_joint - predicted_joint))
        if error > 1e-10:
            raise AssertionError(f"clock circuit check failed: {error}")
        rows.append({"physical_P_zero": physical, "tau_over_pi": tau / math.pi,
                     "common_shift": shift, "axis": axis, "label": label,
                     "predicted_joint_probability": predicted_joint,
                     "statevector_clock_probability": vector_clock,
                     "density_clock_probability": density_clock,
                     "statevector_joint_probability": vector_joint,
                     "density_joint_probability": density_joint,
                     "max_absolute_error": error,
                     "compiled_depth": compiled.depth(),
                     "compiled_size": compiled.size()})
    return {"cases": rows,
            "max_absolute_error": max(row["max_absolute_error"] for row in rows)}


def acquire(shots_per_label: int, workers: int, seed: int) -> dict:
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    compiled = {}
    parameters = {}
    for physical in (0, 1):
        entry = grid.model(N, previous.BETA, bool(physical), previous.SPECTRUM)
        for tau_index, tau in enumerate(previous.TAUS):
            circuit, theta = template(entry, tau, measure=True)
            compiled[(physical, tau_index)] = transpile(
                circuit, basis_gates=["u3", "cx"], optimization_level=1)
            parameters[(physical, tau_index)] = theta
    allocations = np.stack([previous.shot_allocation(shots_per_label, name)
                            for name in previous.DISTRIBUTIONS])
    shape = (2, 2, 2, 2, N, LABELS)
    joint_counts = np.zeros(shape, dtype=int)
    clock_counts = np.zeros(shape, dtype=int)
    for dist_index in range(2):
        for label in range(LABELS):
            circuits = []
            settings = []
            for physical in (0, 1):
                for tau_index in range(2):
                    for shift_index, shift in enumerate(previous.SHIFTS):
                        for axis in range(N):
                            angles = np.full(N, shift + 2 * math.pi * label / LABELS)
                            angles[axis] -= grid.BIAS
                            circuits.append(bind(compiled[(physical, tau_index)],
                                                 parameters[(physical, tau_index)],
                                                 angles))
                            settings.append((physical, tau_index, shift_index, axis))
            shots = int(allocations[dist_index, label])
            result = simulator.run(circuits, shots=shots,
                                   seed_simulator=seed + 100 * dist_index + label).result()
            if not result.success:
                raise RuntimeError(result.status)
            for index, (physical, tau_index, shift_index, axis) in enumerate(settings):
                counts = result.get_counts(index)
                joint_counts[dist_index, physical, tau_index,
                             shift_index, axis, label] = counts.get("00000000", 0)
                clock_counts[dist_index, physical, tau_index,
                             shift_index, axis, label] = sum(
                                 count for bits, count in counts.items()
                                 if bits[:CLOCK_QUBITS] == "00")
            print(f"completed distribution={previous.DISTRIBUTIONS[dist_index]} "
                  f"label={label} shots_per_circuit={shots}", flush=True)
    return {"raw_axes": ["label_distribution", "physical", "tau", "shift",
                         "axis", "phase_label"],
            "raw_joint_zero_counts": joint_counts.tolist(),
            "raw_clock_accept_counts": clock_counts.tolist(),
            "shot_allocations": {name: allocations[index].tolist()
                                 for index, name in enumerate(previous.DISTRIBUTIONS)},
            "compiled_depths": {f"physical_{key[0]}_tau_{key[1]}": value.depth()
                                for key, value in compiled.items()}}


def analyze(raw: dict, bootstraps: int, bootstrap_seed: int) -> dict:
    counts = np.asarray(raw["raw_joint_zero_counts"], dtype=int)
    clock_counts = np.asarray(raw["raw_clock_accept_counts"], dtype=int)
    if counts.shape != (2, 2, 2, 2, N, LABELS) or clock_counts.shape != counts.shape:
        raise ValueError("unexpected raw count shape")
    if np.any(counts > clock_counts):
        raise AssertionError("joint count exceeds clock count")
    expected = exact_joint_probabilities()
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    for dist_index, distribution in enumerate(previous.DISTRIBUTIONS):
        shots = np.asarray(raw["shot_allocations"][distribution], dtype=int)
        weights = shots / shots.sum()
        for physical in (0, 1):
            for tau_index, tau in enumerate(previous.TAUS):
                first = counts[dist_index, physical, tau_index, 0]
                second = counts[dist_index, physical, tau_index, 1]
                exact_delta = (expected[physical, tau_index, 1] -
                               expected[physical, tau_index, 0]) @ weights
                test = weak.weak_null_test(first, second, shots, bootstraps, rng)
                rows.append({"label_distribution": distribution,
                             "physical_P_zero": bool(physical),
                             "tau_over_pi": tau / math.pi,
                             "exact_axis_pooled_joint_differences": exact_delta.tolist(),
                             "test": test})
    denominator = 0
    for allocation in raw["shot_allocations"].values():
        denominator += sum(allocation) * 2 * 2 * 2 * N
    accepted = int(clock_counts.sum())
    threshold = 0.05 / len(rows)
    return {"rows": rows, "familywise_tests": len(rows),
            "familywise_bonferroni_threshold": threshold,
            "detections": [(row["label_distribution"], row["physical_P_zero"],
                            row["tau_over_pi"]) for row in rows if
                           row["test"]["weak_null_bootstrap_pvalue"] <= threshold],
            "total_preparations": denominator,
            "clock_accepted_preparations": accepted,
            "observed_clock_acceptance_rate": accepted / denominator,
            "expected_clock_acceptance_rate": CLOCK_ACCEPTANCE}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots-per-label", type=int, default=SHOTS_PER_LABEL)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    if min(args.shots_per_label, args.workers, args.bootstraps) < 1:
        parser.error("shots, workers, and bootstraps must be positive")
    pilot = structural_pilot()
    checks = circuit_checks()
    if args.checks_only:
        print(json.dumps({"structural_pilot": pilot,
                          "circuit_checks": checks}, indent=2))
        return
    raw = acquire(args.shots_per_label, args.workers, args.seed)
    analysis = analyze(raw, args.bootstraps, BOOTSTRAP_SEED)
    output = {"protocol": PROTOCOL,
              "parameters": {"n": N, "phase_labels": LABELS,
                             "shots_per_label_if_uniform": args.shots_per_label,
                             "workers": args.workers, "aer_seed": args.seed,
                             "bootstrap_seed": BOOTSTRAP_SEED,
                             "bootstraps": args.bootstraps,
                             "circuits_executed": 336},
              "structural_pilot": pilot, "circuit_checks": checks,
              **raw, "exact_joint_probabilities": exact_joint_probabilities().tolist(),
              "analysis": analysis}
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({"clock_acceptance_rate": analysis[
                          "observed_clock_acceptance_rate"],
                      "pvalues": [(row["label_distribution"], row["physical_P_zero"],
                                   row["tau_over_pi"], row["test"][
                                       "weak_null_bootstrap_pvalue"])
                                  for row in analysis["rows"]],
                      "detections": analysis["detections"],
                      "result": str(RESULT)}, indent=2))


if __name__ == "__main__":
    main()
