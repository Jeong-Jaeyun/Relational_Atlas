"""I01 finite-clock intervention with phase labels, drift, and an anchor.

Both unrestricted and P=0 systems are entangled with a five-level energy
clock, encoded in three qubits. We condition on a chosen clock phase, then
reuse the phase-label and known-phase-anchor controls. The constraint and
clock are prepared by construction; this is a circuit-level toy experiment.
"""

from __future__ import annotations

import argparse
import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from relational_atlas_i01.phase_reference import qiskit_condition_probe as probe
from relational_atlas_i01.phase_reference import qiskit_phase_anchor_audit as anchor_audit
from relational_atlas_i01.phase_reference import qiskit_phase_label_intervention as intervention


PROJECT = PROJECT_ROOT
RESULT = PROJECT / "results" / "qiskit_clock_anchor_intervention.json"
EXACT_TAUS = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi)
SHOT_TAUS = (0.0, math.pi / 2)
CLOCK_LEVELS = 5
CLOCK_QUBITS = 3
SYSTEM_QUBITS = 8
TOTAL_QUBITS = SYSTEM_QUBITS + CLOCK_QUBITS
FIT_GRID = anchor_audit.FIT_GRID


def clock_model(physical: bool) -> dict:
    system = probe.model(intervention.N, physical)
    energies = np.sum(system["momenta"] ** 2, axis=1).astype(int)
    assert np.all((0 <= energies) & (energies < CLOCK_LEVELS))
    joint = np.zeros(1 << TOTAL_QUBITS, dtype=complex)
    for k, energy, weight in zip(system["momenta"], energies, system["weights"]):
        joint[probe.basis_index(k) | (int(energy) << SYSTEM_QUBITS)] = np.sqrt(weight)
    uniform = np.zeros(1 << CLOCK_QUBITS, dtype=complex)
    uniform[:CLOCK_LEVELS] = 1 / np.sqrt(CLOCK_LEVELS)
    return {
        "system": system,
        "energies": energies,
        "joint_preparation": StatePreparation(joint),
        "clock_projection_inverse": StatePreparation(uniform).inverse(),
        "constraint_residual": 0.0,  # H_C|E> = -E|E> by construction
    }


def conditional_probability_gradient(entry: dict, tau: float,
                                     angles: np.ndarray) -> tuple[float, np.ndarray]:
    system = entry["system"]
    factors = system["weights"] * np.exp(1j * (
        entry["energies"] * tau + system["momenta"] @ angles))
    overlap = np.sum(factors)
    derivative = 1j * (factors @ system["momenta"])
    return float(abs(overlap) ** 2), 2 * np.real(np.conj(overlap) * derivative)


def conditional_fisher(entry: dict, tau: float, offset: float,
                       keep_labels: bool) -> np.ndarray:
    n, length = intervention.N, intervention.L
    fisher = np.zeros((n, n))
    for i in range(n):
        probabilities = []
        gradients = []
        for t in range(length):
            p, gradient = conditional_probability_gradient(
                entry, tau, intervention.angle(i, t, offset))
            probabilities.append(p)
            gradients.append(gradient)
        if keep_labels:
            for p, gradient in zip(probabilities, gradients):
                fisher += np.outer(gradient, gradient) / (p * (1 - p)) / (n * length)
        else:
            p = float(np.mean(probabilities))
            gradient = np.mean(gradients, axis=0)
            fisher += np.outer(gradient, gradient) / (p * (1 - p)) / n
    return fisher


def profiled_fisher(entry: dict, tau: float, anchor_ratio: float) -> np.ndarray:
    n, blocks = intervention.N, len(anchor_audit.OFFSETS)
    joint = np.zeros((n + blocks, n + blocks))
    ones = np.ones(n)
    known_reference = clock_model(False)
    for block, offset in enumerate(anchor_audit.OFFSETS):
        science = conditional_fisher(entry, tau, offset, True)
        calibration = conditional_fisher(known_reference, tau, offset, True)
        weight = 1 / blocks
        cross = science @ ones
        joint[:n, :n] += weight * science
        joint[:n, n + block] += weight * cross
        joint[n + block, :n] += weight * cross
        joint[n + block, n + block] += weight * (
            float(ones @ science @ ones) + anchor_ratio * float(ones @ calibration @ ones))
    effective = joint[:n, :n] - joint[:n, n:] @ np.linalg.pinv(
        joint[n:, n:], rcond=1e-12) @ joint[n:, :n]
    return (effective + effective.T) / 2


def rank_summary(fisher: np.ndarray) -> dict:
    eigs = np.linalg.eigvalsh(fisher)
    common = np.ones(intervention.N) / np.sqrt(intervention.N)
    return {
        "eigenvalues": eigs.tolist(),
        "rank_tol_1e_minus_8": int(np.sum(eigs > 1e-8)),
        "rank_at_1pct_of_max": int(np.sum(eigs > 0.01 * eigs[-1])),
        "common_direction_information": float(common @ fisher @ common),
    }


def circuit(entry: dict, tau: float | Parameter,
            angles: np.ndarray | ParameterVector, measure: bool = False) -> QuantumCircuit:
    qc = QuantumCircuit(TOTAL_QUBITS)
    qc.append(entry["joint_preparation"], range(TOTAL_QUBITS))
    for clock_bit in range(CLOCK_QUBITS):
        qc.rz((1 << clock_bit) * tau, SYSTEM_QUBITS + clock_bit)
    for axis in range(intervention.N):
        qc.rz(angles[axis], 2 * axis)
        qc.rz(-angles[axis], 2 * axis + 1)
    qc.append(entry["clock_projection_inverse"],
              range(SYSTEM_QUBITS, TOTAL_QUBITS))
    qc.append(entry["system"]["preparation"].inverse(), range(SYSTEM_QUBITS))
    if measure:
        qc.measure_all()
    return qc


def exact_circuit_crosscheck(entry: dict, tau: float, angles: np.ndarray) -> dict:
    state = Statevector.from_instruction(circuit(entry, tau, angles)).data
    clock_probability = float(np.sum(np.abs(state[:1 << SYSTEM_QUBITS]) ** 2))
    joint_probability = float(abs(state[0]) ** 2)
    conditional_probability, _ = conditional_probability_gradient(entry, tau, angles)
    assert np.isclose(clock_probability, 1 / CLOCK_LEVELS, atol=1e-11)
    assert np.isclose(joint_probability, conditional_probability / CLOCK_LEVELS,
                      atol=1e-11)
    return {
        "clock_postselection_probability": clock_probability,
        "joint_zero_probability": joint_probability,
        "formula_joint_zero_probability": conditional_probability / CLOCK_LEVELS,
    }


def exact_study() -> dict:
    entries = {"unrestricted": clock_model(False), "P_zero": clock_model(True)}
    rows = {}
    for name, entry in entries.items():
        tau_rows = []
        for tau in EXACT_TAUS:
            kept = conditional_fisher(entry, tau, 0.0, True)
            erased = conditional_fisher(entry, tau, 0.0, False)
            no_anchor = profiled_fisher(entry, tau, 0.0)
            with_anchor = profiled_fisher(entry, tau, 1.0)
            check = exact_circuit_crosscheck(
                entry, tau, intervention.angle(0, 0, 0.2))
            tau_rows.append({
                "tau_over_pi": float(tau / math.pi),
                "label_kept": rank_summary(kept),
                "label_erased": rank_summary(erased),
                "unknown_offset_no_anchor": rank_summary(no_anchor),
                "unknown_offset_with_anchor_ratio_1": rank_summary(with_anchor),
                "qiskit_statevector_crosscheck": check,
            })
        rows[name] = tau_rows
    return rows


def clock_information_scan(points: int = 201) -> dict:
    """Track information magnitude and rank across the chosen clock interval."""
    if points < 2:
        raise ValueError("points must be at least 2")
    entries = {"unrestricted": clock_model(False), "P_zero": clock_model(True)}
    rows = {}
    for name, entry in entries.items():
        rows[name] = []
        for tau in np.linspace(0.0, math.pi, points):
            summary = rank_summary(conditional_fisher(entry, tau, 0.0, True))
            rows[name].append({
                "tau_over_pi": float(tau / math.pi),
                "smallest_positive_eigenvalue": float(
                    summary["eigenvalues"][0 if name == "unrestricted" else 1]),
                "common_direction_information": summary["common_direction_information"],
                "rank_tol_1e_minus_8": summary["rank_tol_1e_minus_8"],
            })
    return {"grid_points": points, "tau_domain_over_pi": [0.0, 1.0],
            "label_kept": rows}


def full_conditional_probabilities(parameters: np.ndarray, tau: float) -> np.ndarray:
    values = np.asarray(parameters, dtype=float)[..., None]
    phases = values + 2 * np.pi * np.arange(intervention.L) / intervention.L
    a = np.exp(-probe.BETA)
    def characteristic(x: np.ndarray) -> np.ndarray:
        return (1 + 2 * a * np.exp(1j * tau) * np.cos(x)) / (1 + 2 * a)
    overlap = characteristic(phases - intervention.BIAS) * characteristic(phases) ** 3
    return np.abs(overlap) ** 2


def fit_full_counts(joint_success: np.ndarray, clock_success: np.ndarray,
                    tau: float) -> dict:
    successes = np.asarray(joint_success, dtype=int).sum(axis=0)
    accepted = np.asarray(clock_success, dtype=int).sum(axis=0)
    p_grid = np.clip(full_conditional_probabilities(FIT_GRID, tau), 1e-14, 1 - 1e-14)
    likelihood = (successes * np.log(p_grid) +
                  (accepted - successes) * np.log1p(-p_grid)).sum(axis=1)
    best = int(np.argmax(likelihood))
    estimate = float(FIT_GRID[best])
    h = 1e-5
    p = full_conditional_probabilities(np.array([estimate]), tau)[0]
    derivative = (full_conditional_probabilities(np.array([estimate + h]), tau)[0]
                  - full_conditional_probabilities(np.array([estimate - h]), tau)[0]) / (2 * h)
    information = float(np.sum(accepted * derivative ** 2 / (p * (1 - p))))
    interval_grid = FIT_GRID[2 * (likelihood[best] - likelihood) <= 3.841459]
    return {
        "estimate_radians": estimate,
        "fisher_standard_error_radians": float(1 / np.sqrt(information)),
        "profile_likelihood_95pct_interval_radians": [float(interval_grid[0]),
                                                        float(interval_grid[-1])],
        "accepted_clock_counts_by_phase": accepted.tolist(),
        "joint_zero_counts_by_phase": successes.tolist(),
    }


def run_circuits(simulator: AerSimulator, entry: dict, schedule: list[tuple],
                 shots: int, seed: int) -> dict:
    tau_parameter = Parameter("tau")
    phase_parameters = ParameterVector("phase", intervention.N)
    template = circuit(entry, tau_parameter, phase_parameters, measure=True)
    # Aer 0.17.2 on this Windows installation crashes on multiplexer.
    compiled = transpile(template, basis_gates=["u3", "cx"], optimization_level=1)
    circuits = []
    for tau_index, block, run_type, i, t, common in schedule:
        angles = intervention.angle(i, t, common)
        values = {tau_parameter: SHOT_TAUS[tau_index]}
        values.update({phase_parameters[axis]: float(angles[axis])
                       for axis in range(intervention.N)})
        circuits.append(compiled.assign_parameters(values))
    result = simulator.run(circuits, shots=shots, seed_simulator=seed).result()
    if not result.success:
        raise RuntimeError(result.status)
    shape = (len(SHOT_TAUS), len(anchor_audit.OFFSETS), 2,
             intervention.N, intervention.L)
    clock_counts = np.zeros(shape, dtype=int)
    joint_counts = np.zeros(shape, dtype=int)
    errors = []
    for index, (tau_index, block, run_type, i, t, common) in enumerate(schedule):
        counts = result.get_counts(index)
        accepted = sum(value for bits, value in counts.items()
                       if bits[:CLOCK_QUBITS] == "0" * CLOCK_QUBITS)
        joint = counts.get("0" * TOTAL_QUBITS, 0)
        clock_counts[tau_index, block, run_type, i, t] = accepted
        joint_counts[tau_index, block, run_type, i, t] = joint
        expected, _ = conditional_probability_gradient(
            entry, SHOT_TAUS[tau_index], intervention.angle(i, t, common))
        errors.append(abs(joint / shots - expected / CLOCK_LEVELS))
    return {
        "clock_success_counts": clock_counts,
        "joint_zero_counts": joint_counts,
        "circuits": len(circuits),
        "transpiled_depth": compiled.depth(),
        "transpiled_size": compiled.size(),
        "mean_absolute_joint_probability_error": float(np.mean(errors)),
        "overall_clock_postselection_rate": float(clock_counts.sum() /
                                                   (len(circuits) * shots)),
    }


def shot_study(shots: int, workers: int, seed: int, permutations: int) -> dict:
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    full_schedule = [
        (tau_index, block, run_type, i, t,
         (anchor_audit.ALPHA_TRUE if run_type == 0 else 0.0) + offset)
        for tau_index in range(len(SHOT_TAUS))
        for block, offset in enumerate(anchor_audit.OFFSETS)
        for run_type in (0, 1)  # science, known-phase anchor
        for i in range(intervention.N) for t in range(intervention.L)
    ]
    physical_schedule = [
        (tau_index, block, 0, i, t, anchor_audit.ALPHA_TRUE + offset)
        for tau_index in range(len(SHOT_TAUS))
        for block, offset in enumerate(anchor_audit.OFFSETS)
        for i in range(intervention.N) for t in range(intervention.L)
    ]
    full = run_circuits(simulator, clock_model(False), full_schedule, shots, seed)
    physical = run_circuits(simulator, clock_model(True), physical_schedule,
                            shots, seed + 1)
    rows = []
    rng = np.random.default_rng(seed + 100)
    for tau_index, tau in enumerate(SHOT_TAUS):
        block_fits = []
        for block in range(len(anchor_audit.OFFSETS)):
            science = fit_full_counts(full["joint_zero_counts"][tau_index, block, 0],
                                      full["clock_success_counts"][tau_index, block, 0], tau)
            calibration = fit_full_counts(full["joint_zero_counts"][tau_index, block, 1],
                                          full["clock_success_counts"][tau_index, block, 1], tau)
            alpha = science["estimate_radians"] - calibration["estimate_radians"]
            se = math.hypot(science["fisher_standard_error_radians"],
                            calibration["fisher_standard_error_radians"])
            block_fits.append({
                "block": block,
                "science_alpha_plus_offset": science,
                "known_phase_anchor_offset": calibration,
                "calibrated_alpha_estimate": alpha,
                "calibrated_alpha_standard_error": se,
            })
        precisions = np.array([1 / row["calibrated_alpha_standard_error"] ** 2
                               for row in block_fits])
        estimates = np.array([row["calibrated_alpha_estimate"] for row in block_fits])
        combined = float(np.dot(precisions, estimates) / precisions.sum())
        combined_se = float(1 / np.sqrt(precisions.sum()))

        full0 = full["joint_zero_counts"][tau_index, 0, 0]
        full1 = full["joint_zero_counts"][tau_index, 1, 0]
        phys0 = physical["joint_zero_counts"][tau_index, 0, 0]
        phys1 = physical["joint_zero_counts"][tau_index, 1, 0]
        rows.append({
            "tau_over_pi": float(tau / math.pi),
            "block_fits": block_fits,
            "combined_alpha_estimate": combined,
            "combined_alpha_approximate_95pct_interval": [
                combined - 1.96 * combined_se, combined + 1.96 * combined_se],
            "full_science_block_difference_labeled": intervention.permutation_pvalue(
                full0, full1, shots, permutations, rng),
            "full_science_block_difference_erased": intervention.permutation_pvalue(
                full0.sum(axis=1), full1.sum(axis=1), intervention.L * shots,
                permutations, rng),
            "P_zero_block_difference_labeled": intervention.permutation_pvalue(
                phys0, phys1, shots, permutations, rng),
            "P_zero_block_difference_erased": intervention.permutation_pvalue(
                phys0.sum(axis=1), phys1.sum(axis=1), intervention.L * shots,
                permutations, rng),
        })
    return {
        "shots_per_circuit": shots,
        "clock_success_target": 1 / CLOCK_LEVELS,
        "full_metadata": {key: value for key, value in full.items()
                          if not isinstance(value, np.ndarray)},
        "P_zero_metadata": {key: value for key, value in physical.items()
                            if not isinstance(value, np.ndarray)},
        "raw_count_axes": ["tau_index", "block", "run_type_0_science_1_anchor",
                           "reference_axis", "phase_label"],
        "full_clock_success_counts": full["clock_success_counts"].tolist(),
        "full_joint_zero_counts": full["joint_zero_counts"].tolist(),
        "P_zero_clock_success_counts": physical["clock_success_counts"][:, :, 0].tolist(),
        "P_zero_joint_zero_counts": physical["joint_zero_counts"][:, :, 0].tolist(),
        "tau_results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--exact-only", action="store_true")
    args = parser.parse_args()
    if min(args.shots, args.workers, args.permutations) < 1:
        parser.error("shots, workers, permutations must be positive")
    output = {
        "model": "n=4, momentum cutoff 1, five-level energy clock encoded in 3 qubits",
        "physical_limit": "The energy-matched constraint, clock state, and known-phase anchor are prepared resources; not a derived cosmology.",
        "parameters": {
            "clock_levels": CLOCK_LEVELS,
            "clock_postselection_target": 1 / CLOCK_LEVELS,
            "exact_tau_over_pi": [float(t / math.pi) for t in EXACT_TAUS],
            "shot_tau_over_pi": [float(t / math.pi) for t in SHOT_TAUS],
            "alpha_true": anchor_audit.ALPHA_TRUE,
            "block_offsets": anchor_audit.OFFSETS,
            "shots_per_circuit": args.shots,
            "workers": args.workers,
            "seed": args.seed,
        },
        "exact": exact_study(),
        "clock_information_scan": clock_information_scan(),
    }
    if not args.exact_only:
        output["shots"] = shot_study(args.shots, args.workers, args.seed,
                                     args.permutations)
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "exact_ranks": {name: [(row["tau_over_pi"],
                               row["label_kept"]["rank_tol_1e_minus_8"],
                               row["label_erased"]["rank_tol_1e_minus_8"],
                               row["unknown_offset_no_anchor"]["rank_tol_1e_minus_8"],
                               row["unknown_offset_with_anchor_ratio_1"]["rank_tol_1e_minus_8"])
                              for row in rows] for name, rows in output["exact"].items()},
        "shot_alpha": [row["combined_alpha_estimate"] for row in
                       output.get("shots", {}).get("tau_results", [])],
        "result": str(RESULT),
    }, indent=2))


if __name__ == "__main__":
    main()
