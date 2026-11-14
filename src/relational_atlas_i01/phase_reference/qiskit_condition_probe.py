"""Circuit-level I01 controls: physical constraint versus detector restriction.

Each torus momentum k_i in {-1, 0, 1} uses two qubits: 00=0, 01=+1,
10=-1; 11 is unoccupied. RZ(theta) on the positive bit and RZ(-theta)
on the negative bit implement exp(i*k_i*theta) exactly on these codes.

The binary detector asks whether the phase-encoded state returns to the
reference state after inverse preparation. Averaging this detector over a
common phase twirls the total charge P=sum_i k_i. This makes an unrestricted
state look (n-1)-dimensional to that detector without imposing P=0.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from relational_atlas_i01.geometry import condition_atlas as atlas


PROJECT = PROJECT_ROOT
RESULT = PROJECT / "results" / "qiskit_condition_probe.json"
BETA = atlas.BETA


def basis_index(momentum: np.ndarray) -> int:
    """Qiskit's little-endian integer index of a momentum tuple."""
    index = 0
    for axis, value in enumerate(momentum):
        if value == 1:
            index |= 1 << (2 * axis)
        elif value == -1:
            index |= 1 << (2 * axis + 1)
        elif value != 0:
            raise ValueError("This circuit encoding requires cutoff=1")
    return index


def model(n: int, physical: bool) -> dict:
    momenta = atlas.grid(n, 1)
    if physical:
        momenta = momenta[np.sum(momenta, axis=1) == 0]
    energies = np.sum(momenta * momenta, axis=1)
    weights = np.exp(-BETA * energies)
    weights /= weights.sum()
    amplitudes = np.zeros(1 << (2 * n), dtype=complex)
    for k, weight in zip(momenta, weights):
        amplitudes[basis_index(k)] = np.sqrt(weight)
    return {
        "n": n,
        "physical": physical,
        "momenta": momenta,
        "weights": weights,
        "amplitudes": amplitudes,
        "preparation": StatePreparation(amplitudes),
    }


def qfi_from_qiskit_state(entry: dict) -> np.ndarray:
    state = Statevector(entry["amplitudes"])
    weights = np.abs(state.data[[basis_index(k) for k in entry["momenta"]]]) ** 2
    return atlas.qfi_from_weights(entry["momenta"], weights)


def phase_overlap(entry: dict, angles: np.ndarray) -> complex:
    return complex(np.sum(entry["weights"] * np.exp(1j * (entry["momenta"] @ angles))))


def return_probability(entry: dict, angles: np.ndarray) -> float:
    return float(abs(phase_overlap(entry, angles)) ** 2)


def return_gradient(entry: dict, angles: np.ndarray) -> np.ndarray:
    factors = entry["weights"] * np.exp(1j * (entry["momenta"] @ angles))
    z = np.sum(factors)
    dz = 1j * (factors @ entry["momenta"])
    return 2 * np.real(np.conj(z) * dz)


def phase_circuit(entry: dict, angles: np.ndarray, measure: bool = False) -> QuantumCircuit:
    n = entry["n"]
    circuit = QuantumCircuit(2 * n)
    circuit.append(entry["preparation"], range(2 * n))
    for axis, angle in enumerate(angles):
        circuit.rz(float(angle), 2 * axis)
        circuit.rz(-float(angle), 2 * axis + 1)
    circuit.append(entry["preparation"].inverse(), range(2 * n))
    if measure:
        circuit.measure_all()
    return circuit


def template(entry: dict) -> tuple[QuantumCircuit, ParameterVector]:
    n = entry["n"]
    parameters = ParameterVector("phase", n)
    circuit = QuantumCircuit(2 * n)
    circuit.append(entry["preparation"], range(2 * n))
    for axis in range(n):
        circuit.rz(parameters[axis], 2 * axis)
        circuit.rz(-parameters[axis], 2 * axis + 1)
    circuit.append(entry["preparation"].inverse(), range(2 * n))
    circuit.measure_all()
    return circuit, parameters


def probe_settings(n: int, bias: float, step: float, twirled: bool):
    phases = range(2 * n + 1) if twirled else range(1)
    for reference_axis in range(n):
        for derivative_axis, sign in [(None, 0)] + [
            (axis, direction) for axis in range(n) for direction in (-1, 1)
        ]:
            for phase_index in phases:
                angles = np.zeros(n)
                angles[reference_axis] = -bias
                if derivative_axis is not None:
                    angles[derivative_axis] += sign * step
                if twirled:
                    angles += 2 * np.pi * phase_index / (2 * n + 1)
                yield (reference_axis, derivative_axis, sign, phase_index), angles


def fisher_from_measurements(entry: dict, bias: float, step: float,
                             twirled: bool, probabilities: dict | None = None) -> dict:
    n = entry["n"]
    settings = list(probe_settings(n, bias, step, twirled))
    if probabilities is None:
        probabilities = {key: return_probability(entry, angles) for key, angles in settings}
    n_phases = 2 * n + 1 if twirled else 1

    def average(i: int, j: int | None, sign: int) -> float:
        return float(np.mean([probabilities[(i, j, sign, t)] for t in range(n_phases)]))

    fisher = np.zeros((n, n))
    baselines = []
    gradients = []
    for i in range(n):
        p = average(i, None, 0)
        grad = np.array([(average(i, j, 1) - average(i, j, -1)) / (2 * step)
                         for j in range(n)])
        fisher += np.outer(grad, grad) / (p * (1 - p)) / n
        baselines.append(p)
        gradients.append(grad.tolist())
    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    return {
        "eigenvalues": eigenvalues.tolist(),
        "rank_at_1pct_of_max": int(np.sum(eigenvalues > 0.01 * eigenvalues[-1])),
        "smallest_to_next_ratio": float(eigenvalues[0] / eigenvalues[1]),
        "smallest_mode_common_alignment_abs": float(abs(
            eigenvectors[:, 0] @ (np.ones(n) / np.sqrt(n)))),
        "mean_probability": float(np.mean(baselines)),
        "gradients": gradients,
        "fisher_matrix": fisher.tolist(),
    }


def analytic_fisher(entry: dict, bias: float, twirled: bool) -> np.ndarray:
    """Continuous-derivative CFI; separate from the finite-difference estimate."""
    n = entry["n"]
    n_phases = 2 * n + 1 if twirled else 1
    fisher = np.zeros((n, n))
    for i in range(n):
        p = 0.0
        grad = np.zeros(n)
        for t in range(n_phases):
            angles = np.zeros(n)
            angles[i] = -bias
            if twirled:
                angles += 2 * np.pi * t / n_phases
            p += return_probability(entry, angles) / n_phases
            grad += return_gradient(entry, angles) / n_phases
        fisher += np.outer(grad, grad) / (p * (1 - p)) / n
    return fisher


def phase_reference_fisher(entry: dict, bias: float, loss_fraction: float) -> np.ndarray:
    """CFI after unlabelled classical mixing of referenced and twirled trials."""
    n = entry["n"]
    fisher = np.zeros((n, n))
    for i in range(n):
        angle = np.zeros(n)
        angle[i] = -bias
        p_ref = return_probability(entry, angle)
        g_ref = return_gradient(entry, angle)
        p_twirl = 0.0
        g_twirl = np.zeros(n)
        for t in range(2 * n + 1):
            shifted = angle + 2 * np.pi * t / (2 * n + 1)
            p_twirl += return_probability(entry, shifted) / (2 * n + 1)
            g_twirl += return_gradient(entry, shifted) / (2 * n + 1)
        p = (1 - loss_fraction) * p_ref + loss_fraction * p_twirl
        grad = (1 - loss_fraction) * g_ref + loss_fraction * g_twirl
        fisher += np.outer(grad, grad) / (p * (1 - p)) / n
    return fisher


def phase_reference_scan(entry: dict, bias: float) -> list[dict]:
    """Blend a phase-referenced detector with a charge-twirled detector.

    The mixture is on the *binary probabilities*, representing a classical
    loss of the common-phase reference on a fraction of interrogations.
    """
    rows = []
    for loss_fraction in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0):
        fisher = phase_reference_fisher(entry, bias, loss_fraction)
        eigs = np.linalg.eigvalsh(fisher)
        rows.append({
            "phase_reference_loss_fraction": loss_fraction,
            "cfi_eigenvalues": eigs.tolist(),
            "mathematical_rank": int(np.sum(eigs > 1e-10)),
            "rank_at_1pct_of_max": int(np.sum(eigs > 0.01 * eigs[-1])),
            "common_to_relative_eigenvalue_ratio": float(eigs[0] / eigs[1]),
        })
    return rows


def apparent_rank_transition(entry: dict, bias: float, threshold_fraction: float = 0.01) -> float:
    """Loss fraction where min/max CFI eigenvalue crosses a chosen cutoff."""
    low, high = 0.0, 1.0
    for _ in range(45):
        middle = (low + high) / 2
        eigs = np.linalg.eigvalsh(phase_reference_fisher(entry, bias, middle))
        if eigs[0] / eigs[-1] > threshold_fraction:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def exact_study(bias: float, step: float) -> dict:
    dimensions = {}
    for n in (3, 4, 5, 6):
        row = {}
        for name, physical in (("unrestricted", False), ("P_zero", True)):
            entry = model(n, physical)
            qfi = qfi_from_qiskit_state(entry)
            control = atlas.qfi_from_weights(entry["momenta"], entry["weights"].copy())
            assert np.allclose(qfi, control, atol=1e-12)
            eigenvalues = np.linalg.eigvalsh(qfi)
            row[name] = {
                "support_size": len(entry["momenta"]),
                "qfi_eigenvalues": eigenvalues.tolist(),
                "qfi_rank": int(np.sum(eigenvalues > 1e-8)),
            }
        twirled = analytic_fisher(model(n, False), bias, True)
        twirled_eigenvalues = np.linalg.eigvalsh(twirled)
        row["charge_twirled_detector"] = {
            "return_cfi_eigenvalues": twirled_eigenvalues.tolist(),
            "return_cfi_rank": int(np.sum(twirled_eigenvalues > 1e-8)),
        }
        dimensions[str(n)] = row

    circuits = {}
    for name, physical, twirled in (("unrestricted", False, False),
                                     ("P_zero", True, False),
                                     ("unrestricted_charge_twirled", False, True)):
        entry = model(4, physical)
        sampled = fisher_from_measurements(entry, bias, step, twirled)
        analytic = analytic_fisher(entry, bias, twirled)
        eigenvalues = np.linalg.eigvalsh(analytic)
        check_angle = np.array([0.15, -0.45, 0.2, 0.3])
        observed = Statevector.from_instruction(phase_circuit(entry, check_angle)).probabilities()[0]
        expected = return_probability(entry, check_angle)
        assert np.isclose(observed, expected, atol=1e-10)
        circuits[name] = {
            "analytic_cfi_eigenvalues": eigenvalues.tolist(),
            "analytic_cfi_rank": int(np.sum(eigenvalues > 1e-8)),
            "finite_difference_exact": sampled,
            "qiskit_statevector_return_probability": float(observed),
            "formula_return_probability": expected,
        }
    return {
        "dimensions": dimensions,
        "binary_return_detector": circuits,
        "phase_reference_loss_scan_n4": phase_reference_scan(model(4, False), bias),
        "phase_reference_apparent_rank3_transition_at_1pct_n4": apparent_rank_transition(
            model(4, False), bias),
    }


def shot_study(bias: float, step: float, shots: int, twirl_shots: int,
               seed: int, workers: int, bootstrap_reps: int) -> dict:
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    rows = {}
    for name, physical, twirled in (("unrestricted", False, False),
                                     ("P_zero", True, False),
                                     ("unrestricted_charge_twirled", False, True)):
        entry = model(4, physical)
        circuit, parameters = template(entry)
        # Aer 0.17.2 on this Windows environment crashes while assembling the
        # synthesized ``multiplexer`` instruction. Decompose to standard gates.
        compiled = transpile(circuit, basis_gates=["u3", "cx"], optimization_level=1)
        settings = list(probe_settings(4, bias, step, twirled))
        circuits = [compiled.assign_parameters({parameters[i]: float(angles[i])
                                                for i in range(4)})
                    for _, angles in settings]
        this_shots = twirl_shots if twirled else shots
        result = simulator.run(circuits, shots=this_shots,
                               seed_simulator=seed + len(rows)).result()
        if not result.success:
            raise RuntimeError(f"Aer failed on {name}: {result.status}")
        probabilities = {}
        for index, (key, _) in enumerate(settings):
            counts = result.get_counts(index)
            probabilities[key] = counts.get("0" * 8, 0) / this_shots
        finite = fisher_from_measurements(entry, bias, step, twirled, probabilities)
        ideal = fisher_from_measurements(entry, bias, step, twirled)
        rng = np.random.default_rng(seed + 100 + len(rows))
        bootstrap_eigenvalues = []
        bootstrap_ranks = []
        for _ in range(bootstrap_reps):
            resampled = {key: rng.binomial(this_shots, p) / this_shots
                         for key, p in probabilities.items()}
            summary = fisher_from_measurements(entry, bias, step, twirled, resampled)
            bootstrap_eigenvalues.append(summary["eigenvalues"])
            bootstrap_ranks.append(summary["rank_at_1pct_of_max"])
        bootstrap_eigenvalues = np.asarray(bootstrap_eigenvalues)
        rows[name] = {
            "shots_per_circuit": this_shots,
            "circuit_count": len(circuits),
            "transpiled_depth": compiled.depth(),
            "transpiled_size": compiled.size(),
            "shot_finite_difference": finite,
            "exact_finite_difference": ideal,
            "binomial_bootstrap": {
                "replicates": bootstrap_reps,
                "smallest_eigenvalue_95pct_interval": np.percentile(
                    bootstrap_eigenvalues[:, 0], [2.5, 97.5]).tolist(),
                "rank_at_1pct_counts": {str(rank): int(np.sum(np.asarray(bootstrap_ranks) == rank))
                                        for rank in range(1, 5)},
            },
            "mean_absolute_probability_error": float(np.mean([
                abs(probabilities[key] - return_probability(entry, angles))
                for key, angles in settings
            ])),
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-shots", action="store_true")
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--twirl-shots", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if min(args.shots, args.twirl_shots, args.workers, args.bootstrap) < 1:
        parser.error("shots, twirl-shots, workers, and bootstrap must all be positive")
    bias, step = 0.8, 0.15
    output = {
        "model": "finite T^n Fourier cutoff=1, beta=0.35, two-qubit momentum encoding",
        "interpretation": "Gauge support, detector charge twirl, finite-difference bias, and finite-shot leakage are distinct. The 1% measured-rank threshold is a reporting convention, not an exact-rank proof.",
        "parameters": {"bias_radians": bias, "finite_difference_step": step,
                       "seed": args.seed, "workers": args.workers,
                       "bootstrap_replicates": args.bootstrap},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("qiskit", "qiskit-aer", "numpy")},
        "exact": exact_study(bias, step),
    }
    if not args.skip_shots:
        output["shots"] = shot_study(bias, step, args.shots, args.twirl_shots,
                                     args.seed, args.workers, args.bootstrap)
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "qfi_ranks": {n: {case: value["qfi_rank"] for case, value in rows.items()
                           if "qfi_rank" in value}
                      for n, rows in output["exact"]["dimensions"].items()},
        "exact_cfi_ranks": {name: row["analytic_cfi_rank"] for name, row in
                            output["exact"]["binary_return_detector"].items()},
        "shot_cfi_eigenvalues": {name: row["shot_finite_difference"]["eigenvalues"]
                                 for name, row in output.get("shots", {}).items()},
        "result": str(RESULT),
    }, indent=2))


if __name__ == "__main__":
    main()
