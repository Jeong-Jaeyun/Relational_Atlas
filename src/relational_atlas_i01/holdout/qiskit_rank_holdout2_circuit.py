"""Independent Qiskit checks for protocol-2 effects and clock spectra."""

from __future__ import annotations

import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate, StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from relational_atlas_i01.phase_reference import qiskit_condition_probe as encoding
from relational_atlas_i01.holdout import qiskit_rank_holdout2_grid as grid


PROJECT = PROJECT_ROOT
RESULT = PROJECT / "results" / "qiskit_rank_holdout2_circuit.json"
CASES = (
    (3, 0.45, False, "nonlinear", "mismatched", math.pi / 2, 1),
    (3, 1.20, False, "charge_coupled", "mixed", math.pi / 2, 2),
    (3, 0.45, True, "nonlinear", "mixed", 0.0, 3),
    (3, 1.20, False, "nonlinear", "twirled", math.pi / 2, 2),
)


def component_circuit(entry: dict, reference_beta: float, tau: float,
                      angles: np.ndarray) -> QuantumCircuit:
    n = entry["n"]
    basis_size = 1 << (2 * n)
    system = np.zeros(basis_size, dtype=complex)
    reference = np.zeros(basis_size, dtype=complex)
    reference_weights = np.exp(-reference_beta * entry["base_energies"])
    reference_weights /= reference_weights.sum()
    phases = np.ones(basis_size, dtype=complex)
    for k, weight, reference_weight, energy in zip(
        entry["momenta"], entry["weights"], reference_weights,
        entry["energies"]
    ):
        basis = encoding.basis_index(k)
        system[basis] = math.sqrt(weight)
        reference[basis] = math.sqrt(reference_weight)
        phases[basis] = np.exp(1j * tau * energy)
    qc = QuantumCircuit(2 * n)
    qc.append(StatePreparation(system), range(2 * n))
    qc.append(DiagonalGate(phases), range(2 * n))
    for axis, angle in enumerate(angles):
        qc.rz(float(angle), 2 * axis)
        qc.rz(-float(angle), 2 * axis + 1)
    qc.append(StatePreparation(reference).inverse(), range(2 * n))
    return qc


def reference_components(detector: str) -> tuple[tuple[float, float], ...]:
    if detector == "mismatched":
        return ((1.0, 0.15),)
    if detector in ("mixed", "twirled"):
        return ((0.7, 0.15), (0.3, 1.80))
    raise ValueError(detector)


def circuit_probability(entry: dict, detector: str, tau: float,
                        angles: np.ndarray, aer: bool = False) -> tuple[float, int]:
    labels = 2 * entry["n"] + 1
    offsets = range(labels) if detector == "twirled" else range(1)
    total = 0.0
    max_depth = 0
    for offset in offsets:
        shifted = angles + 2 * math.pi * offset / labels
        for mass, reference_beta in reference_components(detector):
            qc = component_circuit(entry, reference_beta, tau, shifted)
            if aer:
                compiled = transpile(qc, basis_gates=["u3", "cx"],
                                     optimization_level=1)
                compiled.save_probabilities()
                simulator = AerSimulator(method="density_matrix",
                                         max_parallel_threads=2,
                                         max_parallel_experiments=1)
                result = simulator.run(compiled, shots=1,
                                       seed_simulator=20261016).result()
                if not result.success:
                    raise RuntimeError(result.status)
                probability = float(result.data(0)["probabilities"][0])
                max_depth = max(max_depth, compiled.depth())
            else:
                probability = float(abs(Statevector.from_instruction(qc).data[0]) ** 2)
            total += mass * probability / len(offsets)
    return total, max_depth


def check_case(spec: tuple) -> dict:
    n, beta, physical, spectrum, detector, tau, label = spec
    entry = grid.model(n, beta, physical, spectrum)
    angles = np.full(n, 2 * math.pi * label / (2 * n + 1))
    angles[0] -= grid.BIAS
    analytic, _ = grid.probability_gradient(entry, detector, tau, angles)
    statevector, _ = circuit_probability(entry, detector, tau, angles)
    row = {"n": n, "beta": beta, "physical_P_zero": physical,
           "spectrum": spectrum, "detector": detector,
           "tau_over_pi": tau / math.pi, "phase_label_index": label,
           "analytic_probability": float(analytic),
           "statevector_probability": statevector,
           "statevector_absolute_error": abs(float(analytic) - statevector)}
    if detector != "twirled":
        density, depth = circuit_probability(entry, detector, tau, angles, aer=True)
        row.update({"aer_density_matrix_probability": density,
                    "aer_density_matrix_absolute_error": abs(
                        float(analytic) - density),
                    "maximum_transpiled_depth": depth})
    if row["statevector_absolute_error"] > 1e-10 or row.get(
        "aer_density_matrix_absolute_error", 0.0) > 1e-10:
        raise AssertionError("protocol-2 circuit probability mismatch")
    return row


def check_skewed_pooling() -> dict:
    entry = grid.model(3, 0.45, False, "nonlinear")
    label_count = 2 * entry["n"] + 1
    weights = grid.label_weights(label_count, "skewed")
    angles = grid.angles_for_grid(entry["n"], label_count)[0]
    analytic = np.array([grid.probability_gradient(
        entry, "mixed", math.pi / 2, angle)[0] for angle in angles], dtype=float)
    circuit = np.array([circuit_probability(
        entry, "mixed", math.pi / 2, angle)[0] for angle in angles])
    if np.max(np.abs(analytic - circuit)) > 1e-10:
        raise AssertionError("skewed label circuit pooling mismatch")
    return {"label_weights": weights.tolist(),
            "analytic_erased_probability": float(analytic @ weights),
            "circuit_erased_probability": float(circuit @ weights),
            "maximum_label_probability_error": float(np.max(
                np.abs(analytic - circuit)))}


def main() -> None:
    rows = [check_case(spec) for spec in CASES]
    pooling = check_skewed_pooling()
    output = {"cases": rows, "skewed_pooling": pooling,
              "maximum_statevector_absolute_error": max(
                  row["statevector_absolute_error"] for row in rows),
              "maximum_aer_density_matrix_absolute_error": max(
                  row.get("aer_density_matrix_absolute_error", 0.0)
                  for row in rows)}
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({"maximum_statevector_absolute_error":
                      output["maximum_statevector_absolute_error"],
                      "maximum_aer_density_matrix_absolute_error":
                      output["maximum_aer_density_matrix_absolute_error"],
                      "skewed_pooling": pooling,
                      "result": str(RESULT)}, indent=2))


if __name__ == "__main__":
    main()
