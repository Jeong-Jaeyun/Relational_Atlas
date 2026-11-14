"""Independent circuit-probability checks for selected predeclared holdouts."""

from __future__ import annotations

import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from relational_atlas_i01.phase_reference import qiskit_condition_probe as encoding
from relational_atlas_i01.holdout import qiskit_rank_holdout_grid as holdout


PROJECT = PROJECT_ROOT
RESULT = PROJECT / "results" / "qiskit_rank_holdout_circuit.json"
CASES = (
    (3, 0.45, 0.55, 0.0, 7, 1, False),
    (3, 1.20, 1.25, math.pi / 2, 5, 2, True),
    (4, 1.20, 1.25, math.pi / 2, 9, 3, False),
    (4, 0.45, 0.55, 0.0, 7, 1, True),
)


def circuit(entry: dict, tau: float, angles: np.ndarray) -> tuple[QuantumCircuit, int]:
    n = entry["n"]
    system_qubits = 2 * n
    clock_levels = n + 1
    clock_qubits = math.ceil(math.log2(clock_levels))
    total_qubits = system_qubits + clock_qubits
    system = np.zeros(1 << system_qubits, dtype=complex)
    joint = np.zeros(1 << total_qubits, dtype=complex)
    for k, energy, weight in zip(entry["momenta"], entry["energies"],
                                 entry["weights"]):
        index = encoding.basis_index(k)
        system[index] = math.sqrt(weight)
        joint[index | (int(energy) << system_qubits)] = math.sqrt(weight)
    clock_uniform = np.zeros(1 << clock_qubits, dtype=complex)
    clock_uniform[:clock_levels] = 1 / math.sqrt(clock_levels)
    qc = QuantumCircuit(total_qubits)
    qc.append(StatePreparation(joint), range(total_qubits))
    for bit in range(clock_qubits):
        qc.rz((1 << bit) * tau, system_qubits + bit)
    for axis, angle in enumerate(angles):
        qc.rz(angle, 2 * axis)
        qc.rz(-angle, 2 * axis + 1)
    qc.append(StatePreparation(clock_uniform).inverse(),
              range(system_qubits, total_qubits))
    qc.append(StatePreparation(system).inverse(), range(system_qubits))
    return qc, system_qubits


def check_case(spec: tuple) -> dict:
    n, beta, bias, tau, labels, label_index, physical = spec
    entry = holdout.model(n, beta, physical)
    angles = np.full(n, 2 * math.pi * label_index / labels)
    angles[0] -= bias
    conditional, _ = holdout.probability_gradient(entry, tau, angles)
    qc, system_qubits = circuit(entry, tau, angles)
    state = Statevector.from_instruction(qc).data
    accepted = float(np.sum(np.abs(state[:1 << system_qubits]) ** 2))
    joint = float(abs(state[0]) ** 2)
    target_clock = 1 / (n + 1)
    target_joint = conditional / (n + 1)
    if not np.isclose(accepted, target_clock, atol=1e-10):
        raise AssertionError("statevector clock acceptance mismatch")
    if not np.isclose(joint, target_joint, atol=1e-10):
        raise AssertionError("statevector joint probability mismatch")
    row = {"n": n, "beta": beta, "bias": bias, "tau_over_pi": tau / math.pi,
           "phase_labels": labels, "label_index": label_index,
           "physical_P_zero": physical,
           "analytic_conditional_probability": conditional,
           "statevector_clock_success_probability": accepted,
           "statevector_joint_zero_probability": joint,
           "analytic_joint_zero_probability": target_joint,
           "statevector_absolute_error": abs(joint - target_joint)}
    if n == 3:
        compiled = transpile(qc, basis_gates=["u3", "cx"],
                             optimization_level=1)
        compiled.save_probabilities()
        simulator = AerSimulator(method="density_matrix", max_parallel_threads=2,
                                 max_parallel_experiments=1)
        result = simulator.run(compiled, shots=1, seed_simulator=20261012).result()
        if not result.success:
            raise RuntimeError(result.status)
        probabilities = np.asarray(result.data(0)["probabilities"], dtype=float)
        density_clock = float(np.sum(probabilities[:1 << system_qubits]))
        density_joint = float(probabilities[0])
        if not np.isclose(density_clock, target_clock, atol=1e-10):
            raise AssertionError("Aer density-matrix clock acceptance mismatch")
        if not np.isclose(density_joint, target_joint, atol=1e-10):
            raise AssertionError("Aer density-matrix joint probability mismatch")
        row.update({"aer_density_matrix_clock_success_probability": density_clock,
                    "aer_density_matrix_joint_zero_probability": density_joint,
                    "aer_density_matrix_absolute_error": abs(
                        density_joint - target_joint),
                    "transpiled_depth": compiled.depth()})
    return row


def main() -> None:
    rows = [check_case(spec) for spec in CASES]
    output = {"cases_predeclared": [list(spec) for spec in CASES],
              "rows": rows,
              "max_statevector_absolute_error": max(
                  row["statevector_absolute_error"] for row in rows),
              "max_aer_density_matrix_absolute_error": max(
                  row.get("aer_density_matrix_absolute_error", 0.0) for row in rows)}
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({"max_statevector_absolute_error":
                      output["max_statevector_absolute_error"],
                      "max_aer_density_matrix_absolute_error":
                      output["max_aer_density_matrix_absolute_error"],
                      "result": str(RESULT)}, indent=2))


if __name__ == "__main__":
    main()
