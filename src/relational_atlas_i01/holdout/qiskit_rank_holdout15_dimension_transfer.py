"""Exact Aer dimension scan of the paired transverse support-transfer rule."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import DiagonalGate, StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout2_grid as grid
from relational_atlas_i01.holdout import qiskit_rank_holdout3_shots as previous
from relational_atlas_i01.holdout import qiskit_rank_holdout6_clock as stage6
from relational_atlas_i01.holdout import qiskit_rank_holdout14_transfer_matrix as stage14
from relational_atlas_i01.phase_reference import qiskit_condition_probe as encoding


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout15_dimension_transfer.json"
PROTOCOL = "notes/prevalidation_protocol15_2026-10-04.tex"
DIMENSIONS = (2, 3, 4)
ANGLE = 1.0
TAU = 0.0
SHIFTS = (0.0, 0.60)
SUPPORTS = (
    ("same_support", 0, 0),
    ("conjugate_partner", 0, 1),
    ("different_pair_same_sign", 0, 2),
    ("different_pair_opposite_sign", 0, 3),
)


def clock_qubits(n: int) -> int:
    """Number of clock bits needed to encode base energies 0 through n."""
    if n < 1:
        raise ValueError("n must be positive")
    return max(1, math.ceil(math.log2(n + 1)))


def frozen_label_weights(labels: int) -> np.ndarray:
    """Generalize the stage-6 largest-remainder skewed shot allocation."""
    total = labels * stage6.SHOTS_PER_LABEL
    target = grid.label_weights(labels, "skewed") * total
    allocation = np.floor(target).astype(int)
    missing = total - int(allocation.sum())
    order = np.argsort(-(target - allocation), kind="stable")
    allocation[order[:missing]] += 1
    return allocation / allocation.sum()


def template(n: int, transpile_circuit: bool = True
             ) -> tuple[QuantumCircuit, ParameterVector, int]:
    """Construct the stage-6 energy-correlated clock circuit at arbitrary n."""
    entry = grid.model(n, previous.BETA, True, previous.SPECTRUM)
    system = 2 * n
    clock = clock_qubits(n)
    reference = np.zeros(1 << system, dtype=complex)
    joint = np.zeros(1 << (system + clock), dtype=complex)
    ref_weights = np.exp(-previous.REFERENCE_BETA * entry["base_energies"])
    ref_weights /= ref_weights.sum()
    for momentum, energy, weight, ref_weight in zip(
        entry["momenta"], entry["base_energies"], entry["weights"], ref_weights
    ):
        index = encoding.basis_index(momentum)
        reference[index] = math.sqrt(ref_weight)
        joint[index | (int(energy) << system)] = math.sqrt(weight)
    clock_phase = np.exp(1j * TAU * (
        np.arange(1 << clock) + 0.37 * np.arange(1 << clock) ** 2 / n))
    theta = ParameterVector("theta", n)
    circuit = QuantumCircuit(system + clock)
    circuit.append(StatePreparation(joint), range(system + clock))
    circuit.barrier()
    circuit.append(DiagonalGate(clock_phase), range(system, system + clock))
    for axis in range(n):
        circuit.rz(theta[axis], 2 * axis)
        circuit.rz(-theta[axis], 2 * axis + 1)
    circuit.barrier()
    for bit in range(clock):
        circuit.h(system + bit)
    circuit.append(StatePreparation(reference).inverse(), range(system))
    if transpile_circuit:
        circuit = transpile(circuit, basis_gates=["u3", "cx"], optimization_level=1)
    return circuit, theta, clock


def insert_paired_rx(circuit: QuantumCircuit, source: int, return_: int,
                     angle: float = ANGLE) -> QuantumCircuit:
    """Place a transverse source-return pair around the science phase layer."""
    system = circuit.num_qubits - clock_qubits((circuit.num_qubits - 1) // 2)
    if not 0 <= source < system or not 0 <= return_ < system:
        raise ValueError("support must be a system qubit")
    output = circuit.copy_empty_like()
    barriers = 0
    for item in circuit.data:
        output.append(item.operation, item.qubits, item.clbits)
        if item.operation.name == "barrier":
            barriers += 1
            if barriers == 1:
                output.rx(angle, output.qubits[source])
            elif barriers == 2:
                output.rx(-angle, output.qubits[return_])
    if barriers != 2:
        raise ValueError("missing science barriers")
    return output


def probabilities(n: int, source: int, return_: int, workers: int,
                  prepared: tuple[QuantumCircuit, ParameterVector, int] | None = None,
                  angle: float = ANGLE
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Return block-by-axis-by-label joint and clock probabilities exactly."""
    base, theta, clock = template(n) if prepared is None else prepared
    labels = 2 * n + 1
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    joint = np.zeros((2, n, labels))
    acceptance = np.zeros_like(joint)
    circuits, settings = [], []
    for block, shift in enumerate(SHIFTS):
        for axis in range(n):
            for label in range(labels):
                phases = np.full(n, shift + 2 * math.pi * label / labels)
                phases[axis] -= grid.BIAS
                circuit = base.assign_parameters({theta[i]: float(phases[i]) for i in range(n)})
                circuit = insert_paired_rx(circuit, source, return_, angle)
                circuit.save_probabilities()
                circuits.append(circuit)
                settings.append((block, axis, label))
    result = simulator.run(circuits, shots=1).result()
    if not result.success:
        raise RuntimeError(result.status)
    system = 2 * n
    for index, setting in enumerate(settings):
        values = np.asarray(result.data(index)["probabilities"], dtype=float)
        joint[setting] = values[0]
        acceptance[setting] = values[:1 << system].sum()
    if np.max(np.abs(acceptance - 1 / (1 << clock))) > 1e-10:
        raise AssertionError("clock acceptance changed under a system-only intervention")
    return joint, acceptance


def statevector_probabilities(n: int, source: int, return_: int,
                              angle: float = ANGLE) -> tuple[np.ndarray, np.ndarray]:
    """Independent raw-Qiskit statevector crosscheck without gate decomposition."""
    base, theta, clock = template(n, transpile_circuit=False)
    labels = 2 * n + 1
    joint = np.zeros((2, n, labels))
    acceptance = np.zeros_like(joint)
    for block, shift in enumerate(SHIFTS):
        for axis in range(n):
            for label in range(labels):
                phases = np.full(n, shift + 2 * math.pi * label / labels)
                phases[axis] -= grid.BIAS
                circuit = base.assign_parameters({theta[i]: float(phases[i]) for i in range(n)})
                circuit = insert_paired_rx(circuit, source, return_, angle)
                values = Statevector.from_instruction(circuit).probabilities()
                joint[block, axis, label] = values[0]
                acceptance[block, axis, label] = values[:1 << (2 * n)].sum()
    if np.max(np.abs(acceptance - 1 / (1 << clock))) > 1e-10:
        raise AssertionError("statevector clock acceptance changed")
    return joint, acceptance


def circuit_checks() -> dict:
    """Verify generic dimensions, clock widths, and paired support placement."""
    output = {}
    for n in DIMENSIONS:
        base, _, clock = template(n)
        circuit = insert_paired_rx(base, 0, 1, 0.1)
        data = list(circuit.data)
        barriers = [i for i, item in enumerate(data) if item.operation.name == "barrier"]
        rotations = [i for i, item in enumerate(data) if item.operation.name == "rx"]
        if rotations != [barriers[0] + 1, barriers[1] + 1]:
            raise AssertionError(f"invalid generic intervention placement at n={n}")
        output[str(n)] = {"total_qubits": circuit.num_qubits, "clock_qubits": clock,
                          "rotation_indices": rotations}
    return output


def study(workers: int) -> dict:
    """Compare the four frozen transfer classes at n=2, 3, and 4."""
    if workers < 1:
        raise ValueError("workers must be positive")
    rows = []
    for n in DIMENSIONS:
        prepared = template(n)
        labels = 2 * n + 1
        weights = frozen_label_weights(labels)
        for name, source, return_ in SUPPORTS:
            joint, acceptance = probabilities(n, source, return_, workers, prepared)
            delta = (joint[1] - joint[0]) @ weights
            rows.append({
                "n": n,
                "phase_labels": labels,
                "support_class": name,
                "source_qubit": source,
                "return_qubit": return_,
                "axis_pooled_joint_difference": delta.tolist(),
                "maximum_absolute_joint_difference": float(np.max(np.abs(delta))),
                "maximum_clock_acceptance_error": float(np.max(np.abs(
                    acceptance - 1 / (1 << prepared[2])))),
            })
    n3 = {row["support_class"]: row["maximum_absolute_joint_difference"]
          for row in rows if row["n"] == 3}
    n3_error = None
    if stage14.RESULT.exists():
        reference = np.asarray(json.loads(stage14.RESULT.read_text(encoding="utf-8"))[
            "constraint_tau_0_maximum_contrast_matrix"])
        expected_n3 = {
            "same_support": reference[0, 0],
            "conjugate_partner": reference[0, 1],
            "different_pair_same_sign": reference[0, 2],
            "different_pair_opposite_sign": reference[0, 3],
        }
        n3_error = max(abs(n3[name] - value) for name, value in expected_n3.items())
    return {
        "protocol": PROTOCOL,
        "parameters": {"workers": workers, "dimensions": list(DIMENSIONS),
                       "angle_radians": ANGLE, "tau_over_pi": TAU / math.pi,
                       "label_distribution": "skewed_largest_remainder", "all_axes_explicitly_evaluated": True},
        "circuit_checks": circuit_checks(),
        "rows": rows,
        "n3_reference_reproduction_maximum_error": (
            None if n3_error is None else float(n3_error)),
        "caveat": "Exact Aer comparison of a generalized toy clock; it is not finite-shot evidence or a physical dimension-selection mechanism.",
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
    print(json.dumps({"n3_reference_error": output["n3_reference_reproduction_maximum_error"],
                      "rows": output["rows"]}, indent=2))


if __name__ == "__main__":
    main()
