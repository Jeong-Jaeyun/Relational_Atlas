"""Pure circuit construction for the stage-6 energy-correlated clock.

The preparation, clock phase, and postselection convention live here so
subsequent experiments can reuse the same physical circuit without copying
the acquisition or hypothesis-test machinery.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import DiagonalGate, StatePreparation

from relational_atlas_i01.phase_reference import qiskit_condition_probe as encoding
from relational_atlas_i01.holdout import qiskit_rank_holdout3_shots as previous


N = previous.N
CLOCK_QUBITS = 2
CLOCK_ACCEPTANCE = 0.25


def template(entry: dict, tau: float, measure: bool = False,
             noise_markers: bool = False
             ) -> tuple[QuantumCircuit, ParameterVector]:
    """Prepare the correlated clock and apply the phase/readout sequence.

    ``noise_markers`` preserves two barriers: immediately after preparation and
    immediately after the scientific phase layer.  They are only for localized
    noise studies and do not alter the stage-6 circuit when left at its default.
    """
    if entry["n"] != N:
        raise ValueError("this protocol freezes n=3")
    system_qubits = 2 * N
    reference = np.zeros(1 << system_qubits, dtype=complex)
    joint = np.zeros(1 << (system_qubits + CLOCK_QUBITS), dtype=complex)
    reference_weights = np.exp(-previous.REFERENCE_BETA * entry["base_energies"])
    reference_weights /= reference_weights.sum()
    for momentum, base_energy, weight, ref_weight in zip(
        entry["momenta"], entry["base_energies"], entry["weights"],
        reference_weights
    ):
        index = encoding.basis_index(momentum)
        reference[index] = math.sqrt(ref_weight)
        joint[index | (int(base_energy) << system_qubits)] = math.sqrt(weight)
    clock_diagonal = np.exp(1j * tau * (
        np.arange(4) + 0.37 * np.arange(4) ** 2 / N))
    theta = ParameterVector("theta", N)
    circuit = QuantumCircuit(system_qubits + CLOCK_QUBITS)
    circuit.append(StatePreparation(joint), range(system_qubits + CLOCK_QUBITS))
    if noise_markers:
        circuit.barrier()
    circuit.append(DiagonalGate(clock_diagonal), range(system_qubits,
                                                       system_qubits + CLOCK_QUBITS))
    for axis in range(N):
        circuit.rz(theta[axis], 2 * axis)
        circuit.rz(-theta[axis], 2 * axis + 1)
    if noise_markers:
        circuit.barrier()
    for bit in range(CLOCK_QUBITS):
        circuit.h(system_qubits + bit)
    circuit.append(StatePreparation(reference).inverse(), range(system_qubits))
    if measure:
        circuit.measure_all()
    return circuit, theta


def bind(circuit: QuantumCircuit, theta: ParameterVector,
         angles: np.ndarray) -> QuantumCircuit:
    """Bind the three scientific phases without mutating the template."""
    return circuit.assign_parameters({theta[i]: float(angles[i]) for i in range(N)})


def exact_joint_probabilities() -> np.ndarray:
    """Joint success probabilities before conditioning on clock acceptance."""
    return previous.exact_probabilities() * CLOCK_ACCEPTANCE
