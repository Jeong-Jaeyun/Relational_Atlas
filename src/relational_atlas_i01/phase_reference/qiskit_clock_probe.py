"""Finite energy-matched Page--Wootters-style clock for the I01 cutoff-1 toy.

This explicitly prepares an H_C+H_S=0 state. It tests conditional dynamics
and postselection statistics; it does not derive the constraint or cosmic time.
"""

from __future__ import annotations

import argparse
import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from relational_atlas_i01.phase_reference import qiskit_condition_probe as probe


PROJECT = PROJECT_ROOT
RESULT = PROJECT / "results" / "qiskit_clock_probe.json"
PHASES = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi)


def clock_model() -> dict:
    system = probe.model(4, True)
    energies = np.sum(system["momenta"] ** 2, axis=1).astype(int)
    assert set(energies) == {0, 2, 4}
    amplitudes = np.zeros(1 << 10, dtype=complex)
    for k, weight, energy in zip(system["momenta"], system["weights"], energies):
        clock_code = energy // 2
        amplitudes[probe.basis_index(k) | (clock_code << 8)] = np.sqrt(weight)
    clock_uniform = np.array([1, 1, 1, 0], dtype=complex) / np.sqrt(3)
    return {
        "system": system,
        "energies": energies,
        "amplitudes": amplitudes,
        "joint_preparation": StatePreparation(amplitudes),
        "clock_uniform_inverse": StatePreparation(clock_uniform).inverse(),
        "constraint_residual": float(np.sqrt(np.sum(system["weights"] *
                                                   (energies - 2 * (energies // 2)) ** 2))),
    }


def circuit(entry: dict, tau: float | Parameter, measure: bool = False) -> QuantumCircuit:
    qc = QuantumCircuit(10)
    qc.append(entry["joint_preparation"], range(10))
    qc.rz(2 * tau, 8)
    qc.rz(4 * tau, 9)
    qc.append(entry["clock_uniform_inverse"], [8, 9])
    qc.append(entry["system"]["preparation"].inverse(), range(8))
    if measure:
        qc.measure_all()
    return qc


def expected_fidelity(entry: dict, tau: float) -> float:
    return float(abs(np.sum(entry["system"]["weights"] *
                            np.exp(1j * entry["energies"] * tau))) ** 2)


def exact_study(entry: dict) -> list[dict]:
    rows = []
    for tau in PHASES:
        state = Statevector.from_instruction(circuit(entry, tau)).data
        clock_probability = float(np.sum(np.abs(state[:256]) ** 2))
        joint_probability = float(abs(state[0]) ** 2)
        conditional = state[:256] / np.sqrt(clock_probability)
        target = entry["system"]["amplitudes"].copy()
        for k, energy in zip(entry["system"]["momenta"], entry["energies"]):
            target[probe.basis_index(k)] *= np.exp(1j * energy * tau)
        unprepared_target = Statevector(target).evolve(
            entry["system"]["preparation"].inverse()).data
        state_match = float(abs(np.vdot(unprepared_target, conditional)) ** 2)
        predicted = expected_fidelity(entry, tau)
        assert np.isclose(clock_probability, 1 / 3, atol=1e-11)
        assert np.isclose(joint_probability / clock_probability, predicted, atol=1e-11)
        assert np.isclose(state_match, 1.0, atol=1e-11)
        rows.append({
            "tau_over_pi": float(tau / math.pi),
            "clock_postselection_probability": clock_probability,
            "conditional_return_fidelity": joint_probability / clock_probability,
            "direct_formula_fidelity": predicted,
            "conditional_state_match_fidelity": state_match,
        })
    return rows


def shot_study(entry: dict, shots: int, workers: int, seed: int) -> dict:
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    tau_param = Parameter("tau")
    parameterized = circuit(entry, tau_param, measure=True)
    # The Aer 0.17.2 Windows assembler crashes on the synthesis multiplexer.
    compiled = transpile(parameterized, basis_gates=["u3", "cx"], optimization_level=1)
    circuits = [compiled.assign_parameters({tau_param: tau}) for tau in PHASES]
    result = simulator.run(circuits, shots=shots, seed_simulator=seed).result()
    if not result.success:
        raise RuntimeError(result.status)
    rows = []
    for index, tau in enumerate(PHASES):
        counts = result.get_counts(index)
        clock_success = sum(count for bits, count in counts.items() if bits[:2] == "00")
        joint_success = counts.get("0" * 10, 0)
        rows.append({
            "tau_over_pi": float(tau / math.pi),
            "clock_success_count": clock_success,
            "joint_success_count": joint_success,
            "clock_postselection_probability": clock_success / shots,
            "conditional_return_fidelity": joint_success / clock_success,
            "exact_conditional_return_fidelity": expected_fidelity(entry, tau),
        })
    return {
        "shots_per_circuit": shots,
        "transpiled_depth": compiled.depth(),
        "transpiled_size": compiled.size(),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=16384)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    if args.shots < 1 or args.workers < 1:
        parser.error("shots and workers must be positive")
    entry = clock_model()
    unrestricted = probe.model(4, False)
    unrestricted_energies = np.sum(unrestricted["momenta"] ** 2, axis=1)
    unrestricted_pi_fidelity = float(abs(np.sum(
        unrestricted["weights"] * np.exp(1j * unrestricted_energies * math.pi))) ** 2)
    output = {
        "model": "n=4, cutoff=1, P=0, H_S=sum k_i^2, H_C=-2m for clock code m=0,1,2",
        "interpretation": "The energy constraint is built into the prepared state; the clock is a finite reference, not a derived cosmic time.",
        "constraint_residual": entry["constraint_residual"],
        "clock_energies": [0, -2, -4],
        "unrestricted_system_fidelity_at_pi_analytic_control": unrestricted_pi_fidelity,
        "phase_points_tau_over_pi": [float(t / math.pi) for t in PHASES],
        "exact": exact_study(entry),
        "shots": shot_study(entry, args.shots, args.workers, args.seed),
    }
    RESULT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "constraint_residual": output["constraint_residual"],
        "exact_fidelity": [row["conditional_return_fidelity"] for row in output["exact"]],
        "shot_fidelity": [row["conditional_return_fidelity"] for row in output["shots"]["rows"]],
        "result": str(RESULT),
    }, indent=2))


if __name__ == "__main__":
    main()
