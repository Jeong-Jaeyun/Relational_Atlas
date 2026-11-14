"""Preregistered invalid-code phase intervention after cutoff-two graph test."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator
from qiskit_aer.library import SetStatevector

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout24_charge_graph as graph


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout25_invalid_extension.json"
PROTOCOL = "notes/research_protocol25_invalid_phase_extension_2026-10-06.tex"
SPOTS = ((0, 0, 0), (1, 1, 2), (1, 2, 5))
EXTENSIONS = ("additive", "neutral_invalid")


def local_charge_table(cutoff: int, extension: str) -> np.ndarray:
    """Integer charge assigned to every local one-hot register bit word."""
    if extension not in EXTENSIONS:
        raise ValueError(f"unknown extension: {extension}")
    charges = graph.bit_charges(1, cutoff)
    output = np.zeros(1 << (2 * cutoff), dtype=int)
    for word in range(len(output)):
        if extension == "neutral_invalid" and word.bit_count() > 1:
            continue
        output[word] = sum(charge for bit, charge in enumerate(charges)
                           if word & (1 << bit))
    return output


def phase_diagonal(n: int, cutoff: int, angles: np.ndarray,
                   extension: str) -> np.ndarray:
    width = 2 * cutoff
    mask = (1 << width) - 1
    states = np.arange(1 << (width * n), dtype=np.int64)
    local = local_charge_table(cutoff, extension)
    exponent = np.zeros(len(states))
    for axis in range(n):
        exponent += angles[axis] * local[(states >> (width * axis)) & mask]
    return np.exp(1j * exponent)


def classes(cutoff: int) -> tuple[tuple[str, int, int], ...]:
    return (("same_bit", 0, 0),
            ("conjugate", 0, cutoff),
            ("other_same_sign", 0, 2 * cutoff),
            ("other_opposite_sign", 0, 3 * cutoff))


def analytic_grid(n: int, cutoff: int) -> dict[str, dict[str, np.ndarray]]:
    prepared, reference, _, _ = graph.weights_and_vectors(n, cutoff)
    clock = graph.clock_bits(n, cutoff)
    labels = 2 * n + 1
    output = {}
    for name, source, return_ in classes(cutoff):
        output[name] = {}
        for extension in EXTENSIONS:
            values = np.zeros((2, n, labels))
            for block, axis, label, angles in graph.settings(n):
                values[block, axis, label], _ = graph.analytic_probability(
                    prepared, reference,
                    phase_diagonal(n, cutoff, angles, extension),
                    source, return_, graph.ANGLE, clock)
            output[name][extension] = values
    return output


def aer_circuit(joint: np.ndarray, n: int, cutoff: int, source: int,
                return_: int, angles: np.ndarray, extension: str) -> QuantumCircuit:
    system = 2 * cutoff * n
    clock = graph.clock_bits(n, cutoff)
    circuit = QuantumCircuit(system + clock)
    circuit.append(SetStatevector(joint), circuit.qubits)
    circuit.rx(graph.ANGLE, source)
    levels = np.arange(1 << clock)
    circuit.append(DiagonalGate(np.exp(1j * graph.TAU * (
        levels + 0.37 * levels * levels / n))), range(system, system + clock))
    circuit.append(DiagonalGate(phase_diagonal(n, cutoff, angles, extension)),
                   range(system))
    circuit.rx(-graph.ANGLE, return_)
    for bit in range(clock):
        circuit.h(system + bit)
    circuit.save_statevector()
    return circuit


def joint_preparation(n: int, cutoff: int) -> tuple[np.ndarray, np.ndarray]:
    system = 2 * cutoff * n
    clock = graph.clock_bits(n, cutoff)
    _, reference, indices, energies = graph.weights_and_vectors(n, cutoff)
    weights = np.exp(-graph.BETA * energies.astype(float))
    weights /= weights.sum()
    joint = np.zeros(1 << (system + clock), dtype=complex)
    joint[indices | (energies << system)] = np.sqrt(weights)
    return joint, reference


def study(workers: int = 6) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    comparisons = {}
    exact_n3_k2 = None
    for cutoff in (1, 2):
        grid = analytic_grid(graph.N, cutoff)
        if cutoff == 2:
            exact_n3_k2 = grid
        comparisons[f"K{cutoff}"] = {}
        for name, pair in grid.items():
            a = pair["additive"]
            b = pair["neutral_invalid"]
            difference = float(np.max(abs(a - b)))
            weights = graph.prior.frozen_label_weights(2 * graph.N + 1)
            delta_a = (a[1] - a[0]) @ weights
            delta_b = (b[1] - b[0]) @ weights
            comparisons[f"K{cutoff}"][name] = {
                "maximum_pointwise_probability_difference": difference,
                "additive_axis_pooled_contrast": delta_a.tolist(),
                "neutral_invalid_axis_pooled_contrast": delta_b.tolist(),
                "maximum_contrast_difference": float(np.max(abs(delta_a - delta_b))),
            }
    for name, record in comparisons["K1"].items():
        if record["maximum_pointwise_probability_difference"] >= 1e-12:
            raise AssertionError(f"K1 phase-extension invariance failed: {name}")
    for name in ("conjugate", "other_same_sign", "other_opposite_sign"):
        if comparisons["K2"][name]["maximum_pointwise_probability_difference"] >= 1e-12:
            raise AssertionError(f"K2 cross-class invariance failed: {name}")
    if comparisons["K2"]["same_bit"]["maximum_pointwise_probability_difference"] <= 1e-9:
        raise AssertionError("K2 same-bit extension sensitivity not observed")

    joint, reference = joint_preparation(graph.N, graph.CUTOFF)
    system = 2 * graph.CUTOFF * graph.N
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    spot_results = []
    largest_aer_error = 0.0
    for name, source, return_ in classes(graph.CUTOFF):
        for extension in EXTENSIONS:
            circuits = []
            for block, axis, label in SPOTS:
                shift = graph.SHIFTS[block]
                angles = np.full(graph.N, shift + 2 * math.pi * label / (2 * graph.N + 1))
                angles[axis] -= graph.BIAS
                circuits.append(aer_circuit(joint, graph.N, graph.CUTOFF,
                                            source, return_, angles, extension))
            result = simulator.run(circuits, shots=1).result()
            if not result.success:
                raise RuntimeError(result.status)
            for index, (block, axis, label) in enumerate(SPOTS):
                state = np.asarray(result.data(index)["statevector"].data)
                observed = float(abs(np.vdot(reference, state[:1 << system])) ** 2)
                predicted = float(exact_n3_k2[name][extension][block, axis, label])
                error = abs(observed - predicted)
                largest_aer_error = max(largest_aer_error, error)
                spot_results.append({"class": name, "extension": extension,
                                     "setting": [block, axis, label],
                                     "analytic": predicted, "aer": observed,
                                     "absolute_error": error})
        print(f"completed {name}", flush=True)
    if largest_aer_error >= graph.TOLERANCE:
        raise AssertionError(f"Aer spot check failed: {largest_aer_error}")

    output = {"protocol": PROTOCOL,
              "parameters": {"n": graph.N, "cutoffs": [1, 2],
                             "beta": graph.BETA,
                             "reference_beta": graph.REFERENCE_BETA,
                             "tau": graph.TAU, "angle": graph.ANGLE,
                             "workers": workers, "spots": SPOTS},
              "local_charge_tables": {
                  f"K{cutoff}": {extension: local_charge_table(
                      cutoff, extension).tolist() for extension in EXTENSIONS}
                  for cutoff in (1, 2)},
              "comparisons": comparisons, "aer_spots": spot_results,
              "maximum_absolute_aer_spot_error": largest_aer_error,
              "all_preregistered_checks_passed": True}
    RESULT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    output = study(args.workers)
    print(json.dumps({"comparisons": output["comparisons"],
                      "maximum_absolute_aer_spot_error":
                      output["maximum_absolute_aer_spot_error"]}, indent=2))


if __name__ == "__main__":
    main()
