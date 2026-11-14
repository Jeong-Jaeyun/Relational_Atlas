"""Preregistered cutoff-two check of the finite-support charge-path rule.

The sparse amplitude calculation and the Aer evolution deliberately use
different representations. See notes/research_protocol24_charge_graph_cutoff2_2026-10-06.tex
for frozen parameters, predicted counts, and decision thresholds.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator
from qiskit_aer.library import SetStatevector

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout15_dimension_transfer as prior


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout24_charge_graph.json"
PROTOCOL = "notes/research_protocol24_charge_graph_cutoff2_2026-10-06.tex"
CUTOFF = 2
N = 3
BETA = 0.45
REFERENCE_BETA = 0.15
TAU = math.pi / 2
ANGLE = 0.83
SHIFTS = (0.0, 0.60)
BIAS = 1.25
TOLERANCE = 1e-11
CLASSES = (
    ("same_bit", 0, 0),
    ("conjugate", 0, 2),
    ("other_same_sign", 0, 4),
    ("other_opposite_sign", 0, 6),
)
FROZEN_COUNTS = {
    (1, 2): (3, 0, 0, 2), (1, 3): (7, 0, 2, 2),
    (1, 4): (19, 0, 4, 6), (2, 2): (5, 0, 0, 2),
    (2, 3): (19, 0, 2, 2), (2, 4): (85, 0, 8, 10),
}


def basis_index(momentum: tuple[int, ...], cutoff: int) -> int:
    """One-hot code: per axis +1,...,+K,-1,...,-K."""
    index = 0
    width = 2 * cutoff
    for axis, value in enumerate(momentum):
        if not -cutoff <= value <= cutoff:
            raise ValueError("momentum exceeds cutoff")
        if value:
            local = (value - 1) if value > 0 else cutoff - value - 1
            index |= 1 << (width * axis + local)
    return index


def support(n: int, cutoff: int) -> tuple[np.ndarray, np.ndarray]:
    """Return P=0 encoded indices and base energies, without Qiskit."""
    momenta = (values for values in itertools.product(
        range(-cutoff, cutoff + 1), repeat=n) if sum(values) == 0)
    pairs = [(basis_index(values, cutoff), sum(x * x for x in values))
             for values in momenta]
    return (np.asarray([index for index, _ in pairs], dtype=np.int64),
            np.asarray([energy for _, energy in pairs], dtype=np.int64))


def coefficient_count(sites: int, charge: int, cutoff: int) -> int:
    """Coefficient [z^charge](sum_{-K}^K z^j)^sites by convolution."""
    counts = {0: 1}
    for _ in range(sites):
        next_counts: dict[int, int] = {}
        for value, mass in counts.items():
            for step in range(-cutoff, cutoff + 1):
                next_counts[value + step] = next_counts.get(value + step, 0) + mass
        counts = next_counts
    return counts.get(charge, 0)


def predicted_counts(n: int, cutoff: int) -> tuple[int, int, int, int]:
    return (coefficient_count(n, 0, cutoff), 0,
            2 * coefficient_count(n - 2, -1, cutoff),
            2 * coefficient_count(n - 2, 0, cutoff))


def direct_graph_counts(n: int, cutoff: int) -> tuple[dict[str, int], int]:
    """Enumerate encoded basis graph paths independently of coefficients."""
    indices, _ = support(n, cutoff)
    allowed = set(indices.tolist())
    classes = (
        ("same_bit", 0, 0),
        ("conjugate", 0, cutoff),
        ("other_same_sign", 0, 2 * cutoff),
        ("other_opposite_sign", 0, 3 * cutoff),
    )
    paths = {name: sum((index ^ (1 << source) ^ (1 << return_)) in allowed
                       for index in indices)
             for name, source, return_ in classes}
    largest_odd = max(sum((index ^ (1 << bit)) in allowed for index in indices)
                      for bit in (0, cutoff, 2 * cutoff, 3 * cutoff))
    return paths, largest_odd


def weights_and_vectors(n: int = N, cutoff: int = CUTOFF
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices, energies = support(n, cutoff)
    size = 1 << (2 * cutoff * n)
    prepared = np.zeros(size, dtype=complex)
    reference = np.zeros(size, dtype=complex)
    first = np.exp(-BETA * energies.astype(float))
    second = np.exp(-REFERENCE_BETA * energies.astype(float))
    first /= first.sum()
    second /= second.sum()
    prepared[indices] = np.sqrt(first) * np.exp(
        1j * TAU * (energies + 0.37 * energies * energies / n))
    reference[indices] = np.sqrt(second)
    return prepared, reference, indices, energies


def clock_bits(n: int, cutoff: int) -> int:
    return max(1, math.ceil(math.log2(n * cutoff * cutoff + 1)))


def bit_charges(n: int, cutoff: int) -> np.ndarray:
    local = tuple(range(1, cutoff + 1)) + tuple(
        range(-1, -cutoff - 1, -1))
    return np.tile(np.asarray(local, dtype=int), n)


def phase_diagonal(n: int, cutoff: int, angles: np.ndarray) -> np.ndarray:
    """Additive charge extension also specifies phases on invalid multi-hot words."""
    size = 1 << (2 * cutoff * n)
    states = np.arange(size, dtype=np.int64)
    charges = bit_charges(n, cutoff)
    phase = np.zeros(size)
    for bit, charge in enumerate(charges):
        phase += ((states >> bit) & 1) * charge * angles[bit // (2 * cutoff)]
    return np.exp(1j * phase)


def analytic_probability(prepared: np.ndarray, reference: np.ndarray,
                         phases: np.ndarray, source: int, return_: int,
                         angle: float, clock: int) -> tuple[float, dict]:
    indices = np.flatnonzero(prepared)
    source_indices = indices ^ (1 << source)
    returned_indices = source_indices ^ (1 << return_)
    direct = np.vdot(reference[indices], phases[indices] * prepared[indices])
    transfer = np.sum(prepared[indices] * np.conj(reference[returned_indices]) *
                      phases[source_indices])
    c2 = math.cos(angle / 2) ** 2
    s2 = math.sin(angle / 2) ** 2
    amplitude = c2 * direct + s2 * transfer
    return float(abs(amplitude) ** 2 / (1 << clock)), {
        "direct": direct, "transfer": transfer,
        "paths": int(np.count_nonzero(reference[returned_indices])),
    }


def aer_circuit(joint: np.ndarray, n: int, cutoff: int, source: int,
                return_: int, angles: np.ndarray, angle: float) -> QuantumCircuit:
    """Independent gate evolution from a known input state, without inverse prep."""
    system = 2 * cutoff * n
    clock = clock_bits(n, cutoff)
    circuit = QuantumCircuit(system + clock)
    circuit.append(SetStatevector(joint), circuit.qubits)
    circuit.rx(angle, source)
    levels = np.arange(1 << clock)
    clock_phase = np.exp(1j * TAU * (levels + 0.37 * levels * levels / n))
    circuit.append(DiagonalGate(clock_phase), range(system, system + clock))
    for bit, charge in enumerate(bit_charges(n, cutoff)):
        circuit.rz(float(charge * angles[bit // (2 * cutoff)]), bit)
    circuit.rx(-angle, return_)
    for bit in range(clock):
        circuit.h(system + bit)
    circuit.save_statevector()
    return circuit


def settings(n: int):
    labels = 2 * n + 1
    for block, shift in enumerate(SHIFTS):
        for axis in range(n):
            for label in range(labels):
                angles = np.full(n, shift + 2 * math.pi * label / labels)
                angles[axis] -= BIAS
                yield block, axis, label, angles


def study(workers: int = 6, batch_size: int = 6) -> dict:
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch_size must be positive")
    graph = {}
    for cutoff in (1, 2):
        for n in (2, 3, 4):
            observed, odd = direct_graph_counts(n, cutoff)
            expected = predicted_counts(n, cutoff)
            frozen = FROZEN_COUNTS[(cutoff, n)]
            if tuple(observed.values()) != expected or expected != frozen or odd:
                raise AssertionError(f"charge graph failed for K={cutoff}, n={n}")
            graph[f"K{cutoff}_n{n}"] = {"enumerated": observed,
                                         "coefficient_prediction": list(expected),
                                         "largest_odd_path_count": odd}

    n, cutoff = N, CUTOFF
    system = 2 * cutoff * n
    clock = clock_bits(n, cutoff)
    prepared, reference, indices, energies = weights_and_vectors(n, cutoff)
    first = np.exp(-BETA * energies.astype(float))
    first /= first.sum()
    joint = np.zeros(1 << (system + clock), dtype=complex)
    joint[indices | (energies << system)] = np.sqrt(first)
    labels = 2 * n + 1
    label_weights = prior.frozen_label_weights(labels)
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    grid = list(settings(n))
    rows = {}
    largest_error = 0.0
    for name, source, return_ in CLASSES:
        exact = np.zeros((2, n, labels))
        independent = np.zeros_like(exact)
        for start in range(0, len(grid), batch_size):
            chunk = grid[start:start + batch_size]
            circuits = []
            for block, axis, label, angles in chunk:
                exact[block, axis, label], _ = analytic_probability(
                    prepared, reference, phase_diagonal(n, cutoff, angles),
                    source, return_, ANGLE, clock)
                circuits.append(aer_circuit(joint, n, cutoff, source,
                                            return_, angles, ANGLE))
            result = simulator.run(circuits, shots=1).result()
            if not result.success:
                raise RuntimeError(result.status)
            for index, (block, axis, label, _) in enumerate(chunk):
                state = np.asarray(result.data(index)["statevector"].data)
                clock_zero = state[:1 << system]
                independent[block, axis, label] = abs(
                    np.vdot(reference, clock_zero)) ** 2
            error = max(abs(exact[block, axis, label] -
                            independent[block, axis, label])
                        for block, axis, label, _ in chunk)
            largest_error = max(largest_error, error)
        exact_delta = (exact[1] - exact[0]) @ label_weights
        aer_delta = (independent[1] - independent[0]) @ label_weights
        rows[name] = {
            "source": source, "return": return_,
            "analytic_joint_probabilities": exact.tolist(),
            "aer_joint_probabilities": independent.tolist(),
            "maximum_absolute_probability_error": float(np.max(abs(exact - independent))),
            "analytic_axis_pooled_contrast": exact_delta.tolist(),
            "aer_axis_pooled_contrast": aer_delta.tolist(),
            "maximum_absolute_analytic_contrast": float(np.max(abs(exact_delta))),
        }
        print(f"completed {name}, max error {rows[name]['maximum_absolute_probability_error']:.3e}",
              flush=True)
    if largest_error >= TOLERANCE:
        raise AssertionError(f"Aer prediction failed: {largest_error}")
    if rows["conjugate"]["maximum_absolute_analytic_contrast"] >= TOLERANCE:
        raise AssertionError("conjugate contrast is nonzero")
    output = {
        "protocol": PROTOCOL,
        "parameters": {"cutoff": cutoff, "n": n, "beta": BETA,
                       "reference_beta": REFERENCE_BETA, "tau": TAU,
                       "angle": ANGLE, "shifts": SHIFTS, "bias": BIAS,
                       "labels": labels, "label_weights": label_weights.tolist(),
                       "clock_bits": clock, "workers": workers,
                       "phase_extension": "additive one-hot bit charge"},
        "graph": graph, "rows": rows,
        "maximum_absolute_aer_probability_error": largest_error,
        "all_preregistered_checks_passed": True,
    }
    RESULT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=6)
    args = parser.parse_args()
    output = study(args.workers, args.batch_size)
    print(json.dumps({"graph": output["graph"],
                      "summary": {name: {"contrast": row["analytic_axis_pooled_contrast"],
                                          "error": row["maximum_absolute_probability_error"]}
                                  for name, row in output["rows"].items()}}, indent=2))


if __name__ == "__main__":
    main()
