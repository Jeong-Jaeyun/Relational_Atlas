"""Code-preserving local access: full-grid sparse/Aer crosscheck.

Predictions and thresholds are frozen in
notes/research_protocol26_physical_access_2026-10-15.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, UnitaryGate
from qiskit_aer import AerSimulator
from qiskit_aer.library import SetStatevector

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout24_charge_graph as graph
from relational_atlas_i01.holdout import qiskit_rank_holdout25_invalid_extension as extension


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout26_physical_access.json"
PROTOCOL = "notes/research_protocol26_physical_access_2026-10-15.md"
FROZEN_COUNTS = {
    2: (2, 0, 0, 2),
    3: (9, 0, 2, 2),
    4: (37, 0, 8, 10),
}


def active(index: int, bit: int, cutoff: int) -> bool:
    """Whether |index> is in the local 0--target two-level subspace."""
    width = 2 * cutoff
    axis, local_bit = divmod(bit, width)
    word = (index >> (width * axis)) & ((1 << width) - 1)
    return word in (0, 1 << local_bit)


def predicted_counts(n: int, cutoff: int) -> tuple[int, int, int, int]:
    t = graph.coefficient_count
    return (t(n - 1, 0, cutoff) + t(n - 1, -1, cutoff),
            0, 2 * t(n - 2, -1, cutoff), 2 * t(n - 2, 0, cutoff))


def enumerate_counts(n: int, cutoff: int) -> tuple[dict[str, int], int]:
    """Independently enumerate two-hop paths and one-hop return paths."""
    indices, _ = graph.support(n, cutoff)
    allowed = set(indices.tolist())
    counts = {}
    odd = 0
    for name, source, return_ in extension.classes(cutoff):
        paths = 0
        for raw in indices:
            index = int(raw)
            if active(index, source, cutoff):
                middle = index ^ (1 << source)
                odd += middle in allowed
                if active(middle, return_, cutoff):
                    final = middle ^ (1 << return_)
                    paths += final in allowed
            if active(index, return_, cutoff):
                odd += (index ^ (1 << return_)) in allowed
        counts[name] = paths
    return counts, odd


def analytic_probability(prepared: np.ndarray, reference: np.ndarray,
                         phases: np.ndarray, source: int, return_: int,
                         cutoff: int, clock: int,
                         angle: float = graph.ANGLE) -> tuple[float, dict]:
    """Independent sparse formula using D=I+(cos(e/2)-1)Q and H hops."""
    c = math.cos(angle / 2)
    s = math.sin(angle / 2)
    diagonal = 0j
    transfer = 0j
    transfer_paths = 0
    for raw in np.flatnonzero(prepared):
        index = int(raw)
        source_active = active(index, source, cutoff)
        return_active = active(index, return_, cutoff)
        diagonal += (np.conj(reference[index]) * prepared[index] * phases[index]
                     * (c if source_active else 1)
                     * (c if return_active else 1))
        if source_active:
            middle = index ^ (1 << source)
            if active(middle, return_, cutoff):
                final = middle ^ (1 << return_)
                if reference[final] != 0:
                    transfer_paths += 1
                    transfer += (np.conj(reference[final]) * prepared[index]
                                 * phases[middle])
    amplitude = diagonal + s * s * transfer
    return float(abs(amplitude) ** 2 / (1 << clock)), {
        "diagonal": diagonal, "transfer": transfer,
        "transfer_paths": transfer_paths,
    }


def code_preserving_rx(circuit: QuantumCircuit, bit: int, cutoff: int,
                       angle: float) -> None:
    """Exact local unitary: rotate 0 <-> one-hot target, fix all others."""
    width = 2 * cutoff
    axis, local = divmod(bit, width)
    target = 1 << local
    matrix = np.eye(1 << width, dtype=complex)
    c = math.cos(angle / 2)
    off_diagonal = -1j * math.sin(angle / 2)
    matrix[0, 0] = c
    matrix[target, target] = c
    matrix[0, target] = off_diagonal
    matrix[target, 0] = off_diagonal
    circuit.append(UnitaryGate(matrix), range(axis * width, (axis + 1) * width))


def aer_circuit(joint: np.ndarray, n: int, cutoff: int, source: int,
                return_: int, angles: np.ndarray, phase_extension: str
                ) -> QuantumCircuit:
    system = 2 * cutoff * n
    clock = graph.clock_bits(n, cutoff)
    circuit = QuantumCircuit(system + clock)
    circuit.append(SetStatevector(joint), circuit.qubits)
    code_preserving_rx(circuit, source, cutoff, graph.ANGLE)
    circuit.save_statevector(label="after_source")
    levels = np.arange(1 << clock)
    circuit.append(DiagonalGate(np.exp(1j * graph.TAU * (
        levels + 0.37 * levels * levels / n))), range(system, system + clock))
    circuit.append(DiagonalGate(extension.phase_diagonal(
        n, cutoff, angles, phase_extension)), range(system))
    code_preserving_rx(circuit, return_, cutoff, -graph.ANGLE)
    for bit in range(clock):
        circuit.h(system + bit)
    circuit.save_statevector(label="final")
    return circuit


def valid_code_indices(n: int, cutoff: int) -> np.ndarray:
    values = range(-cutoff, cutoff + 1)
    return np.asarray([graph.basis_index(momentum, cutoff)
                       for momentum in itertools.product(values, repeat=n)],
                      dtype=np.int64)


def invalid_mass(state: np.ndarray, valid: np.ndarray, system: int) -> float:
    by_clock = np.asarray(state).reshape(-1, 1 << system)
    remaining = 1.0 - float(np.sum(abs(by_clock[:, valid]) ** 2))
    return max(0.0, remaining)


def study(workers: int = 6, batch_size: int = 6) -> dict:
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch_size must be positive")
    counts = {}
    for n, frozen in FROZEN_COUNTS.items():
        enumerated, odd = enumerate_counts(n, graph.CUTOFF)
        predicted = predicted_counts(n, graph.CUTOFF)
        if tuple(enumerated.values()) != frozen or predicted != frozen or odd:
            raise AssertionError(f"physical-code path prediction failed at n={n}")
        counts[str(n)] = {"enumerated": enumerated,
                          "coefficient_prediction": predicted,
                          "one_hop_return_paths": odd}

    n, cutoff = graph.N, graph.CUTOFF
    system = 2 * cutoff * n
    clock = graph.clock_bits(n, cutoff)
    prepared, reference, _, _ = graph.weights_and_vectors(n, cutoff)
    joint, _ = extension.joint_preparation(n, cutoff)
    valid = valid_code_indices(n, cutoff)
    labels = 2 * n + 1
    weights = graph.prior.frozen_label_weights(labels)
    settings = list(graph.settings(n))
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    rows = {}
    maximum_aer_error = 0.0
    maximum_extension_difference = 0.0
    maximum_invalid_mass = 0.0
    for name, source, return_ in extension.classes(cutoff):
        by_extension = {}
        for phase_extension in extension.EXTENSIONS:
            exact = np.zeros((2, n, labels))
            aer = np.zeros_like(exact)
            paths_seen = set()
            for start in range(0, len(settings), batch_size):
                chunk = settings[start:start + batch_size]
                circuits = []
                for block, axis, label, angles in chunk:
                    phases = extension.phase_diagonal(
                        n, cutoff, angles, phase_extension)
                    exact[block, axis, label], terms = analytic_probability(
                        prepared, reference, phases, source, return_,
                        cutoff, clock)
                    paths_seen.add(terms["transfer_paths"])
                    circuits.append(aer_circuit(joint, n, cutoff, source,
                                                return_, angles, phase_extension))
                result = simulator.run(circuits, shots=1).result()
                if not result.success:
                    raise RuntimeError(result.status)
                for index, (block, axis, label, _) in enumerate(chunk):
                    data = result.data(index)
                    before = np.asarray(data["after_source"].data)
                    final = np.asarray(data["final"].data)
                    maximum_invalid_mass = max(
                        maximum_invalid_mass,
                        invalid_mass(before, valid, system),
                        invalid_mass(final, valid, system))
                    aer[block, axis, label] = abs(
                        np.vdot(reference, final[:1 << system])) ** 2
            if paths_seen != {counts[str(n)]["enumerated"][name]}:
                raise AssertionError(f"sparse path count changed: {name}")
            error = float(np.max(abs(exact - aer)))
            maximum_aer_error = max(maximum_aer_error, error)
            by_extension[phase_extension] = {
                "analytic_joint_probabilities": exact.tolist(),
                "aer_joint_probabilities": aer.tolist(),
                "axis_pooled_contrast": ((exact[1] - exact[0]) @ weights).tolist(),
                "maximum_absolute_aer_error": error,
            }
        a = np.asarray(by_extension["additive"]["analytic_joint_probabilities"])
        b = np.asarray(by_extension["neutral_invalid"]["analytic_joint_probabilities"])
        extension_difference = float(np.max(abs(a - b)))
        maximum_extension_difference = max(maximum_extension_difference,
                                           extension_difference)
        rows[name] = {"transfer_paths": counts[str(n)]["enumerated"][name],
                      "maximum_extension_difference": extension_difference,
                      "extensions": by_extension}
        print(f"completed {name}: path {rows[name]['transfer_paths']}, "
              f"extension delta {extension_difference:.3e}", flush=True)
    if maximum_aer_error >= graph.TOLERANCE:
        raise AssertionError(f"Aer mismatch: {maximum_aer_error}")
    if maximum_extension_difference >= 1e-12:
        raise AssertionError(f"invalid-phase sensitivity: {maximum_extension_difference}")
    if maximum_invalid_mass >= 1e-12:
        raise AssertionError(f"code leakage: {maximum_invalid_mass}")
    output = {
        "protocol": PROTOCOL,
        "parameters": {"n": n, "cutoff": cutoff, "angle": graph.ANGLE,
                       "tau": graph.TAU, "beta": graph.BETA,
                       "reference_beta": graph.REFERENCE_BETA,
                       "shifts": graph.SHIFTS, "bias": graph.BIAS,
                       "labels": labels, "label_weights": weights.tolist(),
                       "workers": workers, "batch_size": batch_size},
        "counts": counts,
        "rows": rows,
        "maximum_absolute_aer_error": maximum_aer_error,
        "maximum_extension_difference": maximum_extension_difference,
        "maximum_invalid_code_mass": maximum_invalid_mass,
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
    print(json.dumps({
        "counts": output["counts"],
        "maximum_absolute_aer_error": output["maximum_absolute_aer_error"],
        "maximum_extension_difference": output["maximum_extension_difference"],
        "maximum_invalid_code_mass": output["maximum_invalid_code_mass"],
        "contrasts": {name: row["extensions"]["additive"]["axis_pooled_contrast"]
                      for name, row in output["rows"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
