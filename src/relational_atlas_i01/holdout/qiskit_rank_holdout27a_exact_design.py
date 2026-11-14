"""Stage 27a: shot-free design checks for the paired-access interferometer.

Predictions and thresholds are frozen in
notes/research_protocol27a_exact_design_2026-11-13.md, committed before this
module existed.  Parts:
  A) uniform-label averaging as a Z_L twirl: aliasing of charge q = +-(2n-1)
     reproduces the prevalidation-1 short-label failures;
  B) a Pauli twirl of the access pulses (sign randomization) removes the
     interference term, for the cutoff-one and the stage-26 circuits;
  C) naive k-fold repetition opens self-return paths and loses the null;
  D) diagonal-Kraus noise preserves the conjugate null; a remote Rx does not.
Criteria are recorded with their observed values; a failure is reported, not
raised, so that the frozen model is never adjusted after the fact.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator
from qiskit_aer.library import SetStatevector
from qiskit_aer.noise import phase_damping_error

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout_grid as prevalidation1
from relational_atlas_i01.holdout import qiskit_rank_holdout3_shots as shots3
from relational_atlas_i01.holdout import qiskit_rank_holdout15_dimension_transfer as model15
from relational_atlas_i01.holdout import qiskit_rank_holdout20_sector_paths as sector
from relational_atlas_i01.holdout import qiskit_rank_holdout24_charge_graph as graph
from relational_atlas_i01.holdout import qiskit_rank_holdout25_invalid_extension as extension
from relational_atlas_i01.holdout import qiskit_rank_holdout26_physical_access as physical


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout27a_exact_design.json"
PROTOCOL = "notes/research_protocol27a_exact_design_2026-11-13.md"
N = 3
LABELS = 2 * N + 1
BETA = model15.previous.BETA
REFERENCE_BETA = model15.previous.REFERENCE_BETA
BIAS = model15.grid.BIAS
SHIFTS = model15.SHIFTS
CLASSES = model15.SUPPORTS
TWIRL_ANGLES = (1.0, math.pi / 2)
SIGNS = (1, -1)
REPETITIONS = (1, 2, 3, 4, 5, 6)
REPETITION_ANGLES = (0.1, math.pi / 2)
AER_REPETITIONS = (2, 3)
NULL_REPETITIONS = (2, 3, 4)
ALIAS_SHIFT = prevalidation1.COMMON_SHIFT
NOISE_SEED = 20261113
Z_ERROR_SD = 0.3
DAMPING = 0.03
REMOTE_RX = 0.1
REMOTE_QUBIT = 3
SHOTS_PER_LABEL = 131072
ZERO = 1e-14
NONZERO = 1e-10


def _criterion(rule: str, observed: float, threshold: float,
               above: bool = False) -> dict:
    passed = observed > threshold if above else observed <= threshold
    return {"rule": rule, "observed": float(observed),
            "threshold": threshold, "passed": bool(passed)}


def cutoff_one_settings():
    """Yield (block, axis, label, angles) of the frozen cutoff-one grid."""
    for block, shift in enumerate(SHIFTS):
        for axis in range(N):
            for label in range(LABELS):
                angles = np.full(N, shift + 2 * math.pi * label / LABELS)
                angles[axis] -= BIAS
                yield block, axis, label, angles


def pooled_contrast(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (values[1] - values[0]) @ weights


def noncentrality(values: np.ndarray, weights: np.ndarray,
                  allocation: np.ndarray) -> float:
    first, second = values[0] @ weights, values[1] @ weights
    average = (first + second) / 2
    return float(np.sum((second - first) ** 2 * allocation.sum() /
                        (2 * average * (1 - average))))


# --------------------------------------------------------------------------
# A. Label aliasing
# --------------------------------------------------------------------------

def alias_change(entry: dict, tau: float, bias: float, labels: int,
                 axis: int, alpha: float = ALIAS_SHIFT) -> float:
    """Label-erased common-shift change from charges q = 0 mod L, q != 0."""
    n = entry["n"]
    angles = np.zeros(n)
    angles[axis] -= bias
    amplitude = entry["weights"] * np.exp(
        1j * (entry["energies"] * tau + entry["momenta"] @ angles))
    charge = entry["momenta"].sum(axis=1)
    by_charge = {p: complex(amplitude[charge == p].sum())
                 for p in range(-n, n + 1)}
    total = 0.0
    for q in range(1, 2 * n + 1):
        if q % labels:
            continue
        s_q = sum(by_charge[p + q] * np.conj(by_charge[p])
                  for p in range(-n, n + 1 - q))
        total += 2 * float(np.real((np.exp(1j * alpha * q) - 1) * s_q))
    return total


def part_a() -> dict:
    stored = json.loads(prevalidation1.RESULT.read_text(encoding="utf-8"))
    rows = []
    for row in stored["rows"]:
        n, labels = row["n"], row["phase_labels"]
        entry = prevalidation1.model(n, row["beta"], row["physical_P_zero"])
        tau = row["tau_over_pi"] * math.pi
        predicted = max(abs(alias_change(entry, tau, row["bias"], labels, axis))
                        for axis in range(n))
        rows.append({
            "n": n, "beta": row["beta"], "bias": row["bias"],
            "tau_over_pi": row["tau_over_pi"], "phase_labels": labels,
            "physical_P_zero": row["physical_P_zero"],
            "alias_prediction": predicted,
            "stored_erased_change":
                row["fisher"]["erased_common_shift_max_probability_change"],
        })
    error = max(abs(r["alias_prediction"] - r["stored_erased_change"])
                for r in rows)
    structural = max(r["alias_prediction"] for r in rows
                     if r["phase_labels"] == 2 * r["n"] + 1 or r["physical_P_zero"])
    short = [r for r in rows if r["phase_labels"] == 2 * r["n"] - 1
             and not r["physical_P_zero"]]
    key = lambda r: (r["n"], r["beta"], r["bias"], r["tau_over_pi"])
    predicted_set = {key(r) for r in short if r["alias_prediction"] > NONZERO}
    stored_set = {key(r) for r in short if r["stored_erased_change"] > NONZERO}
    return {
        "rows": rows,
        "short_unrestricted_rows": len(short),
        "predicted_nonzero_short_rows": len(predicted_set),
        "stored_nonzero_short_rows": len(stored_set),
        "nonzero_by_tau_over_pi": {
            str(t): sum(1 for r in short if r["tau_over_pi"] == t
                        and r["alias_prediction"] > NONZERO)
            for t in sorted({r["tau_over_pi"] for r in short})},
        "criteria": {
            "A1": _criterion("max |alias - stored| over 128 rows", error, 1e-12),
            "A2": _criterion("max alias for L=2n+1 or P=0 rows", structural, 0.0),
            "A3": {"rule": "nonzero short-row sets coincide",
                   "observed": sorted(map(list, predicted_set ^ stored_set)),
                   "passed": predicted_set == stored_set},
        },
    }


# --------------------------------------------------------------------------
# B. Pulse-sign twirl
# --------------------------------------------------------------------------

def signed_cutoff_one(source: int, return_: int, angle: float, sign: int,
                      prepared: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Sparse p_sigma = 2^-c |c^2 A0 + sigma s^2 A_ab|^2 on the full grid."""
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    clock = model15.clock_qubits(N)
    values = np.zeros((2, N, LABELS))
    for block, axis, label, angles in cutoff_one_settings():
        terms = sector.amplitude_terms(prepared, reference,
                                       sector.phase_diagonal(N, angles),
                                       source, return_)
        values[block, axis, label] = abs(
            c * c * terms["direct"] + sign * s * s * terms["transfer"]
        ) ** 2 / (1 << clock)
    return values


def insert_at_barriers(circuit: QuantumCircuit, slots: dict) -> QuantumCircuit:
    """Append callables right after barrier 1 and barrier 2, in list order."""
    output = circuit.copy_empty_like()
    barriers = 0
    for item in circuit.data:
        output.append(item.operation, item.qubits, item.clbits)
        if item.operation.name == "barrier":
            barriers += 1
            for operation in slots.get(barriers, ()):
                operation(output)
    if barriers != 2:
        raise ValueError("expected two science barriers")
    return output


def pulses(source: int, return_: int, angle: float, sign: int = 1) -> dict:
    return {1: [lambda qc: qc.rx(angle, source)],
            2: [lambda qc: qc.rx(-sign * angle, return_)]}


def merge_slots(*parts: dict) -> dict:
    merged: dict = {}
    for part in parts:
        for slot, operations in part.items():
            merged.setdefault(slot, []).extend(operations)
    return merged


def aer_cutoff_one(prepared_template, slots_for: callable, simulator,
                   ) -> np.ndarray:
    """Run the 42-setting grid; slots_for(angles) returns barrier inserts."""
    base, theta, _ = prepared_template
    circuits, settings = [], []
    for block, axis, label, angles in cutoff_one_settings():
        circuit = base.assign_parameters(
            {theta[i]: float(angles[i]) for i in range(N)})
        circuit = insert_at_barriers(circuit, slots_for(angles))
        circuit.save_probabilities()
        circuits.append(circuit)
        settings.append((block, axis, label))
    result = simulator.run(circuits, shots=1).result()
    if not result.success:
        raise RuntimeError(result.status)
    values = np.zeros((2, N, LABELS))
    for index, setting in enumerate(settings):
        values[setting] = float(result.data(index)["probabilities"][0])
    return values


def signed_physical(source: int, return_: int, sign: int) -> tuple:
    """Stage-26 sparse grid with return rotation -sign*epsilon."""
    n, cutoff = graph.N, graph.CUTOFF
    clock = graph.clock_bits(n, cutoff)
    prepared, reference, _, _ = graph.weights_and_vectors(n, cutoff)
    s = math.sin(graph.ANGLE / 2)
    labels = 2 * n + 1
    values = np.zeros((2, n, labels))
    transfer_squared = np.zeros_like(values)
    for block, axis, label, angles in graph.settings(n):
        phases = extension.phase_diagonal(n, cutoff, angles, "additive")
        _, terms = physical.analytic_probability(
            prepared, reference, phases, source, return_, cutoff, clock)
        values[block, axis, label] = abs(
            terms["diagonal"] + sign * s * s * terms["transfer"]) ** 2 / (1 << clock)
        transfer_squared[block, axis, label] = abs(terms["transfer"]) ** 2 / (1 << clock)
    return values, transfer_squared


def physical_circuit(joint: np.ndarray, source: int, return_: int,
                     angles: np.ndarray, sign: int) -> QuantumCircuit:
    n, cutoff = graph.N, graph.CUTOFF
    system = 2 * cutoff * n
    clock = graph.clock_bits(n, cutoff)
    circuit = QuantumCircuit(system + clock)
    circuit.append(SetStatevector(joint), circuit.qubits)
    physical.code_preserving_rx(circuit, source, cutoff, graph.ANGLE)
    levels = np.arange(1 << clock)
    circuit.append(DiagonalGate(np.exp(1j * graph.TAU * (
        levels + 0.37 * levels * levels / n))), range(system, system + clock))
    circuit.append(DiagonalGate(extension.phase_diagonal(
        n, cutoff, angles, "additive")), range(system))
    physical.code_preserving_rx(circuit, return_, cutoff, -sign * graph.ANGLE)
    for bit in range(clock):
        circuit.h(system + bit)
    circuit.save_statevector(label="final")
    return circuit


def aer_physical(source: int, return_: int, sign: int, simulator) -> np.ndarray:
    n, cutoff = graph.N, graph.CUTOFF
    system = 2 * cutoff * n
    joint, reference = extension.joint_preparation(n, cutoff)
    settings = list(graph.settings(n))
    circuits = [physical_circuit(joint, source, return_, angles, sign)
                for _, _, _, angles in settings]
    result = simulator.run(circuits, shots=1).result()
    if not result.success:
        raise RuntimeError(result.status)
    values = np.zeros((2, n, 2 * n + 1))
    for index, (block, axis, label, _) in enumerate(settings):
        final = np.asarray(result.data(index)["final"].data)
        values[block, axis, label] = abs(np.vdot(reference, final[:1 << system])) ** 2
    return values


def part_b(workers: int) -> dict:
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    prepared, reference = sector.vectors(N, BETA, REFERENCE_BETA)
    template = model15.template(N)
    weights = model15.frozen_label_weights(LABELS)
    allocation = shots3.shot_allocation(SHOTS_PER_LABEL, "skewed")
    law = sector.angle_law_coefficients(N, BETA, REFERENCE_BETA)
    sparse_error = law_error = twirl_error = conjugate = 0.0
    rows = {}
    for name, source, return_ in CLASSES:
        coefficient_c = np.asarray(law[name]["cross_coefficient_C"])
        coefficient_d = np.asarray(law[name]["transfer_squared_coefficient_D"])
        rows[name] = {}
        for angle in TWIRL_ANGLES:
            by_sign = {}
            for sign in SIGNS:
                sparse = signed_cutoff_one(source, return_, angle, sign,
                                           prepared, reference)
                aer = aer_cutoff_one(
                    template, lambda _a, s=source, r=return_, g=sign, e=angle:
                    pulses(s, r, e, g), simulator)
                sparse_error = max(sparse_error, float(np.max(abs(sparse - aer))))
                delta = pooled_contrast(aer, weights)
                predicted = (sign * math.sin(angle) ** 2 * coefficient_c / 2 +
                             math.sin(angle / 2) ** 4 * coefficient_d)
                law_error = max(law_error, float(np.max(abs(delta - predicted))))
                by_sign[sign] = aer
                rows[name][f"{angle:.6f}_sign_{sign:+d}"] = {
                    "axis_pooled_contrast": delta.tolist(),
                    "approx_noncentrality_stage17_budget":
                        noncentrality(aer, weights, allocation),
                }
                if name == "conjugate_partner":
                    conjugate = max(conjugate, float(np.max(abs(delta))))
            twirled = (by_sign[1] + by_sign[-1]) / 2
            delta_tw = pooled_contrast(twirled, weights)
            twirl_error = max(twirl_error, float(np.max(abs(
                delta_tw - math.sin(angle / 2) ** 4 * coefficient_d))))
            rows[name][f"{angle:.6f}_twirled"] = {
                "axis_pooled_contrast": delta_tw.tolist(),
                "approx_noncentrality_stage17_budget":
                    noncentrality(twirled, weights, allocation),
            }
        print(f"B cutoff-one completed {name}", flush=True)

    physical_sim = AerSimulator(method="statevector", max_parallel_threads=workers,
                                max_parallel_experiments=workers)
    physical_weights = graph.prior.frozen_label_weights(2 * graph.N + 1)
    s4 = math.sin(graph.ANGLE / 2) ** 4
    phys_error = phys_twirl_error = phys_conjugate = 0.0
    phys_rows = {}
    for name, source, return_ in extension.classes(graph.CUTOFF):
        by_sign = {}
        for sign in SIGNS:
            sparse, transfer_squared = signed_physical(source, return_, sign)
            aer = aer_physical(source, return_, sign, physical_sim)
            phys_error = max(phys_error, float(np.max(abs(sparse - aer))))
            by_sign[sign] = aer
            delta = pooled_contrast(aer, physical_weights)
            phys_rows[f"{name}_sign_{sign:+d}"] = delta.tolist()
            if name == "conjugate":
                phys_conjugate = max(phys_conjugate, float(np.max(abs(delta))))
        twirled = pooled_contrast((by_sign[1] + by_sign[-1]) / 2, physical_weights)
        predicted = s4 * pooled_contrast(transfer_squared, physical_weights)
        phys_twirl_error = max(phys_twirl_error, float(np.max(abs(twirled - predicted))))
        phys_rows[f"{name}_twirled"] = twirled.tolist()
        print(f"B stage-26 completed {name}", flush=True)
    return {
        "cutoff_one": rows,
        "stage26_code_preserving": phys_rows,
        "criteria": {
            "B1": _criterion("max |sparse - Aer| p_sigma", sparse_error, 1e-11),
            "B2": _criterion("max |Delta_sigma - signed two-term law| (Aer)",
                             law_error, 1e-12),
            "B3": _criterion("max |Delta_tw - sin^4(e/2) D| (Aer)",
                             twirl_error, 1e-12),
            "B4": _criterion("conjugate max |Delta_sigma| (Aer)", conjugate, ZERO),
            "B5_aer": _criterion("stage-26 max |sparse - Aer|", phys_error, 1e-11),
            "B5_twirl": _criterion("stage-26 max |Delta_tw - s^4 D'| (Aer)",
                                   phys_twirl_error, 1e-12),
            "B5_conjugate": _criterion("stage-26 conjugate max |Delta| (Aer)",
                                       phys_conjugate, ZERO),
        },
    }


# --------------------------------------------------------------------------
# C. Naive repetition
# --------------------------------------------------------------------------

def rx_state(state: np.ndarray, bit: int, angle: float) -> np.ndarray:
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    return c * state - 1j * s * state[np.arange(len(state)) ^ (1 << bit)]


def repeated_probability(prepared: np.ndarray, reference: np.ndarray,
                         phases: np.ndarray, source: int, return_: int,
                         angle: float, repetitions: int) -> float:
    state = prepared.astype(complex)
    for _ in range(repetitions):
        state = rx_state(state, source, angle)
        state = phases * state
        state = rx_state(state, return_, -angle)
    return abs(np.vdot(reference, state)) ** 2 / (1 << model15.clock_qubits(N))


def repeated_grid(source: int, return_: int, angle: float, repetitions: int,
                  prepared: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.zeros((2, N, LABELS))
    for block, axis, label, angles in cutoff_one_settings():
        values[block, axis, label] = repeated_probability(
            prepared, reference, sector.phase_diagonal(N, angles),
            source, return_, angle, repetitions)
    return values


def path_census(source: int, return_: int, prepared: np.ndarray) -> dict:
    support = set(np.flatnonzero(abs(prepared) > 0).tolist())
    transfer = sum((x ^ (1 << source) ^ (1 << return_)) in support
                   for x in support)
    self_return = (sum((x ^ (1 << source)) not in support for x in support) +
                   sum((x ^ (1 << return_)) not in support for x in support))
    return {"direct_transfer_paths": int(transfer),
            "self_return_paths_k_ge_2": int(self_return)}


def cutoff_one_joint() -> tuple[np.ndarray, np.ndarray]:
    """System+clock state with the clock holding the base energy (tau=0)."""
    entry = model15.grid.model(N, BETA, True, model15.previous.SPECTRUM)
    system, clock = 2 * N, model15.clock_qubits(N)
    joint = np.zeros(1 << (system + clock), dtype=complex)
    for momentum, energy, weight in zip(entry["momenta"],
                                        entry["base_energies"], entry["weights"]):
        index = model15.encoding.basis_index(momentum)
        joint[index | (int(energy) << system)] = math.sqrt(weight)
    _, reference = sector.vectors(N, BETA, REFERENCE_BETA)
    return joint, reference


def repeated_circuit(joint: np.ndarray, source: int, return_: int,
                     angles: np.ndarray, angle: float,
                     repetitions: int) -> QuantumCircuit:
    system, clock = 2 * N, model15.clock_qubits(N)
    circuit = QuantumCircuit(system + clock)
    circuit.append(SetStatevector(joint), circuit.qubits)
    for _ in range(repetitions):
        circuit.rx(angle, source)
        for axis in range(N):
            circuit.rz(float(angles[axis]), 2 * axis)
            circuit.rz(-float(angles[axis]), 2 * axis + 1)
        circuit.rx(-angle, return_)
    for bit in range(clock):
        circuit.h(system + bit)
    circuit.save_statevector(label="final")
    return circuit


def part_c(workers: int) -> dict:
    prepared, reference = sector.vectors(N, BETA, REFERENCE_BETA)
    skewed = model15.frozen_label_weights(LABELS)
    uniform = np.full(LABELS, 1 / LABELS)
    stored = json.loads(model15.RESULT.read_text(encoding="utf-8"))
    stage15 = {row["support_class"]: np.asarray(row["axis_pooled_joint_difference"])
               for row in stored["rows"] if row["n"] == N}
    k1_error = max(float(np.max(abs(pooled_contrast(
        repeated_grid(s, r, model15.ANGLE, 1, prepared, reference), skewed)
        - stage15[name]))) for name, s, r in CLASSES)

    census = {name: path_census(s, r, prepared) for name, s, r in CLASSES}
    table = {}
    for name, source, return_ in CLASSES:
        table[name] = {}
        for angle in REPETITION_ANGLES:
            for k in REPETITIONS:
                values = repeated_grid(source, return_, angle, k,
                                       prepared, reference)
                table[name][f"{angle:.6f}_k{k}"] = {
                    "max_abs_contrast_skewed": float(np.max(abs(
                        pooled_contrast(values, skewed)))),
                    "max_abs_contrast_uniform": float(np.max(abs(
                        pooled_contrast(values, uniform)))),
                }
    null_loss = min(table["conjugate_partner"][f"{math.pi / 2:.6f}_k{k}"]
                    ["max_abs_contrast_skewed"] for k in NULL_REPETITIONS)

    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    joint, reference_system = cutoff_one_joint()
    system = 2 * N
    aer_error = 0.0
    for name, source, return_ in CLASSES:
        for k in AER_REPETITIONS:
            settings = list(cutoff_one_settings())
            circuits = [repeated_circuit(joint, source, return_, angles,
                                         math.pi / 2, k)
                        for _, _, _, angles in settings]
            result = simulator.run(circuits, shots=1).result()
            if not result.success:
                raise RuntimeError(result.status)
            aer = np.zeros((2, N, LABELS))
            for index, (block, axis, label, _) in enumerate(settings):
                final = np.asarray(result.data(index)["final"].data)
                aer[block, axis, label] = abs(
                    np.vdot(reference_system, final[:1 << system])) ** 2
            sparse = repeated_grid(source, return_, math.pi / 2, k,
                                   prepared, reference)
            aer_error = max(aer_error, float(np.max(abs(sparse - aer))))
        print(f"C completed {name}", flush=True)
    census_ok = (census["conjugate_partner"]["direct_transfer_paths"] == 0 and
                 census["conjugate_partner"]["self_return_paths_k_ge_2"] ==
                 2 * sector.charge_count(N, 0))
    return {
        "path_census": census,
        "contrast_table": table,
        "criteria": {
            "C1": _criterion("k=1 sparse vs stored stage-15 Aer (e=1.0)",
                             k1_error, 1e-13),
            "C2_paths": {"rule": "conjugate: 0 transfer, 2*T_3(0)=14 self-return",
                         "observed": census["conjugate_partner"],
                         "passed": bool(census_ok)},
            "C2_null_loss": _criterion(
                "min over k=2,3,4 of conjugate max|Delta| (e=pi/2)",
                null_loss, NONZERO, above=True),
            "C3": _criterion("k=2,3 sparse vs independent Aer", aer_error, 1e-11),
        },
    }


# --------------------------------------------------------------------------
# D. Diagonal-noise lemma
# --------------------------------------------------------------------------

def part_d(workers: int) -> dict:
    template = model15.template(N)
    weights = model15.frozen_label_weights(LABELS)
    angle = math.pi / 2
    rng = np.random.default_rng(NOISE_SEED)
    first, second = rng.normal(0, Z_ERROR_SD, 8), rng.normal(0, Z_ERROR_SD, 8)
    damping = phase_damping_error(DAMPING).to_instruction()
    scenarios = {
        "ideal": ("statevector", {}),
        "D1_coherent_z": ("statevector", {
            1: [lambda qc, q=q: qc.rz(float(first[q]), q) for q in range(8)],
            2: [lambda qc, q=q: qc.rz(float(second[q]), q) for q in range(8)]}),
        "D2_phase_damping_slot1": ("density_matrix", {
            1: [lambda qc, q=q: qc.append(damping, [q]) for q in range(8)]}),
        "D3_remote_rx": ("statevector", {
            1: [lambda qc: qc.rx(REMOTE_RX, REMOTE_QUBIT)]}),
    }
    rows = {}
    for scenario, (method, errors) in scenarios.items():
        simulator = AerSimulator(method=method, max_parallel_threads=workers,
                                 max_parallel_experiments=workers)
        rows[scenario] = {}
        for name, source, return_ in CLASSES:
            # Errors come first in each slot, i.e. before the pulse.
            slots = merge_slots(errors, pulses(source, return_, angle))
            values = aer_cutoff_one(template, lambda _a, sl=slots: sl, simulator)
            delta = pooled_contrast(values, weights)
            rows[scenario][name] = {
                "axis_pooled_contrast": delta.tolist(),
                "max_abs_contrast": float(np.max(abs(delta))),
            }
        print(f"D completed {scenario}", flush=True)
    conjugate = {s: rows[s]["conjugate_partner"]["max_abs_contrast"] for s in rows}
    return {
        "z_error_angles": {"slot1": first.tolist(), "slot2": second.tolist()},
        "rows": rows,
        "criteria": {
            "D1": _criterion("coherent Z errors: conjugate max|Delta|",
                             conjugate["D1_coherent_z"], ZERO),
            "D2": _criterion("phase damping slot 1: conjugate max|Delta|",
                             conjugate["D2_phase_damping_slot1"], ZERO),
            "D3": _criterion("remote Rx(0.1) on q3: conjugate max|Delta|",
                             conjugate["D3_remote_rx"], NONZERO, above=True),
        },
    }


def study(workers: int) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    parts = {"A_label_aliasing": part_a()}
    print("A completed", flush=True)
    parts["B_pulse_sign_twirl"] = part_b(workers)
    parts["C_repetition"] = part_c(workers)
    parts["D_diagonal_noise"] = part_d(workers)
    criteria = {name: value for part in parts.values()
                for name, value in part["criteria"].items()}
    output = {
        "protocol": PROTOCOL,
        "parameters": {
            "n": N, "labels": LABELS, "beta": BETA,
            "reference_beta": REFERENCE_BETA, "bias": BIAS, "shifts": SHIFTS,
            "twirl_angles": TWIRL_ANGLES, "repetitions": REPETITIONS,
            "repetition_angles": REPETITION_ANGLES, "alias_shift": ALIAS_SHIFT,
            "noise_seed": NOISE_SEED, "z_error_sd": Z_ERROR_SD,
            "phase_damping": DAMPING, "remote_rx": REMOTE_RX,
            "remote_qubit": REMOTE_QUBIT, "workers": workers,
            "stage26": {"cutoff": graph.CUTOFF, "angle": graph.ANGLE,
                        "tau": graph.TAU, "extension": "additive"},
        },
        **parts,
        "criteria_summary": {name: value["passed"] for name, value in criteria.items()},
        "all_preregistered_checks_passed": all(
            value["passed"] for value in criteria.values()),
    }
    RESULT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    output = study(args.workers)
    print(json.dumps({
        "criteria": {name: {k: v for k, v in value.items() if k != "rule"}
                     for part in ("A_label_aliasing", "B_pulse_sign_twirl",
                                  "C_repetition", "D_diagonal_noise")
                     for name, value in output[part]["criteria"].items()},
        "all_preregistered_checks_passed":
            output["all_preregistered_checks_passed"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
