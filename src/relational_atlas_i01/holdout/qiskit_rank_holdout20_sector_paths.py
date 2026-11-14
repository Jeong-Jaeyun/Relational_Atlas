"""Independent amplitude identity and sector-path audit for paired pulses.

For a clock phase tau absorbed into the initial amplitudes, the
clock-zero/system-zero amplitude is
  2^(-clock_bits/2) [cos(e/2)^2 A0 + sin(e/2)^2 A_sr],
where A0=<r|U(theta)|w> and A_sr=<r|X_return U(theta) X_source|w>.
The single-flip terms vanish exactly for strict P=0 input and detector
support. This module evaluates the expression directly in the computational
basis without circuit construction or Qiskit state simulation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math

import numpy as np

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout15_dimension_transfer as circuit_model


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout20_sector_paths.json"
ANGLE_REFERENCE = PROJECT_ROOT / "results" / "qiskit_rank_holdout16_angle_law.json"
DIMENSIONS = (2, 3, 4)
BETAS = (0.0, 0.45, 1.20)
REFERENCE_BETAS = (0.0, 0.15, 0.80)
ANGLES = (1.0, math.pi / 2)


def charge_count(sites: int, charge: int) -> int:
    return sum(sum(values) == charge for values in
               itertools.product((-1, 0, 1), repeat=sites))


def expected_path_counts(n: int) -> dict:
    """Combinatorial paths for cutoff-one, strict-P=0 reference support."""
    return {
        "same_support": charge_count(n, 0),
        "conjugate_partner": 0,
        "different_pair_same_sign": 2 * charge_count(n - 2, -1),
        "different_pair_opposite_sign": 2 * charge_count(n - 2, 0),
    }


def vectors(n: int, beta: float, reference_beta: float,
            tau: float = 0.0):
    """Return the strict-P=0 preparation and reference amplitudes."""
    entry = circuit_model.grid.model(n, beta, True,
                                     circuit_model.previous.SPECTRUM)
    size = 1 << (2 * n)
    prepared = np.zeros(size, dtype=complex)
    reference = np.zeros(size, dtype=complex)
    reference_weights = np.exp(-reference_beta * entry["base_energies"])
    reference_weights /= reference_weights.sum()
    for k, energy, weight, reference_weight in zip(
            entry["momenta"], entry["base_energies"], entry["weights"],
            reference_weights):
        index = circuit_model.encoding.basis_index(k)
        clock_phase = tau * (energy + 0.37 * energy * energy / n)
        prepared[index] = math.sqrt(weight) * np.exp(1j * clock_phase)
        reference[index] = math.sqrt(reference_weight)
    return prepared, reference


def phase_diagonal(n: int, angles: np.ndarray) -> np.ndarray:
    states = np.arange(1 << (2 * n))
    phase = np.zeros(len(states))
    for axis in range(n):
        positive = (states >> (2 * axis)) & 1
        negative = (states >> (2 * axis + 1)) & 1
        phase += angles[axis] * (positive - negative)
    return np.exp(1j * phase)


def amplitude_terms(prepared: np.ndarray, reference: np.ndarray,
                    phases: np.ndarray, source: int, return_: int) -> dict:
    states = np.arange(len(prepared))
    after_source = states ^ (1 << source)
    after_return = after_source ^ (1 << return_)
    direct = np.vdot(reference, phases * prepared)
    transfer = np.sum(prepared * np.conj(reference[after_return]) *
                      phases[after_source])
    return_only = np.sum(prepared * np.conj(reference[states ^ (1 << return_)]) *
                         phases)
    source_only = np.sum(prepared * np.conj(reference[after_source]) *
                         phases[after_source])
    paths = int(np.count_nonzero((abs(prepared) > 0) &
                                 (abs(reference[after_return]) > 0)))
    return {"direct": direct, "transfer": transfer,
            "return_only": return_only, "source_only": source_only,
            "allowed_transfer_paths": paths}


def probability(n: int, prepared: np.ndarray, reference: np.ndarray,
                phases: np.ndarray, source: int, return_: int,
                angle: float) -> tuple[float, dict]:
    terms = amplitude_terms(prepared, reference, phases, source, return_)
    c = math.cos(angle / 2)
    s = math.sin(angle / 2)
    amplitude = c * c * terms["direct"] + s * s * terms["transfer"]
    # One-qubit-flip terms are zero due to the P=0 support. Keep them in the
    # output as an explicit check of the identity's required premise.
    return abs(amplitude) ** 2 / (1 << circuit_model.clock_qubits(n)), terms


def contrasts(n: int, beta: float, reference_beta: float, angle: float,
              label_distribution: str, tau: float = 0.0) -> tuple[dict, float]:
    prepared, reference = vectors(n, beta, reference_beta, tau)
    labels = 2 * n + 1
    weights = (circuit_model.frozen_label_weights(labels)
               if label_distribution == "skewed" else np.full(labels, 1 / labels))
    rows = {}
    largest_odd = 0.0
    for name, source, return_ in circuit_model.SUPPORTS:
        values = np.zeros((2, n, labels))
        paths = None
        for block, shift in enumerate(circuit_model.SHIFTS):
            for axis in range(n):
                for label in range(labels):
                    angles = np.full(n, shift + 2 * math.pi * label / labels)
                    angles[axis] -= circuit_model.grid.BIAS
                    values[block, axis, label], terms = probability(
                        n, prepared, reference, phase_diagonal(n, angles),
                        source, return_, angle)
                    paths = terms["allowed_transfer_paths"]
                    largest_odd = max(largest_odd, abs(terms["source_only"]),
                                      abs(terms["return_only"]))
        delta = (values[1] - values[0]) @ weights
        rows[name] = {
            "allowed_transfer_paths": paths,
            "axis_pooled_joint_difference": delta.tolist(),
            "maximum_absolute_joint_difference": float(np.max(np.abs(delta))),
        }
    return rows, float(largest_odd)


def angle_law_coefficients(n: int, beta: float, reference_beta: float) -> dict:
    """Return coefficients C,D in Delta(e)=sin(e)^2*C/2+sin(e/2)^4*D."""
    prepared, reference = vectors(n, beta, reference_beta)
    labels = 2 * n + 1
    weights = circuit_model.frozen_label_weights(labels)
    normalization = 1 << circuit_model.clock_qubits(n)
    output = {}
    for name, source, return_ in circuit_model.SUPPORTS:
        cross = np.zeros((2, n, labels))
        transfer_squared = np.zeros_like(cross)
        direct_squared = np.zeros_like(cross)
        for block, shift in enumerate(circuit_model.SHIFTS):
            for axis in range(n):
                for label in range(labels):
                    angles = np.full(n, shift + 2 * math.pi * label / labels)
                    angles[axis] -= circuit_model.grid.BIAS
                    terms = amplitude_terms(prepared, reference,
                                            phase_diagonal(n, angles),
                                            source, return_)
                    direct, transfer = terms["direct"], terms["transfer"]
                    cross[block, axis, label] = (
                        np.real(np.conj(direct) * transfer) / normalization)
                    transfer_squared[block, axis, label] = (
                        abs(transfer) ** 2 / normalization)
                    direct_squared[block, axis, label] = (
                        abs(direct) ** 2 / normalization)
        c = (cross[1] - cross[0]) @ weights
        d = (transfer_squared[1] - transfer_squared[0]) @ weights
        direct_residual = (direct_squared[1] - direct_squared[0]) @ weights
        output[name] = {
            "cross_coefficient_C": c.tolist(),
            "transfer_squared_coefficient_D": d.tolist(),
            "direct_block_difference": direct_residual.tolist(),
            "maximum_absolute_direct_block_difference": float(np.max(
                np.abs(direct_residual))),
        }
    return output


def angle_law_reproduction(coefficients: dict) -> float | None:
    if not ANGLE_REFERENCE.exists():
        return None
    reference = json.loads(ANGLE_REFERENCE.read_text(encoding="utf-8"))
    error = 0.0
    for row in reference["rows"]:
        angle = row["angle_radians"]
        term = coefficients[row["support_class"]]
        c = np.asarray(term["cross_coefficient_C"])
        d = np.asarray(term["transfer_squared_coefficient_D"])
        predicted = (math.sin(angle) ** 2 * c / 2 +
                     math.sin(angle / 2) ** 4 * d)
        observed = np.asarray(row["axis_pooled_joint_difference"])
        error = max(error, float(np.max(np.abs(predicted - observed))))
    return error


def study() -> dict:
    baseline = {}
    reproduction_error = 0.0
    reference = json.loads(circuit_model.RESULT.read_text(encoding="utf-8"))
    expected = {(row["n"], row["support_class"]): row for row in reference["rows"]}
    for n in DIMENSIONS:
        rows, odd = contrasts(n, circuit_model.previous.BETA,
                              circuit_model.previous.REFERENCE_BETA, 1.0,
                              "skewed")
        baseline[str(n)] = {"rows": rows, "largest_single_flip_amplitude": odd}
        if {name: row["allowed_transfer_paths"] for name, row in rows.items()} != (
                expected_path_counts(n)):
            raise AssertionError(f"sector path count mismatch for n={n}")
        for name, row in rows.items():
            reproduced = np.asarray(row["axis_pooled_joint_difference"])
            target = np.asarray(expected[(n, name)]["axis_pooled_joint_difference"])
            reproduction_error = max(reproduction_error,
                                     float(np.max(np.abs(reproduced - target))))
    robustness = []
    for n in DIMENSIONS:
        for beta in BETAS:
            for reference_beta in REFERENCE_BETAS:
                for distribution in ("uniform", "skewed"):
                    rows, odd = contrasts(n, beta, reference_beta,
                                          math.pi / 2, distribution)
                    robustness.append({
                        "n": n, "beta": beta, "reference_beta": reference_beta,
                        "label_distribution": distribution,
                        "largest_single_flip_amplitude": odd,
                        "rows": rows,
                    })
    extension = {}
    for n in (5, 6):
        rows, odd = contrasts(n, circuit_model.previous.BETA,
                              circuit_model.previous.REFERENCE_BETA,
                              math.pi / 2, "skewed")
        extension[str(n)] = {"rows": rows,
                             "largest_single_flip_amplitude": odd}
        if {name: row["allowed_transfer_paths"] for name, row in rows.items()} != (
                expected_path_counts(n)):
            raise AssertionError(f"extended path count mismatch for n={n}")
    clock_tau_scan = {}
    for tau in (0.0, math.pi / 4, math.pi / 2, math.pi):
        rows, odd = contrasts(3, circuit_model.previous.BETA,
                              circuit_model.previous.REFERENCE_BETA,
                              math.pi / 2, "skewed", tau=tau)
        clock_tau_scan[f"{tau / math.pi:.2f}"] = {
            "rows": rows, "largest_single_flip_amplitude": odd}
    uniform_clock_control, _ = contrasts(
        3, circuit_model.previous.BETA,
        circuit_model.previous.REFERENCE_BETA,
        math.pi / 2, "uniform", tau=math.pi / 2)
    if max(row["maximum_absolute_joint_difference"]
           for row in uniform_clock_control.values()) > 1e-12:
        raise AssertionError("uniform phase twirl unexpectedly retained a block contrast")
    angle_law = angle_law_coefficients(3, circuit_model.previous.BETA,
                                       circuit_model.previous.REFERENCE_BETA)
    return {
        "identity": "P_joint=2^(-clock_bits)|cos(e/2)^2 A0+sin(e/2)^2 A_sr|^2 with clock phase absorbed into initial amplitudes",
        "parameters": {"dimensions": DIMENSIONS, "betas": BETAS,
                       "reference_betas": REFERENCE_BETAS,
                       "angles": ANGLES, "bias": circuit_model.grid.BIAS,
                       "shifts": circuit_model.SHIFTS},
        "baseline": baseline,
        "predicted_path_counts_n2_to_n6": {
            str(n): expected_path_counts(n) for n in range(2, 7)},
        "maximum_absolute_probability_contrast_error_vs_Aer": reproduction_error,
        "exact_n3_angle_law": angle_law,
        "maximum_absolute_angle_law_error_vs_Aer":
            angle_law_reproduction(angle_law),
        "robustness": robustness,
        "analytic_extension_without_Aer": extension,
        "clock_tau_scan": clock_tau_scan,
        "uniform_clock_control_tau_half_pi": uniform_clock_control,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    output = study()
    RESULT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "maximum_Aer_reproduction_error":
            output["maximum_absolute_probability_contrast_error_vs_Aer"],
        "maximum_angle_law_error":
            output["maximum_absolute_angle_law_error_vs_Aer"],
        "baseline_paths": {n: {name: row["allowed_transfer_paths"]
                               for name, row in entry["rows"].items()}
                           for n, entry in output["baseline"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
