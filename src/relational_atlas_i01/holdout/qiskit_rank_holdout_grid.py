"""Predeclared I01 holdout grid for label, clock, and constraint rank claims.

The settings are fixed in notes/prevalidation_protocol_2026-10-04.tex.
This is an exact-probability validation grid, not an Aer shot experiment.
"""

from __future__ import annotations

import itertools
import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np


PROJECT = PROJECT_ROOT
RESULT = PROJECT / "results" / "qiskit_rank_holdout_grid.json"
NS = (3, 4, 5, 6)
BETAS = (0.45, 1.20)
BIASES = (0.55, 1.25)
TAUS = (0.0, math.pi / 2)
COMMON_SHIFT = 0.23


def model(n: int, beta: float, physical: bool) -> dict:
    momenta = np.asarray(list(itertools.product((-1, 0, 1), repeat=n)), dtype=int)
    if physical:
        momenta = momenta[np.sum(momenta, axis=1) == 0]
    energies = np.sum(momenta * momenta, axis=1)
    weights = np.exp(-beta * energies)
    weights /= weights.sum()
    return {"n": n, "momenta": momenta, "energies": energies,
            "weights": weights, "physical": physical}


def probability_gradient(entry: dict, tau: float,
                         angles: np.ndarray) -> tuple[float, np.ndarray]:
    phase = entry["energies"] * tau + entry["momenta"] @ angles
    amplitude = entry["weights"] * np.exp(1j * phase)
    overlap = np.sum(amplitude)
    derivative = 1j * amplitude @ entry["momenta"]
    return float(abs(overlap) ** 2), 2 * np.real(np.conj(overlap) * derivative)


def state_qfi(entry: dict) -> dict:
    momenta = entry["momenta"]
    weights = entry["weights"]
    mean = weights @ momenta
    covariance = (momenta.T * weights) @ momenta - np.outer(mean, mean)
    eigenvalues = np.linalg.eigvalsh(4 * covariance)
    maximum = max(float(eigenvalues[-1]), 1e-300)
    return {"eigenvalues": eigenvalues.tolist(),
            "rank_relative_1e_minus_8": int(np.sum(
                eigenvalues > 1e-8 * maximum))}


def settings(n: int, bias: float, labels: int, common: float = 0.0):
    for axis in range(n):
        for index in range(labels):
            angles = np.full(n, common + 2 * math.pi * index / labels)
            angles[axis] -= bias
            yield axis, index, angles


def information(entry: dict, tau: float, bias: float, labels: int) -> dict:
    n = entry["n"]
    probabilities = np.zeros((n, labels))
    gradients = np.zeros((n, labels, n))
    for axis, index, angles in settings(n, bias, labels):
        probabilities[axis, index], gradients[axis, index] = probability_gradient(
            entry, tau, angles)
    if np.any(probabilities <= 0) or np.any(probabilities >= 1):
        raise ArithmeticError("The predeclared settings include a singular Bernoulli point")
    kept = np.einsum("itj,itk,it->jk", gradients, gradients,
                     1 / (probabilities * (1 - probabilities))) / (n * labels)
    pooled_probability = np.mean(probabilities, axis=1)
    pooled_gradient = np.mean(gradients, axis=1)
    erased = np.einsum("ij,ik,i->jk", pooled_gradient, pooled_gradient,
                       1 / (pooled_probability * (1 - pooled_probability))) / n
    common = np.ones(n) / math.sqrt(n)
    result = {}
    for name, matrix in (("label_kept", kept), ("label_erased", erased)):
        eigenvalues = np.linalg.eigvalsh(matrix)
        maximum = max(float(eigenvalues[-1]), 1e-300)
        result[name] = {
            "eigenvalues": eigenvalues.tolist(),
            "rank_relative_1e_minus_8": int(np.sum(eigenvalues > 1e-8 * maximum)),
            "common_direction_information": float(common @ matrix @ common),
            "smallest_positive_to_largest": float(
                min((value for value in eigenvalues if value > 1e-12 * maximum),
                    default=0.0) / maximum),
        }
    shifted = np.zeros((n, labels))
    for axis, index, angles in settings(n, bias, labels, COMMON_SHIFT):
        shifted[axis, index], _ = probability_gradient(entry, tau, angles)
    result["erased_common_shift_max_probability_change"] = float(np.max(
        np.abs(np.mean(shifted, axis=1) - pooled_probability)))
    result["probability_range"] = [float(np.min(probabilities)),
                                   float(np.max(probabilities))]
    return result


def finite_difference_gradient_check(entry: dict, tau: float,
                                     bias: float, labels: int) -> float:
    n = entry["n"]
    _, _, angles = next(settings(n, bias, labels, 0.17))
    _, analytic = probability_gradient(entry, tau, angles)
    h = 1e-5
    numeric = np.empty(n)
    for axis in range(n):
        displacement = np.zeros(n)
        displacement[axis] = h
        plus, _ = probability_gradient(entry, tau, angles + displacement)
        minus, _ = probability_gradient(entry, tau, angles - displacement)
        numeric[axis] = (plus - minus) / (2 * h)
    return float(np.max(np.abs(analytic - numeric)))


def study() -> dict:
    rows = []
    for n, beta, bias, tau in itertools.product(NS, BETAS, BIASES, TAUS):
        for labels in (2 * n - 1, 2 * n + 1):
            for physical in (False, True):
                entry = model(n, beta, physical)
                fisher = information(entry, tau, bias, labels)
                rows.append({
                    "n": n, "beta": beta, "bias": bias,
                    "tau_over_pi": tau / math.pi,
                    "phase_labels": labels,
                    "physical_P_zero": physical,
                    "state_qfi": state_qfi(entry),
                    "fisher": fisher,
                    "analytic_vs_central_difference_max_gradient_error":
                        finite_difference_gradient_check(entry, tau, bias, labels),
                })
    good = [row for row in rows if row["phase_labels"] == 2 * row["n"] + 1]
    short = [row for row in rows if row["phase_labels"] == 2 * row["n"] - 1]
    failures = [row for row in good if (
        row["fisher"]["label_kept"]["rank_relative_1e_minus_8"] !=
        row["n"] - int(row["physical_P_zero"]) or
        row["fisher"]["label_erased"]["rank_relative_1e_minus_8"] !=
        row["n"] - 1 or
        row["fisher"]["erased_common_shift_max_probability_change"] > 1e-10)]
    short_full = [row for row in short if not row["physical_P_zero"]]
    return {
        "protocol": "notes/prevalidation_protocol_2026-10-04.tex",
        "settings": {"n": NS, "beta": BETAS, "bias": BIASES,
                     "tau_over_pi": [tau / math.pi for tau in TAUS],
                     "phase_label_rules": ["2n-1", "2n+1"],
                     "common_shift_for_exact_invariance_check": COMMON_SHIFT},
        "summary": {
            "total_rows": len(rows),
            "adequate_label_rows": len(good),
            "adequate_label_predeclared_pattern_failures": len(failures),
            "state_qfi_pattern_failures": sum(
                row["state_qfi"]["rank_relative_1e_minus_8"] !=
                row["n"] - int(row["physical_P_zero"]) for row in rows),
            "short_label_unrestricted_rows": len(short_full),
            "short_label_rows_with_nonzero_erased_common_change_above_1e_minus_10":
                sum(row["fisher"]["erased_common_shift_max_probability_change"] >
                    1e-10 for row in short_full),
            "maximum_gradient_crosscheck_error": max(
                row["analytic_vs_central_difference_max_gradient_error"]
                for row in rows),
        },
        "adequate_label_pattern_failures": failures,
        "rows": rows,
    }


def main() -> None:
    result = study()
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "result": str(RESULT)},
                     indent=2))


if __name__ == "__main__":
    main()
