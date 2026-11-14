"""Protocol-2 exact grid: detector POVM, clock spectrum, and label distribution.

Freeze: notes/prevalidation_protocol2_2026-10-04.tex. This file computes
probabilities and analytic gradients; a separate Qiskit circuit audits cases.
"""

from __future__ import annotations

import itertools
import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout_grid as previous


PROJECT = PROJECT_ROOT
RESULT = PROJECT / "results" / "qiskit_rank_holdout2_grid.json"
NS = (3, 4, 5)
BETAS = (0.45, 1.20)
DETECTORS = ("mismatched", "mixed", "twirled")
SPECTRA = ("quadratic", "nonlinear", "charge_coupled")
TAUS = (0.0, math.pi / 2)
LABEL_DISTRIBUTIONS = ("uniform", "skewed")
BIAS = 1.25
COMMON_SHIFT = 0.23
RANK_RELATIVE_CUTOFF = 1e-8
EXACT_INVARIANCE_TOLERANCE = 1e-10


def model(n: int, beta: float, physical: bool, spectrum: str) -> dict:
    entry = previous.model(n, beta, physical)
    e0 = entry["energies"].astype(float)
    charge = np.sum(entry["momenta"], axis=1)
    if spectrum == "quadratic":
        energies = e0
    elif spectrum == "nonlinear":
        energies = e0 + 0.37 * e0 * e0 / n
    elif spectrum == "charge_coupled":
        energies = e0 + 0.41 * charge * charge / n
    else:
        raise ValueError(f"unknown spectrum: {spectrum}")
    return {**entry, "base_energies": e0, "energies": energies,
            "charge": charge, "spectrum": spectrum}


def detector_components(entry: dict, detector: str) -> tuple[tuple[float, np.ndarray], ...]:
    if detector == "mismatched":
        choices = ((1.0, 0.15),)
    elif detector in ("mixed", "twirled"):
        choices = ((0.7, 0.15), (0.3, 1.80))
    else:
        raise ValueError(f"unknown detector: {detector}")
    components = []
    for mass, reference_beta in choices:
        reference = np.exp(-reference_beta * entry["base_energies"])
        reference /= reference.sum()
        components.append((mass, np.sqrt(entry["weights"] * reference)))
    return tuple(components)


def label_weights(labels: int, distribution: str) -> np.ndarray:
    if distribution == "uniform":
        return np.full(labels, 1 / labels)
    if distribution == "skewed":
        t = np.arange(labels)
        return (1 + 0.35 * np.cos(2 * np.pi * t / labels)) / labels
    raise ValueError(f"unknown label distribution: {distribution}")


def angles_for_grid(n: int, labels: int, common: float = 0.0) -> np.ndarray:
    result = np.empty((n, labels, n))
    for axis in range(n):
        for t in range(labels):
            result[axis, t] = common + 2 * math.pi * t / labels
            result[axis, t, axis] -= BIAS
    return result


def probability_gradient(entry: dict, detector: str, tau: float,
                         angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return p and dp/dtheta for any trailing angle-vector dimension."""
    mom = entry["momenta"]
    flat = np.asarray(angles, dtype=float).reshape(-1, entry["n"])
    phase = tau * entry["energies"][None, :] + flat @ mom.T
    factors = np.exp(1j * phase)
    probability = np.zeros(len(flat))
    gradient = np.zeros((len(flat), entry["n"]))
    twirled = detector == "twirled"
    charge_values = np.unique(entry["charge"]) if twirled else (None,)
    for mass, coefficient in detector_components(entry, detector):
        weighted = factors * coefficient[None, :]
        for charge in charge_values:
            if charge is None:
                part = weighted
                part_mom = mom
            else:
                select = entry["charge"] == charge
                part = weighted[:, select]
                part_mom = mom[select]
            amplitude = np.sum(part, axis=1)
            derivative = 1j * (part @ part_mom)
            probability += mass * np.abs(amplitude) ** 2
            gradient += mass * 2 * np.real(
                np.conj(amplitude)[:, None] * derivative)
    leading_shape = np.asarray(angles).shape[:-1]
    return (probability.reshape(leading_shape),
            gradient.reshape(*leading_shape, entry["n"]))


def fisher_summary(matrix: np.ndarray) -> dict:
    eigenvalues = np.linalg.eigvalsh(matrix)
    maximum = max(float(eigenvalues[-1]), 1e-300)
    common = np.ones(len(matrix)) / math.sqrt(len(matrix))
    return {"eigenvalues": eigenvalues.tolist(),
            "rank_relative_1e_minus_8": int(np.sum(
                eigenvalues > RANK_RELATIVE_CUTOFF * maximum)),
            "common_direction_information": float(common @ matrix @ common),
            "minimum_eigenvalue": float(eigenvalues[0])}


def information(entry: dict, detector: str, tau: float,
                distribution: str) -> dict:
    n = entry["n"]
    labels = 2 * n + 1
    weights = label_weights(labels, distribution)
    angles = angles_for_grid(n, labels)
    probabilities, gradients = probability_gradient(entry, detector, tau, angles)
    if np.any(probabilities <= 0) or np.any(probabilities >= 1):
        raise ArithmeticError("singular Bernoulli probability")
    kept = np.einsum(
        "itj,itk,it->jk", gradients, gradients,
        weights[None, :] / (n * probabilities * (1 - probabilities)))
    pooled_probability = probabilities @ weights
    pooled_gradient = np.einsum("itj,t->ij", gradients, weights)
    if np.any(pooled_probability <= 0) or np.any(pooled_probability >= 1):
        raise ArithmeticError("singular pooled Bernoulli probability")
    erased = np.einsum(
        "ij,ik,i->jk", pooled_gradient, pooled_gradient,
        1 / (n * pooled_probability * (1 - pooled_probability)))
    shifted, _ = probability_gradient(
        entry, detector, tau, angles + COMMON_SHIFT)
    shifted_pooled = shifted @ weights
    return {"label_kept": fisher_summary(kept),
            "label_erased": fisher_summary(erased),
            "erased_common_shift_max_probability_change": float(np.max(
                np.abs(shifted_pooled - pooled_probability))),
            "probability_range": [float(probabilities.min()),
                                  float(probabilities.max())],
            "label_weight_range": [float(weights.min()), float(weights.max())]}


def gradient_crosscheck(entry: dict, detector: str, tau: float) -> float:
    angles = angles_for_grid(entry["n"], 2 * entry["n"] + 1, common=0.17)[0, 1]
    _, analytic = probability_gradient(entry, detector, tau, angles)
    numeric = np.zeros(entry["n"])
    h = 1e-5
    for axis in range(entry["n"]):
        displacement = np.zeros(entry["n"])
        displacement[axis] = h
        plus, _ = probability_gradient(entry, detector, tau, angles + displacement)
        minus, _ = probability_gradient(entry, detector, tau, angles - displacement)
        numeric[axis] = (plus - minus) / (2 * h)
    return float(np.max(np.abs(analytic - numeric)))


def study() -> dict:
    rows = []
    for n, beta, spectrum, detector, tau, distribution, physical in itertools.product(
        NS, BETAS, SPECTRA, DETECTORS, TAUS, LABEL_DISTRIBUTIONS, (False, True)
    ):
        entry = model(n, beta, physical, spectrum)
        fisher = information(entry, detector, tau, distribution)
        rows.append({
            "n": n, "beta": beta, "physical_P_zero": physical,
            "spectrum": spectrum, "detector": detector,
            "tau_over_pi": tau / math.pi,
            "label_distribution": distribution, "phase_labels": 2 * n + 1,
            "state_qfi": previous.state_qfi(entry), "fisher": fisher,
            "analytic_vs_central_difference_max_gradient_error":
                gradient_crosscheck(entry, detector, tau),
        })
    def is_invariant(row: dict) -> bool:
        return (row["physical_P_zero"] or row["detector"] == "twirled" or
                row["label_distribution"] == "uniform")
    guaranteed = [row for row in rows if is_invariant(row)]
    coherent_full = [row for row in rows if not row["physical_P_zero"] and
                     row["detector"] != "twirled"]
    skewed_coherent_full = [row for row in coherent_full if
                            row["label_distribution"] == "skewed"]
    expected_kept_rank_failures = [row for row in rows if
        row["fisher"]["label_kept"]["rank_relative_1e_minus_8"] !=
        (row["n"] if not row["physical_P_zero"] and
         row["detector"] != "twirled" else row["n"] - 1)]
    return {
        "protocol": "notes/prevalidation_protocol2_2026-10-04.tex",
        "summary": {
            "rows": len(rows),
            "guaranteed_invariant_rows": len(guaranteed),
            "guaranteed_invariance_failures_above_1e_minus_10": sum(
                row["fisher"]["erased_common_shift_max_probability_change"] >
                EXACT_INVARIANCE_TOLERANCE for row in guaranteed),
            "guaranteed_erased_rank_failures_relative_1e_minus_8": sum(
                row["fisher"]["label_erased"]["rank_relative_1e_minus_8"] !=
                row["n"] - 1 for row in guaranteed),
            "expected_kept_rank_failures_relative_1e_minus_8":
                len(expected_kept_rank_failures),
            "state_qfi_rank_failures_relative_1e_minus_8": sum(
                row["state_qfi"]["rank_relative_1e_minus_8"] !=
                row["n"] - int(row["physical_P_zero"]) for row in rows),
            "skewed_coherent_full_rows": len(skewed_coherent_full),
            "skewed_coherent_full_rows_with_residual_above_1e_minus_10": sum(
                row["fisher"]["erased_common_shift_max_probability_change"] >
                EXACT_INVARIANCE_TOLERANCE for row in skewed_coherent_full),
            "skewed_coherent_full_erased_rank_n_rows_relative_1e_minus_8": sum(
                row["fisher"]["label_erased"]["rank_relative_1e_minus_8"] ==
                row["n"] for row in skewed_coherent_full),
            "maximum_gradient_crosscheck_error": max(
                row["analytic_vs_central_difference_max_gradient_error"]
                for row in rows),
        },
        "expected_kept_rank_failures": expected_kept_rank_failures,
        "rows": rows,
    }


def main() -> None:
    result = study()
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({"summary": result["summary"],
                      "result": str(RESULT)}, indent=2))


if __name__ == "__main__":
    main()
