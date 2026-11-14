"""I01 condition atlas for relationally accessible dimensions on a torus.

This is a finite Fourier, clock-conditioned toy model. The exact diagonal
gauge, low-energy collective gap, and restricted-detector cases are kept
separate. None is a derivation of physical cosmic dimensionality.
"""

from __future__ import annotations

import itertools
import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = PROJECT_ROOT
OUT = PROJECT / "results" / "condition_map.json"
FIG = PROJECT / "results" / "condition_atlas.png"
BETA = 0.35


def grid(n: int, cutoff: int) -> np.ndarray:
    return np.asarray(list(itertools.product(range(-cutoff, cutoff + 1), repeat=n)), dtype=int)


def qfi_from_weights(k: np.ndarray, weights: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(weights, dtype=float)
    probabilities /= probabilities.sum()
    mean = probabilities @ k
    centered = k - mean
    return 4.0 * (centered.T * probabilities) @ centered


def eig_summary(qfi: np.ndarray) -> dict:
    eigenvalues, eigenvectors = np.linalg.eigh(qfi)
    null_direction = eigenvectors[:, 0]
    if null_direction.sum() < 0:
        null_direction = -null_direction
    return {
        "rank": int(np.linalg.matrix_rank(qfi, tol=1e-9)),
        "eigenvalues": np.round(eigenvalues, 9).tolist(),
        "smallest_eigenvector": np.round(null_direction, 6).tolist(),
    }


def exact_sectors() -> dict:
    results = {}
    for n, cutoff in ((4, 1), (4, 2), (4, 3), (5, 1), (6, 1)):
        k = grid(n, cutoff)
        e = np.sum(k * k, axis=1)
        allowed = np.sum(k, axis=1) == 0
        weights = np.exp(-BETA * e[allowed])
        results[f"n{n}_cutoff{cutoff}"] = {
            "full_basis_size": len(k),
            "gauge_basis_size": int(allowed.sum()),
            "gauge_qfi": eig_summary(qfi_from_weights(k[allowed], weights)),
            "unrestricted_qfi": eig_summary(qfi_from_weights(k, np.exp(-BETA * e))),
        }
    k4 = grid(4, 1)
    mask = k4[:, 3] == 0
    e4 = np.sum(k4[mask] * k4[mask], axis=1)
    results["axis_4_explicit_control"] = eig_summary(
        qfi_from_weights(k4[mask], np.exp(-BETA * e4))
    )
    return results


def hard_gap_scan() -> dict:
    k = grid(4, 2)
    free_e = np.sum(k * k, axis=1)
    charge = np.sum(k, axis=1)
    cutoffs = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.9, 5.0, 6.0, 8.0)
    out = {}
    for coupling in (0.0, 1.0, 2.0, 4.0):
        energy = free_e + coupling * charge * charge
        rows = []
        for limit in cutoffs:
            included = energy <= limit + 1e-12
            qfi = qfi_from_weights(k[included], np.ones(included.sum()))
            rows.append({
                "energy_limit": limit,
                "basis_size": int(included.sum()),
                "charged_modes": int(np.sum(included & (charge != 0))),
                "qfi": eig_summary(qfi),
            })
        out[str(coupling)] = rows
    return {
        "hamiltonian": "H_lambda=sum_i p_i^2+lambda*(sum_i p_i)^2",
        "first_relative_excitation_energy": 2.0,
        "first_nonzero_charge_energy_formula": "1+lambda",
        "scan": out,
    }


def soft_gap_scan() -> list[dict]:
    k = grid(4, 2)
    free_e = np.sum(k * k, axis=1)
    charge = np.sum(k, axis=1)
    rows = []
    for coupling in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        weights = np.exp(-BETA * (free_e + coupling * charge * charge))
        eigs = np.linalg.eigvalsh(qfi_from_weights(k, weights))
        rows.append({
            "lambda": coupling,
            "qfi_eigenvalues": np.round(eigs, 9).tolist(),
            "common_to_relative_eigenvalue_ratio": float(eigs[0] / np.median(eigs[1:])),
            "charged_weight": float(weights[charge != 0].sum() / weights.sum()),
        })
    return rows


def symmetric_branch_selection() -> dict:
    """Permutation-symmetric activation model; three active axes only in a branch."""
    activation_cost = -2.5
    pair_repulsion = 1.0
    inactive_gap = 4.0
    active_k = grid(3, 1)
    axis_qfi = float(np.diag(qfi_from_weights(
        active_k, np.exp(-BETA * np.sum(active_k * active_k, axis=1))
    ))[0])
    out = {}
    for n in (4, 5, 6):
        energies = np.array([activation_cost * m +
                             pair_repulsion * m * (m - 1) / 2 for m in range(n + 1)])
        ground_m = np.flatnonzero(np.isclose(energies, energies.min())).tolist()
        distributions = []
        for inverse_temp in (0.0, 2.0, 5.0, 10.0):
            weights = np.array([math.comb(n, m) *
                                math.exp(-inverse_temp * (energies[m] - energies.min()))
                                for m in range(n + 1)])
            probabilities = weights / weights.sum()
            distributions.append({
                "selection_inverse_temperature": inverse_temp,
                "probability_of_three_active_axes": float(probabilities[3]),
                "probabilities_by_active_count": probabilities.tolist(),
            })
        branch_qfi = np.diag([axis_qfi] * 3 + [0.0] * (n - 3))
        branch_average_qfi = np.eye(n) * axis_qfi * 3 / n
        out[f"n{n}"] = {
            "energy_by_active_count": energies.tolist(),
            "ground_active_counts": ground_m,
            "degenerate_three_axis_branches": math.comb(n, 3),
            "one_branch_qfi": eig_summary(branch_qfi),
            "branch_resolved_average_qfi": eig_summary(branch_average_qfi),
            "branch_probability_scan": distributions,
        }
    return {
        "activation_potential": "V(s)=a*m+b*m*(m-1)/2, m=sum_i s_i",
        "parameters": {"a": activation_cost, "b": pair_repulsion,
                       "inactive_excitation_penalty": inactive_gap},
        "three_active_axis_condition": "b>0 and -3*b<a<-2*b for n>=4",
        "within_branch_low_energy_window": "1<=E_probe<1+inactive_excitation_penalty",
        "interpretation_warning": "The parameter ratio is tuned to select 3. The orientation must additionally be conditioned or locked for longer than the observation time; V alone supplies no switching barrier. The branch-resolved average has rank n.",
        "by_n": out,
    }


def orbit_and_topology() -> dict:
    k = grid(4, 2)
    directions = {
        "closed_diagonal_integer": np.ones(4),
        "partly_irrational": np.array([1.0, math.sqrt(2), 0.0, 0.0]),
        "fully_rationally_independent": np.array([1.0, math.sqrt(2), math.sqrt(3), math.sqrt(6)]),
    }
    modes = {}
    for name, v in directions.items():
        invariant = np.abs(k @ v) < 1e-10
        modes[name] = {
            "flow_vector": v.tolist(),
            "invariant_fourier_modes_cutoff2": int(invariant.sum()),
            "invariant_qfi_rank": eig_summary(qfi_from_weights(k[invariant], np.ones(invariant.sum())))["rank"],
        }
    b = np.array([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1]])
    rng = np.random.default_rng(101)
    q = rng.uniform(-math.pi, math.pi, size=(1000, 4))
    alpha = rng.uniform(-math.pi, math.pi, size=(1000, 1))
    residual = np.max(np.abs(np.angle(np.exp(1j * ((q + alpha) @ b.T - q @ b.T)))))
    return {
        "quotient_map_B": b.tolist(),
        "B_rank": int(np.linalg.matrix_rank(b)),
        "diagonal_shift_invariance_max_error": float(residual),
        "kernel": "q -> q + alpha*(1,1,1,1) modulo 2pi",
        "exact_quotient": "T^4 / diagonal U(1) is T^3 via (q1-q4,q2-q4,q3-q4)",
        "flow_modes": modes,
    }


def state(k: np.ndarray, beta: float = BETA) -> tuple[np.ndarray, np.ndarray]:
    energies = np.sum(k * k, axis=1)
    weights = np.exp(-beta * energies)
    amplitudes = np.sqrt(weights / weights.sum())
    return energies, amplitudes


def shift_terms(k: np.ndarray, energies: np.ndarray, amplitudes: np.ndarray,
                shift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lookup = {tuple(row): i for i, row in enumerate(k)}
    products = []
    differences = []
    for index, row in enumerate(k):
        target = lookup.get(tuple(row + shift))
        if target is not None:
            products.append(amplitudes[index] * amplitudes[target])
            differences.append(energies[target] - energies[index])
    return np.asarray(products), np.asarray(differences)


def coherence(terms: tuple[np.ndarray, np.ndarray], tau: float, clock_sigma: float) -> complex:
    products, differences = terms
    factors = np.exp(1j * differences * tau - 0.5 * clock_sigma**2 * differences**2)
    return complex(np.sum(products * factors))


def measurement_fisher(k: np.ndarray, energies: np.ndarray, amplitudes: np.ndarray,
                       tau: float, clock_sigma: float, include_absolute: bool,
                       harmonics: tuple[int, ...] = (1,)) -> np.ndarray:
    n = k.shape[1]
    shifts = []
    for harmonic in harmonics:
        for i in range(n):
            for j in range(i + 1, n):
                delta = np.zeros(n, dtype=int)
                delta[i], delta[j] = harmonic, -harmonic
                shifts.append(delta)
    if include_absolute:
        delta = np.zeros(n, dtype=int)
        delta[0] = 1
        shifts.append(delta)
    fisher = np.zeros((n, n))
    for delta in shifts:
        c = coherence(shift_terms(k, energies, amplitudes, delta), tau, clock_sigma)
        for bias in (math.pi / 4, 3 * math.pi / 4):
            z = c * np.exp(1j * bias)
            p = (1.0 + z.real) / 2.0
            gradient = -0.5 * z.imag * delta
            fisher += np.outer(gradient, gradient) / (p * (1 - p))
    return fisher / (2 * len(shifts))


def clock_observables() -> dict:
    all_k = grid(4, 2)
    physical_k = all_k[np.sum(all_k, axis=1) == 0]
    physical_e, physical_a = state(physical_k)
    full_e, full_a = state(all_k)
    assert np.all(physical_e % 2 == 0)
    delta = np.array([1, -1, 0, 0])
    terms = shift_terms(physical_k, physical_e, physical_a, delta)
    tau_points = np.linspace(0, math.pi, 33)
    trajectory = []
    for tau in tau_points:
        c = coherence(terms, float(tau), 0.0)
        fisher = measurement_fisher(physical_k, physical_e, physical_a, float(tau), 0.0, False)
        multi = measurement_fisher(physical_k, physical_e, physical_a, float(tau), 0.0, False, (1, 2))
        all_harmonics = measurement_fisher(physical_k, physical_e, physical_a, float(tau), 0.0,
                                           False, (1, 2, 3, 4))
        trajectory.append({
            "tau": float(tau),
            "relative_coherence_real": float(c.real),
            "relative_coherence_imag": float(c.imag),
            "fisher_eigenvalues": np.round(np.linalg.eigvalsh(fisher), 9).tolist(),
            "fisher_rank_at_per_shot_threshold_0p01": int(np.sum(np.linalg.eigvalsh(fisher) > 0.01)),
            "two_harmonic_fisher_eigenvalues": np.round(np.linalg.eigvalsh(multi), 9).tolist(),
            "two_harmonic_rank_at_per_shot_threshold_0p01": int(np.sum(np.linalg.eigvalsh(multi) > 0.01)),
            "four_harmonic_fisher_eigenvalues": np.round(np.linalg.eigvalsh(all_harmonics), 9).tolist(),
            "four_harmonic_rank_at_per_shot_threshold_0p01": int(np.sum(np.linalg.eigvalsh(all_harmonics) > 0.01)),
        })
    fid = lambda e, a, tau: float(abs(np.sum(a * a * np.exp(-1j * e * tau))) ** 2)
    blur = []
    for sigma in (0.0, 0.2, 0.6, 1.2):
        fisher = measurement_fisher(physical_k, physical_e, physical_a, 0.0, sigma, False)
        blur.append({
            "clock_sigma": sigma,
            "pair_coherence_at_tau0": float(coherence(terms, 0.0, sigma).real),
            "fisher_eigenvalues_at_tau0": np.round(np.linalg.eigvalsh(fisher), 9).tolist(),
            "fisher_rank_at_per_shot_threshold_0p01": int(np.sum(np.linalg.eigvalsh(fisher) > 0.01)),
        })
    physical_fisher = measurement_fisher(physical_k, physical_e, physical_a, 0.0, 0.0, False)
    full_relative = measurement_fisher(all_k, full_e, full_a, 0.0, 0.0, False)
    full_augmented = measurement_fisher(all_k, full_e, full_a, 0.0, 0.0, True)
    physical_augmented = measurement_fisher(physical_k, physical_e, physical_a, 0.0, 0.0, True)
    return {
        "clock_constraint_residual": 0.0,
        "clock_energy_levels": int(len(np.unique(physical_e))),
        "conditional_state_fidelity_at_pi_over_2": fid(physical_e, physical_a, math.pi / 2),
        "conditional_state_fidelity_at_pi": fid(physical_e, physical_a, math.pi),
        "unrestricted_state_fidelity_at_pi": fid(full_e, full_a, math.pi),
        "physical_relative_detector_rank_at_tau0": eig_summary(physical_fisher)["rank"],
        "unrestricted_relative_detector_rank_at_tau0": eig_summary(full_relative)["rank"],
        "unrestricted_augmented_detector_rank_at_tau0": eig_summary(full_augmented)["rank"],
        "physical_augmented_detector_rank_at_tau0": eig_summary(physical_augmented)["rank"],
        "clock_blur_at_tau0": blur,
        "trajectory": trajectory,
    }


def clock_sensitivity() -> list[dict]:
    """Check whether conditional detector features survive truncation and preparation changes."""
    rows = []
    for cutoff in (1, 2, 3):
        all_k = grid(4, cutoff)
        physical_k = all_k[np.sum(all_k, axis=1) == 0]
        for beta in (0.2, 0.35, 0.8):
            energies, amplitudes = state(physical_k, beta)
            terms = shift_terms(physical_k, energies, amplitudes,
                                np.array([1, -1, 0, 0]))
            fisher_at_zero = measurement_fisher(physical_k, energies, amplitudes,
                                                0.0, 0.0, False)
            fisher_at_half = measurement_fisher(physical_k, energies, amplitudes,
                                                math.pi / 2, 0.0, False)
            fisher_at_half_multi = measurement_fisher(physical_k, energies, amplitudes,
                                                      math.pi / 2, 0.0, False, (1, 2))
            eigen_zero = np.linalg.eigvalsh(fisher_at_zero)[1:]
            eigen_half = np.linalg.eigvalsh(fisher_at_half)[1:]
            eigen_half_multi = np.linalg.eigvalsh(fisher_at_half_multi)[1:]
            recurrence = abs(np.sum(amplitudes**2 * np.exp(-1j * energies * math.pi)))**2
            rows.append({
                "cutoff": cutoff,
                "beta": beta,
                "physical_basis_size": len(physical_k),
                "relative_coherence_at_tau0": float(coherence(terms, 0.0, 0.0).real),
                "relative_coherence_at_pi_over_2": float(coherence(terms, math.pi / 2, 0.0).real),
                "fisher_nonzero_eigenvalue_at_tau0": float(np.median(eigen_zero)),
                "fisher_nonzero_eigenvalue_at_pi_over_2": float(np.median(eigen_half)),
                "two_harmonic_eigenvalue_at_pi_over_2": float(np.median(eigen_half_multi)),
                "conditional_fidelity_at_pi": float(recurrence),
            })
    return rows


def spectral_count() -> dict:
    k = grid(4, 8)
    e = np.sum(k * k, axis=1)
    gauge = np.sum(k, axis=1) == 0
    thresholds = np.arange(10, 51, 2)
    full_counts = np.array([np.sum(e <= cutoff) for cutoff in thresholds])
    gauge_counts = np.array([np.sum((e <= cutoff) & gauge) for cutoff in thresholds])
    full_slope = np.polyfit(np.log(thresholds), np.log(full_counts), 1)[0]
    gauge_slope = np.polyfit(np.log(thresholds), np.log(gauge_counts), 1)[0]
    return {
        "energy_thresholds": thresholds.tolist(),
        "unrestricted_counts": full_counts.tolist(),
        "diagonal_gauge_counts": gauge_counts.tolist(),
        "finite_range_log_count_slope_unrestricted": float(full_slope),
        "finite_range_log_count_slope_gauge": float(gauge_slope),
        "expected_asymptotic_slopes_for_T4_and_T3": [2.0, 1.5],
        "warning": "Finite-range fit of free-laplacian mode counts; not a measured cosmological spectral dimension.",
    }


def make_figure(results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for coupling, rows in results["hard_gap"]["scan"].items():
        axes[0].step([r["energy_limit"] for r in rows], [r["qfi"]["rank"] for r in rows],
                     where="post", label=f"lambda={coupling}")
    axes[0].set(xlabel="energy ceiling", ylabel="QFI rank", title="Low-energy dimension window", ylim=(-0.2, 4.3))
    axes[0].legend(fontsize=8)
    soft = results["soft_gap"]
    axes[1].semilogy([r["lambda"] for r in soft],
                     [r["common_to_relative_eigenvalue_ratio"] for r in soft], marker="o")
    axes[1].set(xlabel="collective penalty lambda", ylabel="common / relative QFI",
                title="Finite-gap crossover")
    trajectory = results["clock"]["trajectory"]
    axes[2].plot([r["tau"] for r in trajectory],
                 [r["relative_coherence_real"] for r in trajectory], marker=".")
    axes[2].set(xlabel="relational clock phase tau", ylabel="Re relative coherence",
                title="Conditional dynamics and recurrence")
    fig.tight_layout()
    fig.savefig(FIG, dpi=180)
    plt.close(fig)


def main() -> None:
    results = {
        "model": "T^n Fourier modes with symmetric collective-momentum penalty and an energy-matched finite clock",
        "beta": BETA,
        "interpretation_boundary": "Exact gauge, low-energy state support, and detector restriction are different claims.",
        "exact_sectors": exact_sectors(),
        "hard_gap": hard_gap_scan(),
        "soft_gap": soft_gap_scan(),
        "symmetric_branch_selection": symmetric_branch_selection(),
        "orbits_topology": orbit_and_topology(),
        "clock": clock_observables(),
        "clock_sensitivity": clock_sensitivity(),
        "spectral_count": spectral_count(),
    }
    assert results["exact_sectors"]["n4_cutoff1"]["gauge_qfi"]["rank"] == 3
    assert results["exact_sectors"]["n4_cutoff3"]["gauge_qfi"]["rank"] == 3
    assert results["exact_sectors"]["n5_cutoff1"]["gauge_qfi"]["rank"] == 4
    assert results["exact_sectors"]["n6_cutoff1"]["gauge_qfi"]["rank"] == 5
    assert results["clock"]["physical_relative_detector_rank_at_tau0"] == 3
    assert results["clock"]["unrestricted_augmented_detector_rank_at_tau0"] == 4
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    make_figure(results)
    print(json.dumps({
        "exact_ranks": {key: value["gauge_qfi"]["rank"] for key, value in results["exact_sectors"].items()
                        if "gauge_qfi" in value},
        "detector_ranks": {key: value for key, value in results["clock"].items() if "rank_at_tau0" in key},
        "spectral_slopes": [results["spectral_count"]["finite_range_log_count_slope_unrestricted"],
                            results["spectral_count"]["finite_range_log_count_slope_gauge"]],
        "results": str(OUT),
        "figure": str(FIG),
    }, indent=2))


if __name__ == "__main__":
    main()
