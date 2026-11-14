"""Exact finite-n audit of the symmetric three-active-axis toy.

This code tests whether a nonzero parameter basin translates into stable
three-axis observations at finite selection temperature and under single-site
Metropolis updates.  It is a diagnostic of a deliberately phenomenological
activation potential, not a microscopic theory of physical space.
"""

from __future__ import annotations

import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp


ROOT = PROJECT_ROOT
OUTPUT = ROOT / "results" / "activation_stability.json"
FIGURE = ROOT / "results" / "activation_stability.png"


def energy(active: np.ndarray, ratio: float) -> np.ndarray:
    """V(m)/b at b=1 and ratio r=-a/b."""
    return -ratio * active + active * (active - 1) / 2


def count_probabilities(n: int, beta: float, ratio: float) -> np.ndarray:
    counts = np.arange(n + 1)
    log_multiplicity = gammaln(n + 1) - gammaln(counts + 1) - gammaln(n - counts + 1)
    logits = log_multiplicity - beta * energy(counts, ratio)
    return np.exp(logits - logsumexp(logits))


def first_exit_time_from_three(n: int, beta: float, ratio: float) -> float:
    """Per-site Poisson attempt rate 1, Metropolis acceptance for one spin flip."""
    if n < 4:
        raise ValueError("n must be at least 4")
    down_gap = ratio - 2.0
    up_gap = 3.0 - ratio
    down_rate = 3 * math.exp(-beta * max(down_gap, 0.0))
    up_rate = (n - 3) * math.exp(-beta * max(up_gap, 0.0))
    return 1.0 / (down_rate + up_rate)


def finite_temperature_basin(n: int, beta: float, threshold: float = 0.9) -> dict:
    ratios = np.linspace(2.0, 3.0, 2001)
    probabilities = np.array([count_probabilities(n, beta, float(r))[3] for r in ratios])
    mask = probabilities >= threshold
    if np.any(mask):
        interval = [float(ratios[np.flatnonzero(mask)[0]]),
                    float(ratios[np.flatnonzero(mask)[-1]])]
        width = interval[1] - interval[0]
    else:
        interval = None
        width = 0.0
    return {
        "n": n,
        "selection_inverse_temperature": beta,
        "probability_threshold": threshold,
        "ground_state_ratio_interval": [2.0, 3.0],
        "finite_temperature_interval_reaching_threshold": interval,
        "finite_temperature_interval_width": width,
        "maximum_probability_of_three": float(probabilities.max()),
        "ratio_of_maximum": float(ratios[np.argmax(probabilities)]),
    }


def beta_for_probability(n: int, target: float = 0.9, ratio: float = 2.5) -> float:
    function = lambda beta: float(count_probabilities(n, beta, ratio)[3] - target)
    if function(0.0) >= 0:
        return 0.0
    return float(brentq(function, 0.0, 100.0))


def make_figure(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ratios = np.linspace(1.5, 3.5, 400)
    for n in (4, 6, 20, 50):
        probabilities = [count_probabilities(n, 10.0, float(r))[3] for r in ratios]
        axes[0].plot(ratios, probabilities, label=f"n={n}")
    axes[0].axvspan(2, 3, color="gray", alpha=0.12, label="zero-T m=3 basin")
    axes[0].set(xlabel="r=-a/b", ylabel="P(active count = 3)",
                title="Finite-temperature basin at beta=10", ylim=(0, 1.02))
    axes[0].legend(fontsize=8)
    inverse_temps = np.linspace(0.5, 20.0, 150)
    for n in (4, 6, 20, 50):
        axes[1].semilogy(inverse_temps,
                         [first_exit_time_from_three(n, float(beta), 2.5)
                          for beta in inverse_temps], label=f"n={n}")
    axes[1].axhline(100, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="selection inverse temperature beta",
                ylabel="mean time to leave m=3 (site-attempt units)",
                title="Single-flip stability at r=2.5")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)


def main() -> None:
    ns = (4, 6, 9, 20, 50)
    betas = (2.0, 5.0, 10.0, 20.0)
    rows = []
    for n in ns:
        critical_beta = beta_for_probability(n)
        for beta in betas:
            rows.append({
                "n": n,
                "selection_inverse_temperature": beta,
                "ratio": 2.5,
                "probability_three_active": float(count_probabilities(n, beta, 2.5)[3]),
                "mean_time_to_leave_three": first_exit_time_from_three(n, beta, 2.5),
                "inverse_temperature_for_90_percent_three_at_ratio_2p5": critical_beta,
                "inverse_temperature_for_100_site_attempt_exit_time": 2 * math.log(100 * n),
            })
    basins = [finite_temperature_basin(n, beta) for n in ns for beta in (5.0, 10.0, 20.0)]
    results = {
        "model": "V(m)/b=-r*m+m*(m-1)/2, r=-a/b, b>0",
        "zero_temperature_three_axis_interval": "2<r<3 (finite width 1); no singled-out axis",
        "branch_count": {str(n): math.comb(n, 3) for n in ns},
        "thermal_count_distribution": "P(m)=C(n,m)*exp(-beta*V(m))/Z",
        "dynamics": "Continuous-time Metropolis single-site attempts, rate 1 per site",
        "exit_formula_at_r_2p5": "mean first exit time from m=3 = exp(beta/2)/n",
        "warning": "First exit from m=3 is not the orientation-switching time or a cosmological lifetime. The activation potential does not derive r from microscopic physics.",
        "center_scan": rows,
        "thermal_basin_scan": basins,
    }
    assert all(count_probabilities(n, 10.0, 2.5)[3] > 0.9 for n in ns)
    assert all(abs(first_exit_time_from_three(n, 10.0, 2.5) -
                   math.exp(5) / n) < 1e-9 for n in ns)
    OUTPUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    make_figure(results)
    print(json.dumps({
        "p_three_at_beta5": {str(n): float(count_probabilities(n, 5.0, 2.5)[3]) for n in ns},
        "beta_for_p_three_0p9": {str(n): beta_for_probability(n) for n in ns},
        "beta_for_exit_time_100": {str(n): 2 * math.log(100 * n) for n in ns},
        "results": str(OUTPUT),
        "figure": str(FIGURE),
    }, indent=2))


if __name__ == "__main__":
    main()
