"""Post-shot technical audit: exact-binomial counts and label reweighting.

The reweighting analysis was added after the eight preregistered p-values were
seen. It is diagnostic only, not another confirmatory family of tests.
"""

from __future__ import annotations

import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np


PROJECT = PROJECT_ROOT
SOURCE = PROJECT / "results" / "qiskit_rank_holdout3_shots.json"
RESULT = PROJECT / "results" / "qiskit_rank_holdout3_audit.json"


def chi_square_three_survival(value: float) -> float:
    if value < 0:
        raise ValueError("chi-square must be nonnegative")
    return math.erfc(math.sqrt(value / 2)) + math.sqrt(2 * value / math.pi) * math.exp(
        -value / 2)


def audit(source: dict, panels: int = 5000, seed: int = 20261018) -> dict:
    counts = np.asarray(source["raw_zero_counts"], dtype=int)
    exact = np.asarray(source["exact_zero_probabilities"], dtype=float)
    names = ("uniform", "skewed")
    allocations = np.stack([source["shot_allocations"][name] for name in names])
    n = allocations[:, None, None, None, None, :]
    p = exact[None, :, :, :, :, :]
    standardized = (counts - n * p) / np.sqrt(n * p * (1 - p))
    observed_mean = float(standardized.mean())
    observed_variance = float(standardized.var(ddof=1))
    rng = np.random.default_rng(seed)
    null_means = []
    null_variances = []
    for start in range(0, panels, 250):
        batch = min(250, panels - start)
        sample = rng.binomial(n[None, :, :, :, :, :, :],
                              p[None, :, :, :, :, :, :],
                              size=(batch,) + counts.shape)
        z = (sample - n[None] * p[None]) / np.sqrt(
            n[None] * p[None] * (1 - p[None]))
        null_means.extend(z.mean(axis=(1, 2, 3, 4, 5, 6)).tolist())
        null_variances.extend(z.var(axis=(1, 2, 3, 4, 5, 6), ddof=1).tolist())
    null_means = np.asarray(null_means)
    null_variances = np.asarray(null_variances)
    reweighted = []
    labels = counts.shape[-1]
    for dist_index, distribution in enumerate(names):
        by_label_n = allocations[dist_index]
        for physical in (0, 1):
            for tau_index, tau in enumerate(source["parameters"]["tau_over_pi"]):
                first = counts[dist_index, physical, tau_index, 0]
                second = counts[dist_index, physical, tau_index, 1]
                first_rates = first / by_label_n
                second_rates = second / by_label_n
                estimated_uniform_difference = np.mean(
                    second_rates - first_rates, axis=1)
                exact_uniform_difference = np.mean(
                    exact[physical, tau_index, 1] -
                    exact[physical, tau_index, 0], axis=1)
                estimated_variance = np.sum(
                    first_rates * (1 - first_rates) / by_label_n +
                    second_rates * (1 - second_rates) / by_label_n,
                    axis=1) / (labels * labels)
                chi_square = float(np.sum(estimated_uniform_difference ** 2 /
                                          estimated_variance))
                reweighted.append({
                    "label_distribution": distribution,
                    "physical_P_zero": bool(physical), "tau_over_pi": tau,
                    "exact_uniform_target_max_difference": float(np.max(
                        np.abs(exact_uniform_difference))),
                    "observed_uniform_target_max_difference": float(np.max(
                        np.abs(estimated_uniform_difference))),
                    "per_axis_difference": estimated_uniform_difference.tolist(),
                    "per_axis_standard_error": np.sqrt(estimated_variance).tolist(),
                    "exploratory_chi_square_df3": chi_square,
                    "exploratory_asymptotic_pvalue": chi_square_three_survival(
                        chi_square),
                })
    return {
        "source": str(SOURCE), "caveat":
            "Reweighting and this calibration audit were added after observing the eight preregistered shot tests; diagnostic only.",
        "exact_binomial_panels": panels,
        "standardized_count_mean": observed_mean,
        "standardized_count_variance": observed_variance,
        "null_mean_95pct_interval": np.quantile(null_means, [0.025, 0.975]).tolist(),
        "null_variance_95pct_interval": np.quantile(null_variances,
                                                    [0.025, 0.975]).tolist(),
        "mean_two_sided_tail": float(np.mean(np.abs(null_means) >= abs(observed_mean))),
        "variance_upper_tail": float(np.mean(
            null_variances >= observed_variance)),
        "reweighted_uniform_target": reweighted,
    }


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    output = audit(source)
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({
        "standardized_count_mean": output["standardized_count_mean"],
        "standardized_count_variance": output["standardized_count_variance"],
        "mean_null_95pct": output["null_mean_95pct_interval"],
        "variance_null_95pct": output["null_variance_95pct_interval"],
        "reweighted": [(row["label_distribution"], row["physical_P_zero"],
                        row["tau_over_pi"], row["exploratory_asymptotic_pvalue"])
                       for row in output["reweighted_uniform_target"]],
        "result": str(RESULT),
    }, indent=2))


if __name__ == "__main__":
    main()
