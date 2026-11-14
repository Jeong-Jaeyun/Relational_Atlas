"""Post-hoc exact-model QC for random label counts and Aer output.

This audit is not another preregistered physical contrast. It separates
Bernoulli noise conditional on realized labels from multinomial label noise.
"""

from __future__ import annotations

import argparse
import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np
from scipy.stats import chi2

from relational_atlas_i01.holdout import qiskit_rank_holdout5_random_labels as experiment


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout5_allocation_audit.json"


def study(panels: int = 20000, seed: int = 20261025) -> dict:
    source = json.loads(experiment.RESULT.read_text(encoding="utf-8"))
    allocations = np.asarray(source["raw_label_counts"], dtype=int)
    counts = np.asarray(source["raw_zero_counts"], dtype=int)
    exact = np.asarray(source["exact_zero_probabilities"], dtype=float)
    probability = exact[None, :, :, :, :]
    standardized = (counts - allocations * probability) / np.sqrt(
        allocations * probability * (1 - probability))
    observed_mean = float(standardized.mean())
    observed_variance = float(standardized.var(ddof=1))
    rng = np.random.default_rng(seed)
    null_means = []
    null_variances = []
    for start in range(0, 5000, 250):
        size = min(250, 5000 - start)
        sample = rng.binomial(allocations[None], probability[None],
                              size=(size,) + counts.shape)
        z = (sample - allocations[None] * probability[None]) / np.sqrt(
            allocations[None] * probability[None] * (1 - probability[None]))
        null_means.extend(z.mean(axis=(1, 2, 3, 4, 5)).tolist())
        null_variances.extend(z.var(axis=(1, 2, 3, 4, 5), ddof=1).tolist())
    rows = []
    target = np.full(experiment.LABELS, 1 / experiment.LABELS)
    total_shots = experiment.TOTAL_SHOTS
    for physical in (0, 1):
        for tau_index, tau in enumerate((0.0, math.pi / 2)):
            p = exact[physical, tau_index]
            mean = p @ target
            iid_variance = mean * (1 - mean) / total_shots
            fixed_variance = np.sum(target[None, :] * p * (1 - p),
                                    axis=1) / total_shots
            first_n = rng.multinomial(total_shots, target,
                                      size=(panels, experiment.N))
            second_n = rng.multinomial(total_shots, target,
                                       size=(panels, experiment.N))
            first_y = rng.binomial(first_n, p[None, :, :])
            second_y = rng.binomial(second_n, p[None, :, :])
            first_rate = first_y.sum(axis=-1) / total_shots
            second_rate = second_y.sum(axis=-1) / total_shots
            delta = second_rate - first_rate
            correct_stat = np.sum(delta ** 2 / (2 * iid_variance), axis=1)
            naive_fixed_stat = np.sum(delta ** 2 / (2 * fixed_variance), axis=1)
            rows.append({
                "physical_P_zero": bool(physical), "tau_over_pi": tau / math.pi,
                "panels": panels,
                "mean_theoretical_random_to_fixed_variance_ratio": float(np.mean(
                    iid_variance / fixed_variance)),
                "mean_empirical_to_fixed_variance_ratio": float(np.mean(
                    first_rate.var(axis=0, ddof=1) / fixed_variance)),
                "mean_empirical_to_random_variance_ratio": float(np.mean(
                    first_rate.var(axis=0, ddof=1) / iid_variance)),
                "correct_random_design_rejections_at_0_point_05": int(np.sum(
                    chi2.sf(correct_stat, df=experiment.N) <= 0.05)),
                "naive_fixed_design_rejections_at_0_point_05": int(np.sum(
                    chi2.sf(naive_fixed_stat, df=experiment.N) <= 0.05)),
                "correct_random_design_rejections_at_0_point_00625": int(np.sum(
                    chi2.sf(correct_stat, df=experiment.N) <= 0.00625)),
                "naive_fixed_design_rejections_at_0_point_00625": int(np.sum(
                    chi2.sf(naive_fixed_stat, df=experiment.N) <= 0.00625)),
            })
    return {"source": str(experiment.RESULT),
            "caveat": "Post-hoc exact-model design and Aer-count quality audit; not an independent physical test.",
            "aer_standardized_count_mean": observed_mean,
            "aer_standardized_count_variance": observed_variance,
            "aer_exact_binomial_null_panels": 5000,
            "aer_null_mean_95pct_interval": np.quantile(null_means,
                                                         [0.025, 0.975]).tolist(),
            "aer_null_variance_95pct_interval": np.quantile(null_variances,
                                                             [0.025, 0.975]).tolist(),
            "random_allocation_panels_per_case": panels,
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    if args.panels < 1:
        parser.error("panels must be positive")
    output = study(args.panels, args.seed)
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
