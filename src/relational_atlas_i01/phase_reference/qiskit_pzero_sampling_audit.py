"""Audit Aer P=0 common-phase null counts against the exact binomial model."""

from __future__ import annotations

import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np

from relational_atlas_i01.phase_reference import qiskit_clock_alias_audit as alias
from relational_atlas_i01.phase_reference import qiskit_clock_anchor_intervention as clock
from relational_atlas_i01.phase_reference import qiskit_phase_label_intervention as labels


PROJECT = PROJECT_ROOT
SOURCE = PROJECT / "results" / "qiskit_clock_alias_audit.json"
RESULT = PROJECT / "results" / "qiskit_pzero_sampling_audit.json"


def exact_joint_probabilities() -> np.ndarray:
    entry = alias.model(True)
    phase = alias.segment_phases()["P_zero_science_true"]
    return np.array([[
        [clock.conditional_probability_gradient(
            entry, tau, labels.angle(i, t, phase))[0] / clock.CLOCK_LEVELS
         for t in range(labels.L)] for i in range(labels.N)]
        for tau in alias.SHOT_TAUS])


def audit(source: dict, panels: int = 5000, seed: int = 20261011) -> dict:
    shots = source["parameters"]["shots_per_circuit"]
    counts = [np.asarray(source["shots"]["P_zero_joint_zero_counts"], dtype=int)]
    counts.extend(np.asarray(row["joint_zero_counts"], dtype=int)
                  for row in source["P_zero_control_repeats"])
    counts = np.stack(counts)  # seed, tau, arm, axis, label
    assert counts.shape[1:] == (2, 2, labels.N, labels.L)
    probability = exact_joint_probabilities()
    denominator = np.sqrt(2 * shots * probability * (1 - probability))
    differences = ((counts[:, :, 0] - counts[:, :, 1]) /
                   denominator[None, :, :, :])
    observed_variance = float(np.var(differences, ddof=1))
    observed_mean = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    null_variances = []
    null_means = []
    batches = math.ceil(panels / 250)
    for batch_index in range(batches):
        size = min(250, panels - 250 * batch_index)
        simulated = rng.binomial(
            shots, probability[None, None, :, None, :, :],
            size=(size, counts.shape[0], 2, 2, labels.N, labels.L))
        simulated_difference = ((simulated[:, :, :, 0] -
                                 simulated[:, :, :, 1]) /
                                denominator[None, None, :, :, :])
        null_variances.extend(np.var(simulated_difference,
                                     axis=(1, 2, 3, 4), ddof=1).tolist())
        null_means.extend(np.mean(simulated_difference,
                                  axis=(1, 2, 3, 4)).tolist())
    null_variances = np.asarray(null_variances)
    null_means = np.asarray(null_means)
    pvalues = [row["P_zero_hidden_mode_vs_absent_labeled"]["permutation_pvalue"]
               for row in source["shots"]["tau_results"]]
    pvalues.extend(p for row in source["P_zero_control_repeats"]
                   for p in row["pvalues_by_tau"])
    nominal_positives = int(np.sum(np.asarray(pvalues) <= 0.05))
    nominal_tail = float(sum(math.comb(len(pvalues), k) * 0.05 ** k *
                             0.95 ** (len(pvalues) - k)
                             for k in range(nominal_positives, len(pvalues) + 1)))
    return {
        "source": str(SOURCE), "seed_count": counts.shape[0],
        "shots_per_circuit": shots,
        "standardized_difference_definition": "(arm_true-arm_absent)/sqrt(2*N*p*(1-p)) per tau, axis, label",
        "observed_mean_standardized_difference": observed_mean,
        "observed_variance_standardized_difference": observed_variance,
        "exact_binomial_null_panels": panels,
        "null_variance_95pct_interval": np.quantile(null_variances,
                                                     [0.025, 0.975]).tolist(),
        "null_mean_95pct_interval": np.quantile(null_means,
                                                 [0.025, 0.975]).tolist(),
        "empirical_upper_tail_for_variance": float(np.mean(
            null_variances >= observed_variance)),
        "empirical_two_sided_tail_for_mean": float(np.mean(
            np.abs(null_means) >= abs(observed_mean))),
        "nominal_p_below_0_point_05": nominal_positives,
        "number_of_p_tests": len(pvalues),
        "binomial_tail_for_nominal_positives_if_independent": nominal_tail,
        "caveat": "The binomial tail assumes independent tests; it is descriptive, not a diagnosis of an Aer defect.",
    }


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    output = audit(source)
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
