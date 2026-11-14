"""Independent stage-6 Aer seed and predeclared two-seed pooled analysis."""

from __future__ import annotations

import json
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np
from scipy.stats import chi2

from relational_atlas_i01.holdout import qiskit_rank_holdout6_clock as first_experiment


PROJECT = PROJECT_ROOT
FIRST = first_experiment.RESULT
REPLICATE = PROJECT / "results" / "qiskit_rank_holdout6_clock_replicate.json"
COMBINED = PROJECT / "results" / "qiskit_rank_holdout6_clock_combined.json"
PROTOCOL = "notes/prevalidation_protocol6b_2026-10-04.tex"
REPLICATION_SEED = 20261029
REPLICATION_BOOTSTRAP_SEED = 20261030
COMBINED_BOOTSTRAP_SEED = 20261031


def residual_audit(raw: dict) -> dict:
    joint = np.asarray(raw["raw_joint_zero_counts"], dtype=int)
    accepted = np.asarray(raw["raw_clock_accept_counts"], dtype=int)
    allocation = np.stack([
        np.asarray(raw["shot_allocations"][name], dtype=int)
        for name in first_experiment.previous.DISTRIBUTIONS
    ])[:, None, None, None, None, :]
    exact = first_experiment.exact_joint_probabilities()[None, :, :, :, :, :]
    joint_z = (joint - allocation * exact) / np.sqrt(
        allocation * exact * (1 - exact))
    clock_z = (accepted - allocation * 0.25) / np.sqrt(
        allocation * 0.25 * 0.75)
    def summary(z: np.ndarray) -> dict:
        value = float(np.sum(z * z))
        return {"mean": float(z.mean()), "standard_deviation": float(z.std()),
                "maximum_absolute": float(np.max(np.abs(z))),
                "pearson_sum": value, "circuit_count": int(z.size),
                "chi_square_tail_diagnostic_only": float(chi2.sf(value, z.size))}
    return {"joint_success": summary(joint_z),
            "clock_acceptance": summary(clock_z),
            "caveat": "Circuitwise residual diagnostic; shared job seeds and within-circuit joint/clock dependence are not modeled by the chi-square tail."}


def combine(first: dict, second: dict) -> dict:
    if first["parameters"]["shots_per_label_if_uniform"] != first_experiment.SHOTS_PER_LABEL:
        raise ValueError("first run has wrong shot budget")
    for name in first_experiment.previous.DISTRIBUTIONS:
        expected = first_experiment.previous.shot_allocation(
            first_experiment.SHOTS_PER_LABEL, name).tolist()
        if first["shot_allocations"][name] != expected or second[
                "shot_allocations"][name] != expected:
            raise ValueError("allocation does not match the frozen design")
    raw = {"raw_axes": first["raw_axes"]}
    for key in ("raw_joint_zero_counts", "raw_clock_accept_counts"):
        raw[key] = (np.asarray(first[key], dtype=int) +
                    np.asarray(second[key], dtype=int)).tolist()
    raw["shot_allocations"] = {name: (
        np.asarray(first["shot_allocations"][name], dtype=int) +
        np.asarray(second["shot_allocations"][name], dtype=int)).tolist()
        for name in first_experiment.previous.DISTRIBUTIONS}
    analysis = first_experiment.analyze(raw, first_experiment.BOOTSTRAPS,
                                        COMBINED_BOOTSTRAP_SEED)
    return {"protocol": PROTOCOL, "source_files": [str(FIRST), str(REPLICATE)],
            "source_aer_seeds": [first_experiment.SEED, REPLICATION_SEED],
            "combined_bootstrap_seed": COMBINED_BOOTSTRAP_SEED,
            "independent_repetition": True,
            "raw_axes": raw["raw_axes"],
            "raw_joint_zero_counts": raw["raw_joint_zero_counts"],
            "raw_clock_accept_counts": raw["raw_clock_accept_counts"],
            "shot_allocations": raw["shot_allocations"],
            "analysis": analysis,
            "first_residual_audit": residual_audit(first),
            "replicate_residual_audit": residual_audit(second),
            "combined_residual_audit": residual_audit(raw)}


def main() -> None:
    first = json.loads(FIRST.read_text(encoding="utf-8"))
    if first["parameters"]["aer_seed"] != first_experiment.SEED:
        raise ValueError("first run seed differs from protocol")
    checks = first_experiment.circuit_checks()
    raw = first_experiment.acquire(first_experiment.SHOTS_PER_LABEL, 6,
                                   REPLICATION_SEED)
    analysis = first_experiment.analyze(
        raw, first_experiment.BOOTSTRAPS, REPLICATION_BOOTSTRAP_SEED)
    second = {"protocol": PROTOCOL,
              "parameters": {"n": first_experiment.N,
                             "phase_labels": first_experiment.LABELS,
                             "shots_per_label_if_uniform":
                                 first_experiment.SHOTS_PER_LABEL,
                             "workers": 6, "aer_seed": REPLICATION_SEED,
                             "bootstrap_seed": REPLICATION_BOOTSTRAP_SEED,
                             "bootstraps": first_experiment.BOOTSTRAPS,
                             "circuits_executed": 336},
              "circuit_checks": checks, **raw, "analysis": analysis}
    REPLICATE.write_text(json.dumps(second, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    combined = combine(first, second)
    COMBINED.write_text(json.dumps(combined, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    def brief(data: dict) -> list:
        return [(row["label_distribution"], row["physical_P_zero"],
                 row["tau_over_pi"],
                 row["test"]["weak_null_bootstrap_pvalue"])
                for row in data["analysis"]["rows"]]
    print(json.dumps({"replicate_pvalues": brief(second),
                      "combined_pvalues": brief(combined),
                      "replicate_detections": analysis["detections"],
                      "combined_detections": combined["analysis"]["detections"],
                      "replicate": str(REPLICATE), "combined": str(COMBINED)},
                     indent=2))


if __name__ == "__main__":
    main()
