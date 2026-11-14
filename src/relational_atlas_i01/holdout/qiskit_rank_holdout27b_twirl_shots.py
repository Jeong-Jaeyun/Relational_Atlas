"""Stage 27b: finite-shot twirl on/off family for the paired-access contrast.

Frozen in notes/research_protocol27b_twirl_shots_2026-11-13.md, committed
before this module.  Three conditions per support class on the cutoff-one
circuit: plus (sigma=+1), minus (sigma=-1) and twirl (independent random sigma
per shot; the per-sign record is stored but discarded by the confirmatory
analysis).  A calibration audit runs before acquisition.  Failed predictions
are reported, never raised.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
from qiskit_aer import AerSimulator

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout3_audit as audit3
from relational_atlas_i01.holdout import qiskit_rank_holdout3_shots as shots3
from relational_atlas_i01.holdout import qiskit_rank_holdout4b_weaknull as weak
from relational_atlas_i01.holdout import qiskit_rank_holdout27a_exact_design as design


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout27b_twirl_shots.json"
CHECKPOINT = PROJECT_ROOT / "results" / "qiskit_rank_holdout27b_twirl_shots_checkpoint.json"
PROTOCOL = "notes/research_protocol27b_twirl_shots_2026-11-13.md"
N, LABELS = design.N, design.LABELS
ANGLE = math.pi / 2
SHOTS_PER_LABEL = 1 << 20
CLASSES = design.CLASSES
CONDITIONS = ("plus", "minus", "twirl")
AER_SEED = 20261127
SPLIT_SEED = 20261128
BOOTSTRAP_SEED = 20261129
CALIBRATION_SEED = 20261130
BOOTSTRAPS = 10000
FAMILY_THRESHOLD = 0.05 / 6
NULL_LEVEL = 0.05
CALIBRATION_PANELS = 2000
CALIBRATION_WEAK_PANELS = 200
CALIBRATION_WEAK_BOOTSTRAPS = 999
POSITIVE = (
    ("S1", "same_support", "plus", "minus"),
    ("S2", "different_pair_same_sign", "plus", "minus"),
    ("S3", "different_pair_opposite_sign", "plus", "minus"),
    ("W1", "same_support", "plus", "twirl"),
    ("W2", "different_pair_same_sign", "plus", "twirl"),
    ("W3", "different_pair_opposite_sign", "plus", "twirl"),
)
NULLS = (
    ("Z1", "same_support", "twirl"),
    ("Z2", "different_pair_same_sign", "twirl"),
    ("Z3", "different_pair_opposite_sign", "twirl"),
    ("N1", "conjugate_partner", "plus"),
    ("N2", "conjugate_partner", "minus"),
    ("N3", "conjugate_partner", "twirl"),
)
FROZEN_LAMBDA = {"S1": 668.7, "S2": 173.0, "S3": 502.4,
                 "W1": 149.9, "W2": 41.3, "W3": 117.5}
ALLOWED = ("same_support", "different_pair_same_sign",
           "different_pair_opposite_sign")


def allocation() -> np.ndarray:
    return shots3.shot_allocation(SHOTS_PER_LABEL, "skewed")


def cell_shots() -> np.ndarray:
    return np.broadcast_to(allocation(), (2, N, LABELS)).astype(np.int64)


def design_weights() -> np.ndarray:
    alloc = allocation()
    return alloc / alloc.sum()


def exact_probabilities() -> dict:
    prepared, reference = design.sector.vectors(N, design.BETA, design.REFERENCE_BETA)
    output = {}
    for name, source, return_ in CLASSES:
        plus = design.signed_cutoff_one(source, return_, ANGLE, 1, prepared, reference)
        minus = design.signed_cutoff_one(source, return_, ANGLE, -1, prepared, reference)
        output[name] = {"plus": plus, "minus": minus, "twirl": (plus + minus) / 2}
    return output


def contrast(counts: np.ndarray, shots: np.ndarray,
             weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label-pooled block contrast and variance; leading axes broadcast."""
    rates = counts / shots
    delta = np.sum(weights * (rates[..., 1, :, :] - rates[..., 0, :, :]), axis=-1)
    variance = np.sum(weights ** 2 * (
        rates[..., 1, :, :] * (1 - rates[..., 1, :, :]) / shots[..., 1, :, :] +
        rates[..., 0, :, :] * (1 - rates[..., 0, :, :]) / shots[..., 0, :, :]),
        axis=-1)
    return delta, variance


def chi3_survival(values) -> np.ndarray:
    return np.vectorize(audit3.chi_square_three_survival)(np.asarray(values, dtype=float))


def wald(delta_a, var_a, delta_b, var_b) -> tuple[np.ndarray, np.ndarray]:
    statistic = np.sum((np.asarray(delta_a) - delta_b) ** 2 /
                       (np.asarray(var_a) + var_b), axis=-1)
    return statistic, chi3_survival(statistic)


def exact_contrast(probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Contrast and variance at the design budget from exact probabilities."""
    shots = cell_shots()
    return contrast(probability * shots, shots, design_weights())


def exact_lambda(exact: dict) -> dict:
    output = {}
    for test, name, first, second in POSITIVE:
        delta_a, var_a = exact_contrast(exact[name][first])
        delta_b, var_b = exact_contrast(exact[name][second])
        output[test] = float(np.sum((delta_a - delta_b) ** 2 / (var_a + var_b)))
    return output


def split_table() -> dict:
    """Per-cell number of sigma=+1 shots in the twirl condition."""
    rng = np.random.default_rng(SPLIT_SEED)
    shots = cell_shots()
    return {name: rng.binomial(shots, 0.5) for name, _, _ in CLASSES}


def seed_for(class_index: int, condition_index: int, label: int,
             block: int, axis: int, sign_index: int) -> int:
    return (AER_SEED + 1_000_000 * class_index + 100_000 * condition_index +
            1_000 * label + 10 * (3 * block + axis) + sign_index)


# --------------------------------------------------------------------------
# Calibration audit (before acquisition)
# --------------------------------------------------------------------------

def calibration(exact: dict) -> dict:
    rng = np.random.default_rng(CALIBRATION_SEED)
    shots, weights, alloc = cell_shots(), design_weights(), allocation()
    comparison = {}
    for name in ALLOWED:
        probability = exact[name]["plus"]
        size = (CALIBRATION_PANELS, 2, N, LABELS)
        first = rng.binomial(shots, probability, size=size)
        second = rng.binomial(shots, probability, size=size)
        delta_a, var_a = contrast(first, shots, weights)
        delta_b, var_b = contrast(second, shots, weights)
        _, pvalues = wald(delta_a, var_a, delta_b, var_b)
        rate = float(np.mean(pvalues <= 0.05))
        comparison[name] = {
            "false_positive_rate_at_0.05": rate,
            "false_positive_rate_at_family_threshold": float(
                np.mean(pvalues <= FAMILY_THRESHOLD)),
            "passed": bool(0.035 <= rate <= 0.065),
        }
    weak_rows = {}
    for label, name, condition in (("conjugate_plus", "conjugate_partner", "plus"),
                                   ("same_support_twirl", "same_support", "twirl")):
        probability = exact[name][condition]
        pvalues = []
        for _ in range(CALIBRATION_WEAK_PANELS):
            first = rng.binomial(alloc, probability[0])
            second = rng.binomial(alloc, probability[1])
            test = weak.weak_null_test(first, second, alloc,
                                       CALIBRATION_WEAK_BOOTSTRAPS, rng)
            pvalues.append(test["weak_null_bootstrap_pvalue"])
        rate = float(np.mean(np.asarray(pvalues) <= 0.05))
        weak_rows[label] = {"false_positive_rate_at_0.05": rate,
                            "passed": bool(0.02 <= rate <= 0.09)}
    return {
        "comparison_test": comparison,
        "weak_null_test": weak_rows,
        "comparison_passed": all(row["passed"] for row in comparison.values()),
        "weak_null_passed": all(row["passed"] for row in weak_rows.values()),
    }


# --------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------

def _checkpoint_header() -> dict:
    return {"shots_per_label": SHOTS_PER_LABEL, "aer_seed": AER_SEED,
            "split_seed": SPLIT_SEED, "angle": ANGLE}


def _load() -> dict:
    if not CHECKPOINT.exists():
        return {}
    saved = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if saved.get("header") != _checkpoint_header():
        return {}
    return saved.get("tables", {})


def _save(tables: dict) -> None:
    CHECKPOINT.write_text(json.dumps({"header": _checkpoint_header(),
                                      "tables": tables}), encoding="utf-8")


def acquire(workers: int) -> dict:
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers)
    base, theta, _ = design.model15.template(N)
    shots = cell_shots()
    splits = split_table()
    tables = _load()
    started = time.time()
    for class_index, (name, source, return_) in enumerate(CLASSES):
        for condition_index, condition in enumerate(CONDITIONS):
            key = f"{name}/{condition}"
            if key in tables:
                print(f"resumed {key}", flush=True)
                continue
            if condition == "twirl":
                by_sign = {1: splits[name], -1: shots - splits[name]}
            else:
                by_sign = {1 if condition == "plus" else -1: shots}
            joint = {sign: np.zeros((2, N, LABELS), dtype=np.int64) for sign in by_sign}
            clock = {sign: np.zeros((2, N, LABELS), dtype=np.int64) for sign in by_sign}
            for block, axis, label, angles in design.cutoff_one_settings():
                assigned = base.assign_parameters(
                    {theta[i]: float(angles[i]) for i in range(N)})
                for sign_index, (sign, table) in enumerate(by_sign.items()):
                    count = int(table[block, axis, label])
                    if count == 0:
                        continue
                    circuit = design.insert_at_barriers(
                        assigned, design.pulses(source, return_, ANGLE, sign))
                    circuit.measure_all()
                    result = simulator.run(
                        circuit, shots=count,
                        seed_simulator=seed_for(class_index, condition_index, label,
                                                block, axis, sign_index)).result()
                    if not result.success:
                        raise RuntimeError(result.status)
                    counts = result.get_counts()
                    joint[sign][block, axis, label] = counts.get("00000000", 0)
                    clock[sign][block, axis, label] = sum(
                        value for bits, value in counts.items() if bits[:2] == "00")
            tables[key] = {
                "shots_by_sign": {str(s): t.tolist() for s, t in by_sign.items()},
                "joint_by_sign": {str(s): t.tolist() for s, t in joint.items()},
                "clock_by_sign": {str(s): t.tolist() for s, t in clock.items()},
            }
            _save(tables)
            print(f"completed {key} ({time.time() - started:.0f} s)", flush=True)
    return tables


def pooled(table: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shots = sum(np.asarray(v, dtype=np.int64) for v in table["shots_by_sign"].values())
    joint = sum(np.asarray(v, dtype=np.int64) for v in table["joint_by_sign"].values())
    clock = sum(np.asarray(v, dtype=np.int64) for v in table["clock_by_sign"].values())
    return joint, clock, shots


def total_preparations(tables: dict) -> int:
    return int(sum(pooled(table)[2].sum() for table in tables.values()))


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def analyze(tables: dict, exact: dict, calibrated: dict) -> dict:
    weights, alloc = design_weights(), allocation()
    estimates, exact_contrasts, acceptance = {}, {}, {}
    for name, _, _ in CLASSES:
        for condition in CONDITIONS:
            joint, clock, shots = pooled(tables[f"{name}/{condition}"])
            if np.any(joint > clock):
                raise AssertionError("joint count exceeds clock acceptance")
            delta, variance = contrast(joint, shots, weights)
            estimates[(name, condition)] = (delta, variance)
            exact_contrasts[(name, condition)] = exact_contrast(
                exact[name][condition])[0]
            acceptance[f"{name}/{condition}"] = float(clock.sum() / shots.sum())

    positive = {}
    for test, name, first, second in POSITIVE:
        statistic, pvalue = wald(*estimates[(name, first)], *estimates[(name, second)])
        positive[test] = {
            "class": name, "comparison": f"{first} vs {second}",
            "wald_chi_square_df3": float(statistic), "pvalue": float(pvalue),
            "observed_difference": (estimates[(name, first)][0] -
                                    estimates[(name, second)][0]).tolist(),
            "exact_difference": (exact_contrasts[(name, first)] -
                                 exact_contrasts[(name, second)]).tolist(),
            "passed": bool(pvalue <= FAMILY_THRESHOLD),
        }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    null = {}
    for test, name, condition in NULLS:
        joint, _, shots = pooled(tables[f"{name}/{condition}"])
        if np.any(shots != alloc):
            raise AssertionError("pooled shots differ from the design allocation")
        result = weak.weak_null_test(joint[0], joint[1], alloc, BOOTSTRAPS, rng)
        null[test] = {
            "class": name, "condition": condition,
            "weak_null_bootstrap_pvalue": result["weak_null_bootstrap_pvalue"],
            "wald_chi_square_df3": result["wald_chi_square_df3"],
            "chi_square_df3_pvalue_crosscheck":
                result["chi_square_df3_pvalue_crosscheck"],
            "observed_contrast": result["axis_differences"],
            "exact_contrast": exact_contrasts[(name, condition)].tolist(),
            "passed": bool(result["weak_null_bootstrap_pvalue"] > NULL_LEVEL),
        }

    law = design.sector.angle_law_coefficients(N, design.BETA, design.REFERENCE_BETA)
    direction = {}
    for index, name in enumerate(ALLOWED, start=1):
        observed = estimates[(name, "plus")][0] - estimates[(name, "minus")][0]
        interference = np.asarray(law[name]["cross_coefficient_C"])
        product = float(observed @ interference)
        direction[f"D{index}"] = {"class": name, "inner_product": product,
                                  "passed": bool(product > 0)}

    kept = {}
    for name in ALLOWED:
        table = tables[f"{name}/twirl"]
        by_sign = {}
        for sign in ("1", "-1"):
            counts = np.asarray(table["joint_by_sign"][sign], dtype=np.int64)
            shots = np.asarray(table["shots_by_sign"][sign], dtype=np.int64)
            by_sign[sign] = contrast(counts, shots, weights)
        statistic, pvalue = wald(*by_sign["1"], *by_sign["-1"])
        kept[name] = {"wald_chi_square_df3": float(statistic), "pvalue": float(pvalue),
                      "observed_sign_difference":
                          (by_sign["1"][0] - by_sign["-1"][0]).tolist()}

    residuals = {
        f"{name}/{condition}": ((delta - exact_contrasts[(name, condition)]) /
                                np.sqrt(variance)).tolist()
        for (name, condition), (delta, variance) in estimates.items()}
    projection = {
        name: float(estimates[(name, "twirl")][0] @ estimates[(name, "plus")][0] /
                    (estimates[(name, "plus")][0] @ estimates[(name, "plus")][0]))
        for name in ALLOWED}

    withheld = []
    if not calibrated["comparison_passed"]:
        withheld += list(positive)
    if not calibrated["weak_null_passed"]:
        withheld += list(null)
    return {
        "positive_family": positive,
        "null_predictions": null,
        "secondary_direction": direction,
        "family_threshold": FAMILY_THRESHOLD,
        "null_level": NULL_LEVEL,
        "all_positive_passed": all(row["passed"] for row in positive.values()),
        "all_null_passed": all(row["passed"] for row in null.values()),
        "all_secondary_passed": all(row["passed"] for row in direction.values()),
        "withheld_by_calibration": withheld,
        "exploratory": {
            "twirl_with_record_kept_plus_vs_minus": kept,
            "standardized_residuals_vs_exact": residuals,
            "twirl_projection_ratio_onto_plus": projection,
            "clock_acceptance": acceptance,
        },
        "total_preparations": total_preparations(tables),
    }


def checks() -> dict:
    exact = exact_probabilities()
    lambdas = exact_lambda(exact)
    mismatch = {test: (value, FROZEN_LAMBDA[test]) for test, value in lambdas.items()
                if abs(value - FROZEN_LAMBDA[test]) > 0.06}
    if mismatch:
        raise AssertionError(f"design noncentrality differs from protocol: {mismatch}")
    if int(cell_shots().sum()) * len(CLASSES) * len(CONDITIONS) != 528_482_304:
        raise AssertionError("frozen budget changed")
    return {"exact_lambda": lambdas}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    design_checks = checks()
    if args.checks_only:
        print(json.dumps(design_checks, indent=2))
        return
    exact = exact_probabilities()
    calibrated = calibration(exact)
    print(json.dumps({"calibration": calibrated}, indent=2), flush=True)
    tables = acquire(args.workers)
    analysis = analyze(tables, exact, calibrated)
    output = {
        "protocol": PROTOCOL,
        "parameters": {
            "n": N, "labels": LABELS, "angle": ANGLE,
            "shots_per_label": SHOTS_PER_LABEL,
            "allocation": allocation().tolist(), "conditions": CONDITIONS,
            "aer_seed": AER_SEED, "split_seed": SPLIT_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED, "calibration_seed": CALIBRATION_SEED,
            "bootstraps": BOOTSTRAPS, "workers": args.workers,
        },
        "design_checks": design_checks,
        "calibration": calibrated,
        "twirl_splits": {name: table.tolist() for name, table in split_table().items()},
        "tables": tables,
        "analysis": analysis,
    }
    RESULT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "positive": {k: (round(v["pvalue"], 6), v["passed"])
                     for k, v in analysis["positive_family"].items()},
        "null": {k: (round(v["weak_null_bootstrap_pvalue"], 5), v["passed"])
                 for k, v in analysis["null_predictions"].items()},
        "secondary": {k: v["passed"] for k, v in analysis["secondary_direction"].items()},
        "withheld": analysis["withheld_by_calibration"],
        "total_preparations": analysis["total_preparations"],
    }, indent=2))


if __name__ == "__main__":
    main()
