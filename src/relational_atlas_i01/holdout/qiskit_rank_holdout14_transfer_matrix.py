"""Exact Aer 6-by-6 source-to-return support map for paired Rx leakage."""

from __future__ import annotations

import argparse
import json

import numpy as np

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout6_clock as stage6
from relational_atlas_i01.holdout import qiskit_rank_holdout7_noise_stress as stage7
from relational_atlas_i01.holdout import qiskit_rank_holdout12_intervention_atlas as atlas
from relational_atlas_i01.holdout._clock_analysis import condition_rows
from relational_atlas_i01.holdout._clock_postselection import CLOCK_ACCEPTANCE


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout14_transfer_matrix.json"
CHECKPOINT = PROJECT_ROOT / "results" / "qiskit_rank_holdout14_transfer_matrix_checkpoint.json"
PROTOCOL = "notes/prevalidation_protocol14_2026-10-04.tex"
SYSTEM_QUBITS = 6


def scenario_specs() -> tuple[tuple, ...]:
    """Freeze every transverse source-to-return support pair in lexicographic order."""
    return tuple((f"q{source}_to_q{return_}", "x", source, "x", return_, -1.0)
                 for source in range(SYSTEM_QUBITS)
                 for return_ in range(SYSTEM_QUBITS))


SCENARIOS = scenario_specs()


def conjugate_partner(qubit: int) -> int:
    """Return the opposite-sign qubit in a momentum-pair encoding."""
    if not 0 <= qubit < SYSTEM_QUBITS:
        raise ValueError("qubit is outside the system")
    return qubit ^ 1


def circuit_checks() -> dict:
    """Verify barrier bracketing and declared source-return supports for all 36 rows."""
    base, _ = stage7.compiled_templates()[(0, 0)]
    checked = []
    for spec in SCENARIOS:
        circuit = atlas.insert_intervention(base, spec, 0.1)
        data = list(circuit.data)
        barriers = [i for i, item in enumerate(data) if item.operation.name == "barrier"]
        rotations = [i for i, item in enumerate(data) if item.operation.name == "rx"]
        supports = [circuit.find_bit(data[i].qubits[0]).index for i in rotations]
        if rotations != [barriers[0] + 1, barriers[1] + 1] or supports != [spec[2], spec[4]]:
            raise AssertionError(f"invalid placement or support for {spec[0]}")
        checked.append(spec[0])
    return {"checked_scenarios": checked, "count": len(checked)}


def _constraint(rows: list[dict]) -> dict:
    return next(row for row in rows if row["role"] == "constraint_null_tau_0")


def _checkpoint_state() -> tuple[np.ndarray, dict[str, list[float]]]:
    """Load only a checkpoint matching this frozen transfer-matrix design."""
    if not CHECKPOINT.exists():
        return np.zeros((SYSTEM_QUBITS, SYSTEM_QUBITS)), {}
    saved = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    expected = [spec[0] for spec in SCENARIOS]
    if (saved.get("protocol") != PROTOCOL or
            saved.get("scenario_order") != expected):
        return np.zeros((SYSTEM_QUBITS, SYSTEM_QUBITS)), {}
    matrix = np.asarray(saved.get("matrix"), dtype=float)
    vectors = saved.get("axis_vectors", {})
    if matrix.shape != (SYSTEM_QUBITS, SYSTEM_QUBITS) or not isinstance(vectors, dict):
        return np.zeros((SYSTEM_QUBITS, SYSTEM_QUBITS)), {}
    return matrix, vectors


def _save_checkpoint(matrix: np.ndarray, axis_vectors: dict[str, list[float]]) -> None:
    """Persist every completed exact row to make the long atlas resumable."""
    CHECKPOINT.write_text(json.dumps({
        "protocol": PROTOCOL,
        "scenario_order": [spec[0] for spec in SCENARIOS],
        "matrix": matrix.tolist(),
        "axis_vectors": axis_vectors,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def study(workers: int) -> dict:
    """Calculate the complete exact-Aer source-return contrast matrix."""
    templates = stage7.compiled_templates()
    matrix, axis_vectors = _checkpoint_state()
    for spec in SCENARIOS:
        if spec[0] in axis_vectors:
            print(f"resumed row={spec[0]}", flush=True)
            continue
        joint, clock = atlas.full_axis_probabilities(spec, workers, templates)
        constraint = _constraint(condition_rows(joint, clock))
        source, return_ = spec[2], spec[4]
        matrix[source, return_] = constraint["maximum_absolute_joint_difference"]
        axis_vectors[spec[0]] = constraint["axis_pooled_joint_difference"]
        _save_checkpoint(matrix, axis_vectors)
        print(f"completed row={spec[0]}", flush=True)
    diagonal = np.diag(matrix)
    partner = np.asarray([matrix[q, conjugate_partner(q)] for q in range(SYSTEM_QUBITS)])
    off_pair = np.asarray([matrix[source, return_]
                           for source in range(SYSTEM_QUBITS)
                           for return_ in range(SYSTEM_QUBITS)
                           if return_ not in {source, conjugate_partner(source)}])
    return {
        "protocol": PROTOCOL,
        "parameters": {"workers": workers, "angle_radians": atlas.ANGLE,
                       "system_qubits": SYSTEM_QUBITS,
                       "all_axes_explicitly_evaluated": True,
                       "scenario_order": [spec[0] for spec in SCENARIOS]},
        "circuit_checks": circuit_checks(),
        "constraint_tau_0_maximum_contrast_matrix": matrix.tolist(),
        "constraint_tau_0_axis_pooled_differences": axis_vectors,
        "diagonal_same_support_range": [float(diagonal.min()), float(diagonal.max())],
        "conjugate_partner_range": [float(partner.min()), float(partner.max())],
        "off_pair_range": [float(off_pair.min()), float(off_pair.max())],
        "caveat": "Exact statevector transfer map of coherent interventions; not finite-shot evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.checks_only:
        print(json.dumps(circuit_checks(), indent=2))
        return
    output = study(args.workers)
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "diagonal_same_support_range": output["diagonal_same_support_range"],
        "conjugate_partner_range": output["conjugate_partner_range"],
        "off_pair_range": output["off_pair_range"],
        "matrix": output["constraint_tau_0_maximum_contrast_matrix"],
    }, indent=2))


if __name__ == "__main__":
    main()
