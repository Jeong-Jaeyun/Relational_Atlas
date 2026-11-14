"""Attribute conjugate-partner readout leakage to individual bit outcomes."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from qiskit_aer import AerSimulator

from relational_atlas_i01.paths import PROJECT_ROOT
from relational_atlas_i01.holdout import qiskit_rank_holdout7_noise_stress as noise
from relational_atlas_i01.holdout import qiskit_rank_holdout19_transfer_noise as transfer


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout22_readout_paths.json"
LEVELS = (0.001, 0.01, 0.03)


def acquire(workers: int) -> np.ndarray:
    """Get all 256 ideal output probabilities for the partner intervention."""
    if workers < 1:
        raise ValueError("workers must be positive")
    base, theta, _ = transfer.circuit_model.template(3)
    simulator = AerSimulator(method="statevector", max_parallel_threads=workers,
                             max_parallel_experiments=workers)
    probabilities = np.zeros((2, 3, 7, 256))
    for label in range(7):
        circuits, settings = [], []
        for block in range(2):
            for axis in range(3):
                circuit = transfer.build_circuit(base, theta, "ideal", 0.0,
                                                 0, 1, block, axis, label)
                circuit.save_probabilities()
                circuits.append(circuit)
                settings.append((block, axis))
        result = simulator.run(circuits, shots=1).result()
        if not result.success:
            raise RuntimeError(result.status)
        for i, (block, axis) in enumerate(settings):
            probabilities[block, axis, label] = np.asarray(
                result.data(i)["probabilities"], dtype=float)
    if np.max(abs(probabilities.sum(axis=-1) - 1)) > 1e-10:
        raise AssertionError("output distributions are not normalized")
    return probabilities


def contrasts(probabilities: np.ndarray) -> dict:
    weights = transfer.circuit_model.frozen_label_weights(7)
    per_bit = {}
    for qubit in range(8):
        delta = (probabilities[1, :, :, 1 << qubit] -
                 probabilities[0, :, :, 1 << qubit]) @ weights
        per_bit[str(qubit)] = delta.tolist()
    direct = (probabilities[1, :, :, 0] - probabilities[0, :, :, 0]) @ weights
    derivative = np.sum(np.asarray(list(per_bit.values())), axis=0) - 8 * direct
    levels = {}
    for level in LEVELS:
        measured = np.zeros((2, 3, 7))
        for block in range(2):
            for axis in range(3):
                for label in range(7):
                    measured[block, axis, label] = noise.readout_events(
                        probabilities[block, axis, label], level)[0]
        delta = (measured[1] - measured[0]) @ weights
        levels[str(level)] = {
            "axis_pooled_joint_difference": delta.tolist(),
            "maximum_absolute_joint_difference": float(np.max(abs(delta))),
            "linear_prediction": (level * derivative).tolist(),
            "maximum_linear_approximation_error": float(np.max(
                abs(delta - level * derivative))),
        }
    return {
        "direct_zero_outcome_difference": direct.tolist(),
        "single_one_outcome_contributions_by_qubit": per_bit,
        "readout_first_derivative_at_zero": derivative.tolist(),
        "levels": levels,
    }


def _stage19_error(levels: dict) -> float | None:
    if not transfer.RESULT.exists():
        return None
    reference = json.loads(transfer.RESULT.read_text(encoding="utf-8"))[
        "scenarios"]
    error = 0.0
    for level in LEVELS:
        key = transfer.scenario_key("symmetric_readout", level)
        if key not in reference:
            return None
        predicted = levels[str(level)]["axis_pooled_joint_difference"]
        actual = reference[key]["analysis"]["conjugate_partner"][
            "axis_pooled_joint_difference"]
        error = max(error, float(np.max(abs(np.asarray(predicted) - actual))))
    return error


def study(workers: int) -> dict:
    values = acquire(workers)
    analysis = contrasts(values)
    output = {
        "parameters": {"n": 3, "source": 0, "return": 1,
                       "angle_radians": math.pi / 2, "workers": workers,
                       "readout_flip_levels": LEVELS},
        "analysis": analysis,
        "maximum_difference_from_stage19": _stage19_error(analysis["levels"]),
    }
    RESULT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    output = study(args.workers)
    print(json.dumps({
        "readout_first_derivative_at_zero": output["analysis"][
            "readout_first_derivative_at_zero"],
        "stage19_reproduction_error": output[
            "maximum_difference_from_stage19"],
    }, indent=2))


if __name__ == "__main__":
    main()
