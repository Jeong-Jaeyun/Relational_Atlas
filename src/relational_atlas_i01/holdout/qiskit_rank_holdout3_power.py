"""Pre-Aer power pilot using protocol-2 exact probabilities, not shot data."""

from __future__ import annotations

import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import numpy as np

from relational_atlas_i01.holdout import qiskit_rank_holdout2_grid as grid


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout3_power.json"


def study() -> dict:
    n, beta, spectrum, detector, labels = 3, 0.45, "nonlinear", "mismatched", 7
    output = []
    for physical in (False, True):
        entry = grid.model(n, beta, physical, spectrum)
        angles = grid.angles_for_grid(n, labels)
        for tau in (0.0, math.pi / 2):
            for distribution in ("uniform", "skewed"):
                weights = grid.label_weights(labels, distribution)
                baseline, _ = grid.probability_gradient(entry, detector, tau, angles)
                for shift in (0.23, 0.60, 1.00):
                    shifted, _ = grid.probability_gradient(entry, detector, tau,
                                                           angles + shift)
                    p = baseline @ weights
                    q = shifted @ weights
                    expected_chi_square_noncentrality = float(np.sum(
                        (q - p) ** 2 * (labels * 8192) /
                        (2 * p * (1 - p))))
                    output.append({
                        "physical_P_zero": physical, "tau_over_pi": tau / math.pi,
                        "label_distribution": distribution,
                        "common_shift": shift,
                        "max_erased_probability_difference": float(np.max(
                            np.abs(q - p))),
                        "approx_noncentrality_at_8192_shots_per_label":
                            expected_chi_square_noncentrality,
                    })
    return {"caveat": "Power pilot based only on exact protocol-2 model; no Aer shots observed.",
            "rows": output}


def main() -> None:
    result = study()
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
