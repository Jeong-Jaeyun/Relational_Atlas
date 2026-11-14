"""Local moving-string encounter baseline for I01's 'why three?' question.

Two affine worldsheet patches meet at equal time when
    A @ (t, s1, s2) = b
in d spatial dimensions.  A and b are isotropic Gaussian random variables;
the three parameters are restricted to [-1/2, 1/2].  This is deliberately a
local kinematic model, not a simulation of string winding, gravity, or a torus.
The exact Gaussian projection identity supplies a low-noise encounter
probability and direct Monte Carlo checks it independently.
"""

from __future__ import annotations

import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammainc


ROOT = PROJECT_ROOT
OUTPUT = ROOT / "results" / "encounter_threshold.json"
FIGURE = ROOT / "results" / "encounter_threshold.png"
SEED = 20261002
DELTAS = np.array([0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32], dtype=float)


def chi_ball_probability(codimension: int, radius: float, offset_sigma: float) -> float:
    if codimension == 0:
        return 1.0
    return float(gammainc(codimension / 2, radius**2 / (2 * offset_sigma**2)))


def sample_one_dimension(
    dimension: int,
    offset_sigma: float,
    trials: int,
    rng: np.random.Generator,
    chunk_size: int = 10000,
) -> dict:
    in_patch = 0
    direct_encounters = np.zeros(len(DELTAS), dtype=np.int64)
    residual_farthest = 0.0
    for start in range(0, trials, chunk_size):
        count = min(chunk_size, trials - start)
        a = rng.normal(size=(count, dimension, 3))
        b = rng.normal(scale=offset_sigma, size=(count, dimension))
        gram = np.matmul(a.transpose(0, 2, 1), a)
        rhs = np.einsum("ndk,nd->nk", a, b)
        parameters = np.linalg.solve(gram, rhs[..., None])[..., 0]
        residual = b - np.einsum("ndk,nk->nd", a, parameters)
        distances = np.linalg.norm(residual, axis=1)
        interior = np.max(np.abs(parameters), axis=1) <= 0.5
        in_patch += int(np.sum(interior))
        direct_encounters += np.sum(
            interior[:, None] & (distances[:, None] <= DELTAS[None, :]), axis=0
        )
        residual_farthest = max(residual_farthest, float(distances.max()))

    patch_probability = in_patch / trials
    codimension = dimension - 3
    rows = []
    for delta, direct_count in zip(DELTAS, direct_encounters):
        ball_probability = chi_ball_probability(codimension, float(delta), offset_sigma)
        rows.append({
            "encounter_radius": float(delta),
            "gaussian_orthogonal_ball_probability": ball_probability,
            "rao_blackwell_encounter_probability": patch_probability * ball_probability,
            "direct_mc_encounter_count": int(direct_count),
            "direct_mc_encounter_probability": float(direct_count / trials),
        })
    small_radius = np.array([row["rao_blackwell_encounter_probability"] for row in rows[:3]])
    slope = float(np.polyfit(np.log(DELTAS[:3]), np.log(small_radius), 1)[0])
    return {
        "dimension": dimension,
        "offset_sigma": offset_sigma,
        "trials": trials,
        "interior_count": in_patch,
        "interior_probability": patch_probability,
        "largest_observed_residual": residual_farthest,
        "codimension": codimension,
        "small_radius_log_slope": slope,
        "encounter_scan": rows,
    }


def staged_three_probability(p3: float, p4: float, attempt_count: float) -> float:
    """Conditional toy with Poisson-mean opportunities at each stage."""
    return float(-math.expm1(-attempt_count * p3) * math.exp(-attempt_count * p4))


def selection_window(rows_by_dimension: dict[str, dict]) -> list[dict]:
    dimension_3 = rows_by_dimension["3"]["encounter_scan"]
    dimension_4 = rows_by_dimension["4"]["encounter_scan"]
    out = []
    for three, four in zip(dimension_3, dimension_4):
        p3 = three["rao_blackwell_encounter_probability"]
        p4 = four["rao_blackwell_encounter_probability"]
        optimum_attempts = math.log1p(p3 / p4) / p3
        out.append({
            "encounter_radius": three["encounter_radius"],
            "stage_three_probability_per_attempt": p3,
            "stage_four_probability_per_attempt": p4,
            "rate_ratio_three_to_four": p3 / p4,
            "attempts_for_90_percent_stage_three_success": -math.log(0.1) / p3,
            "attempts_for_10_percent_stage_four_success": -math.log(0.9) / p4,
            "optimal_attempt_count": optimum_attempts,
            "maximum_conditional_three_stop_probability": staged_three_probability(
                p3, p4, optimum_attempts
            ),
        })
    return out


def make_figure(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    baseline = results["primary_scan"]
    for d in (3, 4, 5, 6):
        rows = baseline[str(d)]["encounter_scan"]
        axes[0].loglog(
            [r["encounter_radius"] for r in rows],
            [r["rao_blackwell_encounter_probability"] for r in rows],
            marker="o", label=f"d={d}, slope {baseline[str(d)]['small_radius_log_slope']:.2f}"
        )
    axes[0].set(xlabel="encounter radius / local scale", ylabel="per-patch encounter probability",
                title="Two moving 1D string patches")
    axes[0].legend(fontsize=8)
    attempts = np.logspace(-1, 4, 250)
    for row in results["conditional_stage_window"]:
        if row["encounter_radius"] not in (0.01, 0.04, 0.16):
            continue
        p3 = row["stage_three_probability_per_attempt"]
        p4 = row["stage_four_probability_per_attempt"]
        axes[1].semilogx(attempts, [staged_three_probability(p3, p4, m) for m in attempts],
                         label=f"radius={row['encounter_radius']}")
    axes[1].set(xlabel="mean independent opportunities M", ylabel="P(stage 3 succeeds, stage 4 fails)",
                title="Conditional toy selection window", ylim=(0, 1.02))
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(SEED)
    primary = {
        str(d): sample_one_dimension(d, 0.35, 150000, rng)
        for d in (3, 4, 5, 6)
    }
    sensitivity = {
        str(sigma): {
            str(d): sample_one_dimension(d, sigma, 60000, rng)
            for d in (3, 4, 5)
        }
        for sigma in (0.25, 0.5)
    }
    results = {
        "model": "Two independent affine string worldsheets; equal-time local spatial separation",
        "seed": SEED,
        "parameter_box": [-0.5, 0.5],
        "random_ensemble": "A_ij iid N(0,1); spatial offset b_i iid N(0,sigma^2)",
        "validity_boundary": "Local kinematic encounter only; no torus winding, interactions, expansion, gravity, or dimensional dynamics.",
        "projection_identity": "P(encounter)=P(z_hat in box)*CDF_chi_(d-3)(radius/sigma), for d>=3 and isotropic Gaussian b",
        "primary_scan": primary,
        "offset_sensitivity": sensitivity,
        "conditional_stage_window": selection_window(primary),
        "stage_model_warning": "Assumes Poisson-distributed independent opportunities with mean M at each stage, that contacts enable winding annihilation and growth, and that stage 3 is reached. None is derived from encounter geometry.",
    }
    assert abs(primary["3"]["small_radius_log_slope"]) < 1e-8
    for d in (4, 5, 6):
        assert abs(primary[str(d)]["small_radius_log_slope"] - (d - 3)) < 0.08
    OUTPUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    make_figure(results)
    print(json.dumps({
        "slopes": {d: row["small_radius_log_slope"] for d, row in primary.items()},
        "interior_probabilities": {d: row["interior_probability"] for d, row in primary.items()},
        "conditional_optima": [{"radius": row["encounter_radius"],
                                "max_three_stop_probability": row["maximum_conditional_three_stop_probability"]}
                               for row in results["conditional_stage_window"]],
        "results": str(OUTPUT),
        "figure": str(FIGURE),
    }, indent=2))


if __name__ == "__main__":
    main()
