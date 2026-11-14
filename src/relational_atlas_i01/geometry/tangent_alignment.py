"""Stress test of the three-dimensional encounter threshold.

The generic-worldsheet result assumes two independent string tangents.
Exactly parallel straight winding strings have only two independent meeting
parameters (time and relative position along the common tangent).  This code
compares exact alignment to finite transverse tangent fluctuations.
"""

from __future__ import annotations

import itertools
import json
from relational_atlas_i01.paths import PROJECT_ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from relational_atlas_i01.geometry.encounter_threshold import chi_ball_probability


ROOT = PROJECT_ROOT
OUTPUT = ROOT / "results" / "tangent_alignment.json"
FIGURE = ROOT / "results" / "tangent_alignment.png"
SEED = 20261003
SIGMA = 0.35
RADII = np.array([0.005, 0.01, 0.02, 0.04, 0.08], dtype=float)
ETAS = (0.0, 0.01, 0.1, 1.0)


def bounded_min_distances(a: np.ndarray, offset: np.ndarray,
                          lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Exact least-squares distance to a 2- or 3-parameter affine box.

    An optimum has each coordinate free, on its lower face, or on its upper
    face.  Enumerating those 3**p active sets is exact for this convex box.
    """
    count, _, parameters = a.shape
    best_squared = np.full(count, np.inf)
    for status in itertools.product((-1, 0, 1), repeat=parameters):
        fixed = [j for j, value in enumerate(status) if value]
        free = [j for j, value in enumerate(status) if not value]
        target = offset.copy()
        if fixed:
            fixed_values = np.array([lower[j] if status[j] == -1 else upper[j]
                                     for j in fixed])
            target -= np.einsum("ndk,k->nd", a[:, :, fixed], fixed_values)
        if free:
            a_free = a[:, :, free]
            gram = np.matmul(a_free.transpose(0, 2, 1), a_free)
            rhs = np.einsum("ndk,nd->nk", a_free, target)
            solution = np.linalg.solve(gram, rhs[..., None])[..., 0]
            feasible = np.all((solution >= lower[free] - 1e-10) &
                              (solution <= upper[free] + 1e-10), axis=1)
            target -= np.einsum("ndk,nk->nd", a_free, solution)
        else:
            feasible = np.ones(count, dtype=bool)
        squared = np.einsum("nd,nd->n", target, target)
        best_squared = np.minimum(best_squared, np.where(feasible, squared, np.inf))
    return np.sqrt(best_squared)


def sample_configuration(dimension: int, eta: float, trials: int,
                         rng: np.random.Generator, chunk_size: int = 10000) -> dict:
    aligned = eta == 0.0
    parameter_dimension = 2 if aligned else 3
    interior_count = 0
    interior_direct_counts = np.zeros(len(RADII), dtype=np.int64)
    bounded_direct_counts = np.zeros(len(RADII), dtype=np.int64)
    for start in range(0, trials, chunk_size):
        count = min(chunk_size, trials - start)
        tangent = rng.normal(size=(count, dimension))
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        velocity = rng.normal(size=(count, dimension))
        offset = rng.normal(scale=SIGMA, size=(count, dimension))
        if aligned:
            # ell=s_A-s_B lies in [-1,1] when both s parameters lie in [-1/2,1/2].
            a = np.stack((velocity, tangent), axis=2)
        else:
            second_tangent = tangent + eta * rng.normal(size=(count, dimension))
            a = np.stack((velocity, tangent, -second_tangent), axis=2)
        gram = np.matmul(a.transpose(0, 2, 1), a)
        rhs = np.einsum("ndk,nd->nk", a, offset)
        parameters = np.linalg.solve(gram, rhs[..., None])[..., 0]
        residual = offset - np.einsum("ndk,nk->nd", a, parameters)
        distance = np.linalg.norm(residual, axis=1)
        if aligned:
            interior = (np.abs(parameters[:, 0]) <= 0.5) & (np.abs(parameters[:, 1]) <= 1.0)
            lower = np.array([-0.5, -1.0])
            upper = np.array([0.5, 1.0])
        else:
            interior = np.max(np.abs(parameters), axis=1) <= 0.5
            lower = np.full(3, -0.5)
            upper = np.full(3, 0.5)
        interior_count += int(interior.sum())
        interior_direct_counts += np.sum(interior[:, None] &
                                         (distance[:, None] <= RADII[None, :]), axis=0)
        bounded_distance = bounded_min_distances(a, offset, lower, upper)
        bounded_direct_counts += np.sum(bounded_distance[:, None] <= RADII[None, :], axis=0)

    interior_probability = interior_count / trials
    codimension = dimension - parameter_dimension
    scan = []
    for radius, interior_direct, bounded_direct in zip(
        RADII, interior_direct_counts, bounded_direct_counts
    ):
        scan.append({
            "radius": float(radius),
            "interior_projection_probability": interior_probability *
                chi_ball_probability(codimension, float(radius), SIGMA),
            "interior_direct_mc_count": int(interior_direct),
            "interior_direct_mc_probability": float(interior_direct / trials),
            "bounded_patch_mc_count": int(bounded_direct),
            "bounded_patch_mc_probability": float(bounded_direct / trials),
        })
    probabilities = np.array([row["interior_projection_probability"] for row in scan[:3]])
    bounded_probabilities = np.array([row["bounded_patch_mc_probability"] for row in scan[:3]])
    return {
        "spatial_dimension": dimension,
        "transverse_tangent_fluctuation_eta": eta,
        "number_of_independent_meeting_parameters": parameter_dimension,
        "codimension": codimension,
        "trials": trials,
        "interior_count": interior_count,
        "interior_probability": interior_probability,
        "interior_asymptotic_small_radius_slope": float(np.polyfit(np.log(RADII[:3]), np.log(probabilities), 1)[0]),
        "bounded_patch_finite_radius_slope": float(np.polyfit(np.log(RADII[:3]), np.log(bounded_probabilities), 1)[0]),
        "radius_scan": scan,
    }


def make_figure(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for d, axis in ((3, axes[0]), (4, axes[1])):
        for eta in ETAS:
            row = data["by_dimension"][str(d)][str(eta)]
            axis.loglog(RADII, [x["bounded_patch_mc_probability"]
                                for x in row["radius_scan"]], marker="o",
                        label=f"eta={eta}, finite slope={row['bounded_patch_finite_radius_slope']:.1f}")
        axis.set(xlabel="encounter radius", ylabel="interior encounter probability",
                 title=f"d={d}: straight vs fluctuating tangents")
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(SEED)
    data = {
        "model": "Two local moving line patches with exactly aligned or independently fluctuating tangents",
        "seed": SEED,
        "offset_sigma": SIGMA,
        "eta_values": ETAS,
        "warning": "The bounded-patch minimum includes box faces but omits torus windings. Eta is an assigned tangent perturbation, not a derived physical oscillation spectrum. Interior-projection slopes describe the true small-radius limit only at fixed nonzero eta; limits eta->0 and radius->0 need not commute.",
        "by_dimension": {
            str(d): {str(eta): sample_configuration(
                d, eta, 300000 if d == 4 and eta <= 0.01 else 80000, rng
            )
                     for eta in ETAS}
            for d in (3, 4)
        },
    }
    assert abs(data["by_dimension"]["3"]["0.0"]["interior_asymptotic_small_radius_slope"] - 1) < 0.01
    assert abs(data["by_dimension"]["4"]["0.0"]["interior_asymptotic_small_radius_slope"] - 2) < 0.01
    assert abs(data["by_dimension"]["3"]["1.0"]["interior_asymptotic_small_radius_slope"]) < 0.01
    assert abs(data["by_dimension"]["4"]["1.0"]["interior_asymptotic_small_radius_slope"] - 1) < 0.01
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figure(data)
    print(json.dumps({
        "d3": {eta: {"interior": row["interior_probability"],
                     "interior_slope": row["interior_asymptotic_small_radius_slope"],
                     "bounded_finite_slope": row["bounded_patch_finite_radius_slope"]}
               for eta, row in data["by_dimension"]["3"].items()},
        "d4": {eta: {"interior": row["interior_probability"],
                     "interior_slope": row["interior_asymptotic_small_radius_slope"],
                     "bounded_finite_slope": row["bounded_patch_finite_radius_slope"]}
               for eta, row in data["by_dimension"]["4"].items()},
        "results": str(OUTPUT),
        "figure": str(FIGURE),
    }, indent=2))


if __name__ == "__main__":
    main()
