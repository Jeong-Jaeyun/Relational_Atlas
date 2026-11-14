"""Opposite-winding string-pair encounters on a periodic spatial torus.

This is a deliberately minimal next step beyond the local tangent model:
each pair has winding numbers +1 and -1 around one coordinate of T^d,
rigid transverse drift during one observation interval, and a periodic
two-harmonic transverse shape.  A contact is found by minimizing exact
piecewise-linear distance over time, winding coordinate, and periodic images.
Contact only makes annihilation *eligible*; no radius equation or microscopic
string interaction is claimed.
"""

from __future__ import annotations

import itertools
import json
import math
from relational_atlas_i01.paths import PROJECT_ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = PROJECT_ROOT
OUTPUT = ROOT / "results" / "torus_winding.json"
FIGURE = ROOT / "results" / "torus_winding.png"
SEED = 20261004
RADII = np.array([0.01, 0.02, 0.04, 0.08], dtype=float)
AMPLITUDES = (0.0, 0.05, 0.15)
HARMONICS = 2


def make_geometry(dimension: int, trials: int, rng: np.random.Generator,
                  drift_max: float = 0.25) -> dict:
    transverse = dimension - 1
    return {
        "offset": rng.uniform(-0.5, 0.5, size=(trials, transverse)),
        "drift": rng.uniform(-drift_max, drift_max, size=(trials, transverse)),
        "sin_coefficients": rng.normal(size=(trials, HARMONICS, transverse)),
        "cos_coefficients": rng.normal(size=(trials, HARMONICS, transverse)),
    }


def raw_fourier_profile(geometry: dict, positions: np.ndarray) -> np.ndarray:
    sin_coefficients = geometry["sin_coefficients"]
    cos_coefficients = geometry["cos_coefficients"]
    trials, harmonics, transverse = sin_coefficients.shape
    profile = np.zeros((trials, len(positions), transverse), dtype=float)
    for harmonic in range(1, harmonics + 1):
        phase = 2 * math.pi * harmonic * positions
        profile += (sin_coefficients[:, harmonic - 1, None, :] *
                    np.sin(phase)[None, :, None] +
                    cos_coefficients[:, harmonic - 1, None, :] *
                    np.cos(phase)[None, :, None]) / harmonic
    return profile


def profile_nodes(geometry: dict, amplitude: float, segments: int) -> np.ndarray:
    trials, transverse = geometry["offset"].shape
    if amplitude == 0.0:
        return np.zeros((trials, 2, transverse))
    dense = raw_fourier_profile(geometry, np.linspace(0, 1, 129))
    maximum = np.max(np.abs(dense), axis=(1, 2), keepdims=True)
    sampled = raw_fourier_profile(geometry, np.linspace(0, 1, segments + 1))
    return amplitude * sampled / np.maximum(maximum, 1e-15)


def periodic_images(transverse: int) -> np.ndarray:
    return np.asarray(list(itertools.product((-1.0, 0.0, 1.0), repeat=transverse)))


def affine_patch_squared_distances(offset: np.ndarray, drift: np.ndarray,
                                   tangent: np.ndarray, images: np.ndarray) -> np.ndarray:
    """Exact minimum of ||offset+drift*t+tangent*s-image||^2 for t,s in [0,1]."""
    target = images[None, :, :] - offset[:, None, :]
    bb = np.einsum("nid,nid->ni", target, target)
    vv = np.einsum("nd,nd->n", drift, drift)[:, None]
    ww = np.einsum("nd,nd->n", tangent, tangent)[:, None]
    vw = np.einsum("nd,nd->n", drift, tangent)[:, None]
    vb = np.einsum("nd,nid->ni", drift, target)
    wb = np.einsum("nd,nid->ni", tangent, target)
    determinant = vv * ww - vw * vw
    good = determinant > 1e-13
    time = np.divide(vb * ww - wb * vw, determinant,
                     out=np.zeros_like(vb), where=good)
    string_position = np.divide(wb * vv - vb * vw, determinant,
                                out=np.zeros_like(wb), where=good)
    interior = (good & (time >= 0) & (time <= 1) &
                (string_position >= 0) & (string_position <= 1))
    best = np.where(interior, np.maximum(0.0, bb - time * vb - string_position * wb), np.inf)

    for fixed_time in (0.0, 1.0):
        projected = wb - fixed_time * vw
        position = np.clip(np.divide(projected, ww, out=np.zeros_like(wb), where=ww > 1e-15), 0, 1)
        candidate = bb - 2 * fixed_time * vb + fixed_time**2 * vv - 2 * position * projected + position**2 * ww
        best = np.minimum(best, np.maximum(0.0, candidate))
    for fixed_position in (0.0, 1.0):
        projected = vb - fixed_position * vw
        time = np.clip(np.divide(projected, vv, out=np.zeros_like(vb), where=vv > 1e-15), 0, 1)
        candidate = bb - 2 * fixed_position * wb + fixed_position**2 * ww - 2 * time * projected + time**2 * vv
        best = np.minimum(best, np.maximum(0.0, candidate))
    return np.min(best, axis=1)


def pair_minimum_distances(geometry: dict, amplitude: float, segments: int = 12,
                           chunk_size: int = 2000) -> np.ndarray:
    offset_all = geometry["offset"]
    trials, transverse = offset_all.shape
    images = periodic_images(transverse)
    distances = np.empty(trials)
    for start in range(0, trials, chunk_size):
        end = min(start + chunk_size, trials)
        subset = {key: value[start:end] for key, value in geometry.items()}
        nodes = profile_nodes(subset, amplitude, segments)
        shortest = np.full(end - start, np.inf)
        for segment in range(nodes.shape[1] - 1):
            offset = subset["offset"] + nodes[:, segment]
            tangent = nodes[:, segment + 1] - nodes[:, segment]
            candidate = affine_patch_squared_distances(offset, subset["drift"], tangent, images)
            shortest = np.minimum(shortest, candidate)
        distances[start:end] = np.sqrt(shortest)
    return distances


def contact_scan(distances: np.ndarray) -> list[dict]:
    trials = len(distances)
    rows = []
    for radius in RADII:
        count = int(np.sum(distances <= radius))
        probability = count / trials
        z = 1.96
        denominator = 1 + z**2 / trials
        center = (probability + z**2 / (2 * trials)) / denominator
        half_width = z * math.sqrt(probability * (1 - probability) / trials +
                                   z**2 / (4 * trials**2)) / denominator
        rows.append({
            "contact_radius": float(radius),
            "contact_count": count,
            "contact_probability": probability,
            "binomial_standard_error": math.sqrt(probability * (1 - probability) / trials),
            "wilson_95_percent_interval": [center - half_width, center + half_width],
        })
    return rows


def pair_clearance(contact_probability: float, mean_opportunities: float,
                   annihilation_given_contact: float = 1.0) -> float:
    return -math.expm1(-mean_opportunities * contact_probability * annihilation_given_contact)


def stage_clearance(contact_probability: float, dimension: int, pairs_per_axis: int,
                    mean_opportunities: float,
                    annihilation_given_contact: float = 1.0) -> float:
    return pair_clearance(contact_probability, mean_opportunities,
                          annihilation_given_contact) ** (dimension * pairs_per_axis)


def conditional_three_stop(p3: float, p4: float, mean_opportunities: float,
                           pairs_per_axis: int = 1,
                           annihilation_given_contact: float = 1.0) -> float:
    return (stage_clearance(p3, 3, pairs_per_axis, mean_opportunities,
                            annihilation_given_contact) *
            (1 - stage_clearance(p4, 4, pairs_per_axis, mean_opportunities,
                                 annihilation_given_contact)))


def make_figure(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for amplitude in AMPLITUDES:
        for dimension, linestyle in ((3, "-"), (4, "--")):
            rows = results["contact_results"][str(dimension)][str(amplitude)]["contact_scan"]
            axes[0].loglog([row["contact_radius"] for row in rows],
                           [row["contact_probability"] for row in rows],
                           marker="o", linestyle=linestyle,
                           label=f"d={dimension}, wiggle={amplitude}")
    axes[0].set(xlabel="contact radius on unit torus", ylabel="pair contact probability",
                title="Opposite winding-pair encounter")
    axes[0].legend(fontsize=7)
    stage_rows = results["conditional_stage_scan"]
    for amplitude in (0.0, 0.05, 0.15):
        row = next(r for r in stage_rows if r["wiggle_amplitude"] == amplitude
                   and r["contact_radius"] == 0.02 and r["pairs_per_axis"] == 1)
        axes[1].semilogx(row["mean_opportunities"], row["conditional_three_stop_probability"],
                         label=f"wiggle={amplitude}")
    axes[1].axvline(30, color="gray", linestyle=":", linewidth=1.5, label="M=30 benchmark")
    axes[1].set(xlabel="mean Poisson opportunities per pair", ylabel="conditional P(stop at 3)",
                title="Assumed annihilation/expansion gate", ylim=(0, 1.02))
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(SEED)
    contact_results = {}
    for dimension in (3, 4, 5):
        trials = {3: 20000, 4: 50000, 5: 20000}[dimension]
        geometry = make_geometry(dimension, trials, rng)
        rows = {}
        for amplitude in AMPLITUDES:
            distances = pair_minimum_distances(geometry, amplitude)
            rows[str(amplitude)] = {
                "trials": trials,
                "transverse_dimension": dimension - 1,
                "wiggle_amplitude": amplitude,
                "profile_segments": 1 if amplitude == 0 else 12,
                "zero_thickness_intersection_tolerance": 1e-7,
                "zero_thickness_intersection_count": int(np.sum(distances <= 1e-7)),
                "zero_thickness_intersection_probability": float(np.mean(distances <= 1e-7)),
                "distance_quantiles": np.quantile(distances, [0.01, 0.1, 0.5, 0.9]).tolist(),
                "contact_scan": contact_scan(distances),
            }
        contact_results[str(dimension)] = rows

    resolution = []
    for dimension in (3, 4):
        geometry = make_geometry(dimension, 5000, np.random.default_rng(SEED + dimension))
        for segments in (6, 12, 24):
            distances = pair_minimum_distances(geometry, 0.15, segments)
            resolution.append({"dimension": dimension, "segments": segments,
                               "contact_scan": contact_scan(distances)})

    drift_sensitivity = []
    for drift_max in (0.0, 0.1, 0.25, 0.5):
        for dimension in (3, 4):
            geometry = make_geometry(dimension, 5000,
                                     np.random.default_rng(SEED + 1000 +
                                                           int(100 * drift_max) + dimension),
                                     drift_max=drift_max)
            distances = pair_minimum_distances(geometry, 0.15)
            drift_sensitivity.append({
                "dimension": dimension,
                "maximum_drift_component": drift_max,
                "wiggle_amplitude": 0.15,
                "trials": len(distances),
                "zero_thickness_intersection_count": int(np.sum(distances <= 1e-7)),
                "contact_scan": contact_scan(distances),
            })

    mean_opportunities = np.logspace(-1, 3, 120)
    conditional_stage_scan = []
    for amplitude in AMPLITUDES:
        for radius in (0.02, 0.04):
            p3 = next(row["contact_probability"] for row in
                      contact_results["3"][str(amplitude)]["contact_scan"]
                      if row["contact_radius"] == radius)
            p4 = next(row["contact_probability"] for row in
                      contact_results["4"][str(amplitude)]["contact_scan"]
                      if row["contact_radius"] == radius)
            for pairs_per_axis in (1, 2):
                probabilities = [conditional_three_stop(p3, p4, float(value), pairs_per_axis)
                                 for value in mean_opportunities]
                conditional_stage_scan.append({
                    "wiggle_amplitude": amplitude,
                    "contact_radius": radius,
                    "pairs_per_axis": pairs_per_axis,
                    "contact_probability_d3": p3,
                    "contact_probability_d4": p4,
                    "mean_opportunities": mean_opportunities.tolist(),
                    "conditional_three_stop_probability": probabilities,
                    "maximum_conditional_three_stop_probability": max(probabilities),
                    "mean_opportunities_at_maximum": float(mean_opportunities[np.argmax(probabilities)]),
                })

    annihilation_efficiency_scan = []
    for amplitude in (0.0, 0.15):
        p3 = next(r["contact_probability"] for r in
                  contact_results["3"][str(amplitude)]["contact_scan"]
                  if r["contact_radius"] == 0.02)
        p4 = next(r["contact_probability"] for r in
                  contact_results["4"][str(amplitude)]["contact_scan"]
                  if r["contact_radius"] == 0.02)
        for efficiency in (0.1, 0.3, 0.5, 1.0):
            for pairs_per_axis in (1, 2):
                annihilation_efficiency_scan.append({
                    "wiggle_amplitude": amplitude,
                    "annihilation_given_contact": efficiency,
                    "pairs_per_axis": pairs_per_axis,
                    "contact_radius": 0.02,
                    "conditional_three_stop_M30": conditional_three_stop(
                        p3, p4, 30.0, pairs_per_axis, efficiency
                    ),
                    "conditional_three_stop_M100": conditional_three_stop(
                        p3, p4, 100.0, pairs_per_axis, efficiency
                    ),
                })

    results = {
        "model": "Opposite +/-1 winding pair on T^d with transverse drift and periodic two-harmonic shape",
        "pair_winding_numbers": [1, -1],
        "annihilation_rule": "After a contact, remove the +1 and -1 strings together; total winding remains zero.",
        "seed": SEED,
        "torus_side_length": 1.0,
        "drift_component_range": [-0.25, 0.25],
        "offset_component_range": [-0.5, 0.5],
        "periodic_image_range_per_transverse_axis": [-1, 0, 1],
        "image_range_justification": "|offset+drift+profile|<1.15 even in the drift sensitivity scan, so nearest integer lies in {-1,0,1}.",
        "contact_results": contact_results,
        "resolution_check": resolution,
        "drift_sensitivity": drift_sensitivity,
        "conditional_stage_scan": conditional_stage_scan,
        "annihilation_efficiency_scan": annihilation_efficiency_scan,
        "warning": "Contact geometry is computed, but annihilation-on-contact, independent pairs/opportunities, and radius growth after clearance are separately assumed. No Nambu-Goto dynamics, gravity, backreaction, or cosmological probability.",
    }
    assert contact_results["3"]["0.15"]["zero_thickness_intersection_probability"] > 0.04
    assert contact_results["3"]["0.0"]["zero_thickness_intersection_count"] == 0
    assert contact_results["4"]["0.15"]["zero_thickness_intersection_count"] == 0
    OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figure(results)
    print(json.dumps({
        "contacts_at_radius_0p02": {
            str(d): {str(a): next(r["contact_probability"] for r in
                                contact_results[str(d)][str(a)]["contact_scan"]
                                if r["contact_radius"] == 0.02)
                     for a in AMPLITUDES}
            for d in (3, 4, 5)
        },
        "zero_thickness_intersections": {
            str(d): {str(a): contact_results[str(d)][str(a)]["zero_thickness_intersection_probability"]
                     for a in AMPLITUDES}
            for d in (3, 4, 5)
        },
        "stage_maxima_radius_0p02_one_pair": {
            str(r["wiggle_amplitude"]): r["maximum_conditional_three_stop_probability"]
            for r in conditional_stage_scan
            if r["contact_radius"] == 0.02 and r["pairs_per_axis"] == 1
        },
        "stage_at_30_opportunities_radius_0p02_one_pair": {
            str(a): conditional_three_stop(
                next(r["contact_probability"] for r in contact_results["3"][str(a)]["contact_scan"]
                     if r["contact_radius"] == 0.02),
                next(r["contact_probability"] for r in contact_results["4"][str(a)]["contact_scan"]
                     if r["contact_radius"] == 0.02),
                30.0,
            ) for a in AMPLITUDES
        },
        "results": str(OUTPUT),
        "figure": str(FIGURE),
    }, indent=2))


if __name__ == "__main__":
    main()
