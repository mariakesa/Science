#!/usr/bin/env python3
"""
Null controls for rank-ordered first-difference geometry.

Use after running:
    rank_ordered_semantic_axis_diff_geometry.py

This script loads the saved weight vectors and asks:

1. Is the raw animal-vs-scene decoder anti-alignment non-trivial?
   Null: shuffle scene weights relative to animal weights.

2. Is the rank-differenced texture alignment different from random/unrelated texture?
   Null A: shuffle scene weights, keep animal ordering fixed.
   Null B: use random neuron orderings for both vectors.

3. Is the near-90 degree first-difference result simply what an unrelated texture gives?
   The script reports observed |cosine| versus null |cosine| distributions.
   If observed |cosine| is near the null center, the diff texture behaves like
   unstructured/random relation. If observed |cosine| is far above null, the
   diff texture has reproducible cross-task alignment.

Important:
- These are WEIGHT-SHUFFLE controls, not label-permutation refits.
- They test geometry conditional on the fitted decoders.
- They do not prove noise; they tell whether the first-difference relation is
  stronger than random pairing/order baselines.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

EPS = 1e-12

DEFAULT_INDIR = Path("/home/maria/Science/results/rank_ordered_semantic_axis_diff_geometry")
DEFAULT_NPZ = DEFAULT_INDIR / "rank_ordered_first_difference_geometry_weights_and_diffs.npz"
DEFAULT_OUTDIR = DEFAULT_INDIR / "null_controls"

AREAS_DEFAULT = ["VISal", "VISam", "VISl", "VISp", "VISpm", "VISrl"]


def safe_cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) == 0:
        return np.nan
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= EPS or nb <= EPS:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def angle_from_cos(c: float) -> float:
    if not np.isfinite(c):
        return np.nan
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def axis_angle_from_cos(c: float) -> float:
    # Axis angle ignores sign: arccos(|cos|), range [0, 90].
    if not np.isfinite(c):
        return np.nan
    return float(np.degrees(np.arccos(np.clip(abs(c), 0.0, 1.0))))


def nan_quantiles(x, qs=(0.025, 0.05, 0.5, 0.95, 0.975)):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {f"q{int(q*1000):03d}": np.nan for q in qs}
    return {f"q{int(q*1000):03d}": float(np.quantile(x, q)) for q in qs}


def summarize_null(obs: float, null: np.ndarray, prefix: str) -> dict:
    null = np.asarray(null, dtype=np.float64)
    null = null[np.isfinite(null)]
    out = {
        f"{prefix}_obs": float(obs),
        f"{prefix}_null_mean": float(np.mean(null)) if len(null) else np.nan,
        f"{prefix}_null_std": float(np.std(null)) if len(null) else np.nan,
    }
    for k, v in nan_quantiles(null).items():
        out[f"{prefix}_null_{k}"] = v

    if len(null) and np.isfinite(obs):
        # two-sided alignment-strength p-value for absolute cosines:
        # how often is the random baseline at least as strongly aligned?
        out[f"{prefix}_p_abs_ge_obs"] = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1))
        # signed tail p-values, useful for raw negative cosines/spearman.
        out[f"{prefix}_p_le_obs"] = float((np.sum(null <= obs) + 1) / (len(null) + 1))
        out[f"{prefix}_p_ge_obs"] = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    else:
        out[f"{prefix}_p_abs_ge_obs"] = np.nan
        out[f"{prefix}_p_le_obs"] = np.nan
        out[f"{prefix}_p_ge_obs"] = np.nan
    return out


def metric_bundle(c: float, name: str) -> dict:
    return {
        f"{name}_cos": float(c),
        f"{name}_angle_deg": angle_from_cos(c),
        f"{name}_axis_angle_deg": axis_angle_from_cos(c),
        f"{name}_abs_cos": abs(float(c)) if np.isfinite(c) else np.nan,
    }


def get_key(npz, key):
    if key not in npz.files:
        raise KeyError(f"Missing key in NPZ: {key}\nAvailable example keys: {npz.files[:20]}")
    return np.asarray(npz[key])


def plot_null_hist(null_values, obs, title, xlabel, outpath):
    null_values = np.asarray(null_values, dtype=np.float64)
    null_values = null_values[np.isfinite(null_values)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_values, bins=45, alpha=0.75)
    ax.axvline(obs, linestyle="--", linewidth=2, label=f"observed = {obs:.4f}")
    ax.axvline(np.mean(null_values), linestyle=":", linewidth=2, label=f"null mean = {np.mean(null_values):.4f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def analyze_area(area: str, npz, rng: np.random.Generator, n_null: int, outdir: Path) -> dict:
    prefix = area
    wA = get_key(npz, f"{prefix}__w_animal").astype(np.float64)
    wS = get_key(npz, f"{prefix}__w_scene").astype(np.float64)

    if len(wA) != len(wS):
        raise ValueError(f"{area}: wA and wS length mismatch: {len(wA)} vs {len(wS)}")
    d = len(wA)

    order_A = np.argsort(wA)
    order_S = np.argsort(wS)

    # Observed raw metrics.
    raw_cos = safe_cosine(wA, wS)
    centered_cos = safe_cosine(wA - np.mean(wA), wS - np.mean(wS))
    sp = spearmanr(wA, wS).statistic

    # Observed first-difference metrics under decoder-induced ordering.
    dwA_by_A = np.diff(wA[order_A])
    dwS_by_A = np.diff(wS[order_A])
    dwS_by_S = np.diff(wS[order_S])
    dwA_by_S = np.diff(wA[order_S])

    obs_scene_texture_along_animal = safe_cosine(dwA_by_A, dwS_by_A)
    obs_animal_texture_along_scene = safe_cosine(dwS_by_S, dwA_by_S)

    # Useful sanity control: if scene were a perfect negative copy of animal,
    # diff textures would be perfectly anti-aligned, axis-angle 0.
    perfect_negative_scene = -wA
    perfect_neg_dw_cos = safe_cosine(np.diff(wA[order_A]), np.diff(perfect_negative_scene[order_A]))

    # Null arrays.
    null_raw_cos = np.empty(n_null, dtype=np.float64)
    null_centered_cos = np.empty(n_null, dtype=np.float64)
    null_spearman = np.empty(n_null, dtype=np.float64)

    null_scene_texture_shuffleS_along_A = np.empty(n_null, dtype=np.float64)
    null_animal_texture_shuffleA_along_S = np.empty(n_null, dtype=np.float64)

    null_scene_texture_random_order = np.empty(n_null, dtype=np.float64)
    null_animal_texture_random_order = np.empty(n_null, dtype=np.float64)

    for b in range(n_null):
        # Shuffle scene weights relative to animal weights.
        wS_perm = rng.permutation(wS)
        wA_perm = rng.permutation(wA)

        null_raw_cos[b] = safe_cosine(wA, wS_perm)
        null_centered_cos[b] = safe_cosine(wA - np.mean(wA), wS_perm - np.mean(wS_perm))
        null_spearman[b] = spearmanr(wA, wS_perm).statistic

        # Keep animal ordering fixed; ask if scene texture along animal rank is
        # more aligned than a randomly paired scene vector.
        null_scene_texture_shuffleS_along_A[b] = safe_cosine(
            np.diff(wA[order_A]),
            np.diff(wS_perm[order_A]),
        )

        # Symmetric: keep scene ordering fixed; shuffle animal weights.
        null_animal_texture_shuffleA_along_S[b] = safe_cosine(
            np.diff(wS[order_S]),
            np.diff(wA_perm[order_S]),
        )

        # Random common ordering: what happens to diff-cosines under an arbitrary
        # neuron queue applied to both real vectors?
        rand_order = rng.permutation(d)
        null_scene_texture_random_order[b] = safe_cosine(
            np.diff(wA[rand_order]),
            np.diff(wS[rand_order]),
        )

        rand_order2 = rng.permutation(d)
        null_animal_texture_random_order[b] = safe_cosine(
            np.diff(wS[rand_order2]),
            np.diff(wA[rand_order2]),
        )

    row = {
        "area": area,
        "n_features": d,
    }
    row.update(metric_bundle(raw_cos, "obs_raw_wA_wS"))
    row.update(metric_bundle(centered_cos, "obs_centered_wA_wS"))
    row["obs_spearman_wA_wS"] = float(sp)

    row.update(metric_bundle(obs_scene_texture_along_animal, "obs_scene_texture_along_animal_rankdiff"))
    row.update(metric_bundle(obs_animal_texture_along_scene, "obs_animal_texture_along_scene_rankdiff"))
    row.update(metric_bundle(perfect_neg_dw_cos, "sanity_perfect_negative_scene_rankdiff"))

    row.update(summarize_null(raw_cos, null_raw_cos, "null_shuffleS_raw_cos"))
    row.update(summarize_null(centered_cos, null_centered_cos, "null_shuffleS_centered_cos"))
    row.update(summarize_null(sp, null_spearman, "null_shuffleS_spearman"))
    row.update(summarize_null(obs_scene_texture_along_animal, null_scene_texture_shuffleS_along_A, "null_shuffleS_scene_texture_along_animal_rankdiff_cos"))
    row.update(summarize_null(obs_animal_texture_along_scene, null_animal_texture_shuffleA_along_S, "null_shuffleA_animal_texture_along_scene_rankdiff_cos"))
    row.update(summarize_null(obs_scene_texture_along_animal, null_scene_texture_random_order, "null_random_order_scene_texture_rankdiff_cos"))
    row.update(summarize_null(obs_animal_texture_along_scene, null_animal_texture_random_order, "null_random_order_animal_texture_rankdiff_cos"))

    # Plots: raw null and diff nulls.
    area_plot_dir = outdir / "plots"
    area_plot_dir.mkdir(parents=True, exist_ok=True)
    plot_null_hist(
        null_raw_cos,
        raw_cos,
        f"{area}: raw cosine null by shuffling scene weights",
        "cos(wA, shuffled wS)",
        area_plot_dir / f"{area}__raw_cos_shuffleS_null.png",
    )
    plot_null_hist(
        null_spearman,
        sp,
        f"{area}: Spearman null by shuffling scene weights",
        "Spearman(wA, shuffled wS)",
        area_plot_dir / f"{area}__spearman_shuffleS_null.png",
    )
    plot_null_hist(
        null_scene_texture_shuffleS_along_A,
        obs_scene_texture_along_animal,
        f"{area}: scene texture along animal order null",
        "cos(diff(wA[A-order]), diff(shuffled wS[A-order]))",
        area_plot_dir / f"{area}__scene_texture_along_animal_shuffleS_null.png",
    )
    plot_null_hist(
        null_animal_texture_shuffleA_along_S,
        obs_animal_texture_along_scene,
        f"{area}: animal texture along scene order null",
        "cos(diff(wS[S-order]), diff(shuffled wA[S-order]))",
        area_plot_dir / f"{area}__animal_texture_along_scene_shuffleA_null.png",
    )

    # Save null arrays for this area.
    np.savez_compressed(
        outdir / f"{area}__null_arrays.npz",
        null_raw_cos=null_raw_cos,
        null_centered_cos=null_centered_cos,
        null_spearman=null_spearman,
        null_scene_texture_shuffleS_along_A=null_scene_texture_shuffleS_along_A,
        null_animal_texture_shuffleA_along_S=null_animal_texture_shuffleA_along_S,
        null_scene_texture_random_order=null_scene_texture_random_order,
        null_animal_texture_random_order=null_animal_texture_random_order,
        obs_raw_cos=np.array([raw_cos]),
        obs_centered_cos=np.array([centered_cos]),
        obs_spearman=np.array([sp]),
        obs_scene_texture_along_animal=np.array([obs_scene_texture_along_animal]),
        obs_animal_texture_along_scene=np.array([obs_animal_texture_along_scene]),
    )

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--areas", nargs="*", default=AREAS_DEFAULT)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 88)
    print("Null controls for rank-ordered first-difference geometry")
    print("=" * 88)
    print(f"Input NPZ: {args.npz}")
    print(f"Output dir: {args.outdir}")
    print(f"n_null: {args.n_null}")
    print(f"seed: {args.seed}")

    npz = np.load(args.npz, allow_pickle=True)
    rows = []
    for area in args.areas:
        print("\n" + "#" * 88)
        print(f"Area: {area}")
        print("#" * 88)
        row = analyze_area(area, npz, rng, args.n_null, args.outdir)
        rows.append(row)

        print(
            f"{area:>6s} | raw cos={row['obs_raw_wA_wS_cos']:+.4f} "
            f"p_abs={row['null_shuffleS_raw_cos_p_abs_ge_obs']:.4g} | "
            f"spearman={row['obs_spearman_wA_wS']:+.4f} "
            f"p_abs={row['null_shuffleS_spearman_p_abs_ge_obs']:.4g} | "
            f"diff scene~animal cos={row['obs_scene_texture_along_animal_rankdiff_cos']:+.4f} "
            f"null mean={row['null_shuffleS_scene_texture_along_animal_rankdiff_cos_null_mean']:+.4f} "
            f"p_abs={row['null_shuffleS_scene_texture_along_animal_rankdiff_cos_p_abs_ge_obs']:.4g}"
        )

    df = pd.DataFrame(rows)
    out_csv = args.outdir / "rank_diff_null_control_summary.csv"
    df.to_csv(out_csv, index=False)

    compact_cols = [
        "area", "n_features",
        "obs_raw_wA_wS_cos", "obs_raw_wA_wS_axis_angle_deg", "null_shuffleS_raw_cos_p_abs_ge_obs",
        "obs_centered_wA_wS_cos", "obs_centered_wA_wS_axis_angle_deg", "null_shuffleS_centered_cos_p_abs_ge_obs",
        "obs_spearman_wA_wS", "null_shuffleS_spearman_p_abs_ge_obs",
        "obs_scene_texture_along_animal_rankdiff_cos", "obs_scene_texture_along_animal_rankdiff_axis_angle_deg",
        "null_shuffleS_scene_texture_along_animal_rankdiff_cos_null_mean", "null_shuffleS_scene_texture_along_animal_rankdiff_cos_null_std",
        "null_shuffleS_scene_texture_along_animal_rankdiff_cos_p_abs_ge_obs",
        "obs_animal_texture_along_scene_rankdiff_cos", "obs_animal_texture_along_scene_rankdiff_axis_angle_deg",
        "null_shuffleA_animal_texture_along_scene_rankdiff_cos_null_mean", "null_shuffleA_animal_texture_along_scene_rankdiff_cos_null_std",
        "null_shuffleA_animal_texture_along_scene_rankdiff_cos_p_abs_ge_obs",
        "sanity_perfect_negative_scene_rankdiff_cos", "sanity_perfect_negative_scene_rankdiff_axis_angle_deg",
    ]
    compact_cols = [c for c in compact_cols if c in df.columns]
    compact_csv = args.outdir / "rank_diff_null_control_compact_summary.csv"
    df[compact_cols].to_csv(compact_csv, index=False)

    with open(args.outdir / "config.json", "w") as f:
        json.dump({
            "input_npz": str(args.npz),
            "outdir": str(args.outdir),
            "n_null": args.n_null,
            "seed": args.seed,
            "areas": args.areas,
            "interpretation": {
                "raw_shuffle_null": "Tests whether global animal-scene weight alignment is stronger than random feature pairing.",
                "rankdiff_shuffle_null": "Tests whether first-difference texture alignment is stronger than random pairing of the other task weights.",
                "p_abs_ge_obs": "Fraction of null absolute cosines at least as large as observed absolute cosine; small means unusually strong alignment/anti-alignment.",
                "near_null_diff_result": "If observed rankdiff cosine is near null mean and p_abs_ge_obs is large, the diff texture behaves like unrelated/random texture.",
            },
        }, f, indent=2)

    print("\n" + "=" * 88)
    print("COMPACT SUMMARY")
    print("=" * 88)
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(df[compact_cols].to_string(index=False))
    print("\n[DONE]")
    print(f"Full summary:    {out_csv}")
    print(f"Compact summary: {compact_csv}")
    print(f"Plots:           {args.outdir / 'plots'}")
    print(f"Null arrays:     {args.outdir}/*__null_arrays.npz")


if __name__ == "__main__":
    main()
