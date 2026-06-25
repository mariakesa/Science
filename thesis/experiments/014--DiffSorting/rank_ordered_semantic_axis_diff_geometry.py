#!/usr/bin/env python3
"""
Rank-ordered first-difference geometry of semantic decoder axes.

This script compares animal-vs-not-animal and scene-vs-not-scene logistic
axes inside the SAME neural population for each brain area.

Core idea
---------
For each area, fit two full-data Adam logistic decoders on a shared standardized
matrix:

    w_A = animal-vs-not-animal logistic weight vector
    w_S = scene-vs-not-scene logistic weight vector

Then induce orderings over neurons/features:

    order_A_signed = argsort(w_A)       # anti-animal -> animal
    order_S_signed = argsort(w_S)       # anti-scene  -> scene

and compare first differences under these orderings:

    dwA_by_A = diff(w_A[order_A_signed])
    dwS_by_A = diff(w_S[order_A_signed])

    dwS_by_S = diff(w_S[order_S_signed])
    dwA_by_S = diff(w_A[order_S_signed])

Interpretation
--------------
1. cos(dwA_by_A, dwS_by_A):
   Along the animal-ranked neuron hierarchy, does scene weight texture vary
   like animal weight texture?

2. cos(dwS_by_S, dwA_by_S):
   Along the scene-ranked neuron hierarchy, does animal weight texture vary
   like scene weight texture?

3. cos(dwA_by_A, dwA_by_S):
   Does the animal axis have similar first-difference texture under animal
   ordering vs scene ordering?

4. cos(dwS_by_S, dwS_by_A):
   Does the scene axis have similar first-difference texture under scene
   ordering vs animal ordering?

The script also computes Spearman correlation between w_A and w_S, plus
ordinary vector angles between w_A and w_S.

Important note
--------------
A first difference after sorting is NOT a spatial derivative. It is a
rank-ordered/order-statistic derivative. It measures decoder-weight texture
under an ordering induced by another decoder.
"""

from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from scipy.stats import spearmanr, pearsonr


# =============================================================================
# Paths
# =============================================================================

DATA_DIR = Path("/home/maria/Science/data")

BRAIN_AREA_PATH = DATA_DIR / "brain_area.npy"
NEURAL_PATH = DATA_DIR / "hybrid_neural_responses_reduced.npy"

LABEL_FILES = {
    "animals_vs_everything": DATA_DIR / "image_labels.npy",
    "scenes_vs_everything": DATA_DIR / "scenes_vs_Everything_labels.npy",
}

OUTDIR = Path("/home/maria/Science/results/rank_ordered_semantic_axis_diff_geometry")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY_CSV = OUTDIR / "rank_ordered_first_difference_geometry_summary.csv"
OUT_WEIGHTS_NPZ = OUTDIR / "rank_ordered_first_difference_geometry_weights_and_diffs.npz"
OUT_CONFIG_JSON = OUTDIR / "config.json"

PLOT_DIR = OUTDIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Model settings
# =============================================================================

LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 3000
RANDOM_SEED = 0
EPS = 1e-12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Basic utilities
# =============================================================================

def sigmoid_np(z: np.ndarray | float) -> np.ndarray | float:
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def load_labels(path: Path) -> np.ndarray:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == () and isinstance(obj.item(), dict):
        d = obj.item()
        if "labels" not in d:
            raise ValueError(f"Label dict at {path} does not contain key 'labels'.")
        labels = d["labels"]
    else:
        labels = obj

    labels = np.asarray(labels).astype(np.int64).ravel()
    bad = set(np.unique(labels).tolist()) - {-1, 0, 1}
    if bad:
        raise ValueError(f"Unexpected label values in {path}: {bad}. Expected only -1, 0, 1.")
    return labels


def load_brain_areas(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, allow_pickle=True)).astype(str).ravel()


def load_neural_matrix(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, allow_pickle=True), dtype=np.float64)


def align_x_to_labels(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return X as images x neurons/features."""
    if X.shape[0] == len(y):
        return X
    if X.shape[1] == len(y):
        return X.T
    raise ValueError(
        f"Cannot align X with labels. X={X.shape}, labels={y.shape}. "
        "Expected one X axis to match number of labels."
    )


def clean_features_global(X: np.ndarray, brain_area: np.ndarray):
    if X.shape[1] != len(brain_area):
        raise ValueError(
            f"Feature axis of X does not match brain_area length. "
            f"X features={X.shape[1]}, brain_area={len(brain_area)}."
        )

    finite_cols = np.all(np.isfinite(X), axis=0)
    good_cols = np.zeros(X.shape[1], dtype=bool)
    finite_indices = np.where(finite_cols)[0]

    std_on_finite = np.std(X[:, finite_cols], axis=0)
    nonconstant_finite = std_on_finite > EPS
    good_cols[finite_indices[nonconstant_finite]] = True

    return X[:, good_cols], brain_area[good_cols], good_cols


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= EPS or nb <= EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def angle_from_cos(c: float) -> float:
    if not np.isfinite(c):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def axis_angle_from_angle(angle: float) -> float:
    if not np.isfinite(angle):
        return float("nan")
    return float(min(angle, 180.0 - angle))


def cosine_angle_axis(a: np.ndarray, b: np.ndarray):
    c = safe_cosine(a, b)
    ang = angle_from_cos(c)
    ax = axis_angle_from_angle(ang)
    return c, ang, ax


def safe_corrs(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 3:
        return np.nan, np.nan, np.nan, np.nan
    if np.std(a) <= EPS or np.std(b) <= EPS:
        return np.nan, np.nan, np.nan, np.nan
    spear = spearmanr(a, b)
    pear = pearsonr(a, b)
    return float(spear.statistic), float(spear.pvalue), float(pear.statistic), float(pear.pvalue)


# =============================================================================
# Torch logistic regression
# =============================================================================

class TorchLogisticRegression(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


def fit_adam_logistic_full(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    epochs: int = EPOCHS,
    seed: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_t = torch.tensor(X.astype(np.float32), device=DEVICE)
    y_t = torch.tensor(y.astype(np.float32), device=DEVICE).view(-1, 1)

    model = TorchLogisticRegression(X.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_t).detach().cpu().numpy().ravel().astype(np.float64)
        probs = sigmoid_np(logits)
        preds = (probs >= 0.5).astype(np.int64)
        w = model.linear.weight.detach().cpu().numpy().ravel().astype(np.float64)
        b = float(model.linear.bias.detach().cpu().numpy()[0])

    return {
        "w": w,
        "b": b,
        "logits": logits,
        "probs": probs,
        "preds": preds,
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "auc": float(roc_auc_score(y, probs)),
    }


def poor_mans_contrast(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)


# =============================================================================
# Rank-diff diagnostics
# =============================================================================

def first_diff_by_order(v: np.ndarray, order: np.ndarray) -> np.ndarray:
    return np.diff(v[order])


def add_comparison(row: dict, prefix: str, a: np.ndarray, b: np.ndarray):
    c, ang, ax = cosine_angle_axis(a, b)
    row[f"{prefix}_cosine"] = c
    row[f"{prefix}_angle_deg"] = ang
    row[f"{prefix}_axis_angle_deg"] = ax


def compute_rank_diff_diagnostics(wA: np.ndarray, wS: np.ndarray, prefix: str) -> tuple[dict, dict]:
    """Compute rank-ordered first-difference diagnostics for one vector pair."""
    row = {}
    payload = {}

    # Signed orderings: anti-class -> class.
    order_A = np.argsort(wA)
    order_S = np.argsort(wS)

    # Absolute gain orderings: most prominent -> least prominent.
    order_A_abs = np.argsort(-np.abs(wA))
    order_S_abs = np.argsort(-np.abs(wS))

    # Signed-order first differences.
    dwA_by_A = first_diff_by_order(wA, order_A)
    dwS_by_A = first_diff_by_order(wS, order_A)
    dwS_by_S = first_diff_by_order(wS, order_S)
    dwA_by_S = first_diff_by_order(wA, order_S)

    # Absolute-gain-order first differences.
    dwA_by_Aabs = first_diff_by_order(wA, order_A_abs)
    dwS_by_Aabs = first_diff_by_order(wS, order_A_abs)
    dwS_by_Sabs = first_diff_by_order(wS, order_S_abs)
    dwA_by_Sabs = first_diff_by_order(wA, order_S_abs)

    # Store arrays.
    payload[f"{prefix}__order_A_signed"] = order_A
    payload[f"{prefix}__order_S_signed"] = order_S
    payload[f"{prefix}__order_A_abs"] = order_A_abs
    payload[f"{prefix}__order_S_abs"] = order_S_abs

    payload[f"{prefix}__dwA_by_A"] = dwA_by_A
    payload[f"{prefix}__dwS_by_A"] = dwS_by_A
    payload[f"{prefix}__dwS_by_S"] = dwS_by_S
    payload[f"{prefix}__dwA_by_S"] = dwA_by_S

    payload[f"{prefix}__dwA_by_Aabs"] = dwA_by_Aabs
    payload[f"{prefix}__dwS_by_Aabs"] = dwS_by_Aabs
    payload[f"{prefix}__dwS_by_Sabs"] = dwS_by_Sabs
    payload[f"{prefix}__dwA_by_Sabs"] = dwA_by_Sabs

    # Ordinary vector relationship.
    add_comparison(row, f"{prefix}_raw_wA_vs_wS", wA, wS)
    spear_r, spear_p, pear_r, pear_p = safe_corrs(wA, wS)
    row[f"{prefix}_spearman_wA_wS"] = spear_r
    row[f"{prefix}_spearman_wA_wS_p"] = spear_p
    row[f"{prefix}_pearson_wA_wS"] = pear_r
    row[f"{prefix}_pearson_wA_wS_p"] = pear_p

    # Main signed-order diagnostics.
    add_comparison(row, f"{prefix}_signed_order_scene_texture_along_animal", dwA_by_A, dwS_by_A)
    add_comparison(row, f"{prefix}_signed_order_animal_texture_along_scene", dwS_by_S, dwA_by_S)
    add_comparison(row, f"{prefix}_signed_order_animal_self_texture_A_vs_S_order", dwA_by_A, dwA_by_S)
    add_comparison(row, f"{prefix}_signed_order_scene_self_texture_S_vs_A_order", dwS_by_S, dwS_by_A)

    # Absolute-gain-order diagnostics.
    add_comparison(row, f"{prefix}_abs_order_scene_texture_along_animal_abs", dwA_by_Aabs, dwS_by_Aabs)
    add_comparison(row, f"{prefix}_abs_order_animal_texture_along_scene_abs", dwS_by_Sabs, dwA_by_Sabs)
    add_comparison(row, f"{prefix}_abs_order_animal_self_texture_Aabs_vs_Sabs_order", dwA_by_Aabs, dwA_by_Sabs)
    add_comparison(row, f"{prefix}_abs_order_scene_self_texture_Sabs_vs_Aabs_order", dwS_by_Sabs, dwS_by_Aabs)

    # Roughness / spacing summaries.
    for name, arr in {
        "dwA_by_A": dwA_by_A,
        "dwS_by_A": dwS_by_A,
        "dwS_by_S": dwS_by_S,
        "dwA_by_S": dwA_by_S,
        "dwA_by_Aabs": dwA_by_Aabs,
        "dwS_by_Aabs": dwS_by_Aabs,
        "dwS_by_Sabs": dwS_by_Sabs,
        "dwA_by_Sabs": dwA_by_Sabs,
    }.items():
        row[f"{prefix}_{name}_l2_norm"] = float(np.linalg.norm(arr))
        row[f"{prefix}_{name}_mean_abs"] = float(np.mean(np.abs(arr)))
        row[f"{prefix}_{name}_std"] = float(np.std(arr))

    return row, payload


# =============================================================================
# Plotting
# =============================================================================

def plot_area_profiles(area: str, wA: np.ndarray, wS: np.ndarray, outpath: Path):
    order_A = np.argsort(wA)
    order_S = np.argsort(wS)

    x = np.arange(len(wA))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(x, wA[order_A], label="animal weights")
    axes[0, 0].plot(x, wS[order_A], label="scene weights")
    axes[0, 0].set_title(f"{area}: weights sorted by animal decoder")
    axes[0, 0].set_xlabel("ranked neuron/feature index")
    axes[0, 0].set_ylabel("logistic weight")
    axes[0, 0].legend()

    axes[0, 1].plot(x, wS[order_S], label="scene weights")
    axes[0, 1].plot(x, wA[order_S], label="animal weights")
    axes[0, 1].set_title(f"{area}: weights sorted by scene decoder")
    axes[0, 1].set_xlabel("ranked neuron/feature index")
    axes[0, 1].set_ylabel("logistic weight")
    axes[0, 1].legend()

    dx = np.arange(len(wA) - 1)
    axes[1, 0].plot(dx, np.diff(wA[order_A]), label="diff animal by animal order")
    axes[1, 0].plot(dx, np.diff(wS[order_A]), label="diff scene by animal order")
    axes[1, 0].set_title("First differences along animal ordering")
    axes[1, 0].set_xlabel("ranked gap index")
    axes[1, 0].set_ylabel("first difference")
    axes[1, 0].legend()

    axes[1, 1].plot(dx, np.diff(wS[order_S]), label="diff scene by scene order")
    axes[1, 1].plot(dx, np.diff(wA[order_S]), label="diff animal by scene order")
    axes[1, 1].set_title("First differences along scene ordering")
    axes[1, 1].set_xlabel("ranked gap index")
    axes[1, 1].set_ylabel("first difference")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_area_scatter(area: str, wA: np.ndarray, wS: np.ndarray, outpath: Path):
    order_A = np.argsort(wA)
    order_S = np.argsort(wS)

    dwA_by_A = np.diff(wA[order_A])
    dwS_by_A = np.diff(wS[order_A])
    dwS_by_S = np.diff(wS[order_S])
    dwA_by_S = np.diff(wA[order_S])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(dwA_by_A, dwS_by_A, s=8, alpha=0.55)
    axes[0].axhline(0, linewidth=0.8)
    axes[0].axvline(0, linewidth=0.8)
    axes[0].set_title(f"{area}: scene texture along animal order")
    axes[0].set_xlabel("diff animal weights, animal order")
    axes[0].set_ylabel("diff scene weights, animal order")

    axes[1].scatter(dwS_by_S, dwA_by_S, s=8, alpha=0.55)
    axes[1].axhline(0, linewidth=0.8)
    axes[1].axvline(0, linewidth=0.8)
    axes[1].set_title(f"{area}: animal texture along scene order")
    axes[1].set_xlabel("diff scene weights, scene order")
    axes[1].set_ylabel("diff animal weights, scene order")

    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_summary_bars(summary_df: pd.DataFrame, outdir: Path):
    if summary_df.empty:
        return

    df = summary_df.sort_values("area").copy()

    metrics = [
        (
            "logistic_signed_order_scene_texture_along_animal_axis_angle_deg",
            "Scene texture along animal ordering: axis angle",
            "scene_texture_along_animal_axis_angle.png",
        ),
        (
            "logistic_signed_order_animal_texture_along_scene_axis_angle_deg",
            "Animal texture along scene ordering: axis angle",
            "animal_texture_along_scene_axis_angle.png",
        ),
        (
            "logistic_signed_order_animal_self_texture_A_vs_S_order_axis_angle_deg",
            "Animal self-texture: animal order vs scene order",
            "animal_self_texture_axis_angle.png",
        ),
        (
            "logistic_signed_order_scene_self_texture_S_vs_A_order_axis_angle_deg",
            "Scene self-texture: scene order vs animal order",
            "scene_self_texture_axis_angle.png",
        ),
        (
            "logistic_spearman_wA_wS",
            "Spearman rank correlation between animal and scene weights",
            "spearman_animal_scene_weights.png",
        ),
    ]

    for col, title, fname in metrics:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar(df["area"], df[col])
        ax.set_title(title)
        ax.set_xlabel("brain area")
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=160)
        plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 88)
    print("Rank-ordered first-difference geometry of animal and scene decoder axes")
    print("=" * 88)
    print(f"Device: {DEVICE}")
    print(f"LR={LR}, WEIGHT_DECAY={WEIGHT_DECAY}, EPOCHS={EPOCHS}")
    print(f"Output directory: {OUTDIR}")

    X_raw = load_neural_matrix(NEURAL_PATH)
    y_anim_raw = load_labels(LABEL_FILES["animals_vs_everything"])
    y_scene_raw = load_labels(LABEL_FILES["scenes_vs_everything"])
    brain_area = load_brain_areas(BRAIN_AREA_PATH)

    X = align_x_to_labels(X_raw, y_anim_raw)
    X_clean, brain_area_clean, good_cols = clean_features_global(X, brain_area)

    if len(y_anim_raw) != X_clean.shape[0] or len(y_scene_raw) != X_clean.shape[0]:
        raise ValueError(
            f"Label length mismatch: X images={X_clean.shape[0]}, "
            f"animal labels={len(y_anim_raw)}, scene labels={len(y_scene_raw)}"
        )

    # Critical choice: compare tasks on the same set of images and same scaler.
    common_mask = (y_anim_raw != -1) & (y_scene_raw != -1)
    X_common = X_clean[common_mask]
    y_anim = y_anim_raw[common_mask].astype(np.int64)
    y_scene = y_scene_raw[common_mask].astype(np.int64)
    common_indices = np.where(common_mask)[0]

    print()
    print("Shared image set:")
    print(f"  X_common shape: {X_common.shape}")
    print(f"  animal label counts [0, 1]: {np.bincount(y_anim, minlength=2)}")
    print(f"  scene  label counts [0, 1]: {np.bincount(y_scene, minlength=2)}")

    areas = sorted(np.unique(brain_area_clean).tolist())
    print()
    print("Areas:")
    for area in areas:
        print(f"  {area}: {np.sum(brain_area_clean == area)} features")

    summary_rows = []
    payload = {
        "common_indices": common_indices,
        "y_anim": y_anim,
        "y_scene": y_scene,
        "good_cols": good_cols,
    }

    for area_idx, area in enumerate(areas):
        print()
        print("#" * 88)
        print(f"Area {area_idx + 1}/{len(areas)}: {area}")
        print("#" * 88)

        area_mask = brain_area_clean == area
        X_area_raw = X_common[:, area_mask]

        if X_area_raw.shape[1] < 3:
            print(f"[SKIP] {area}: fewer than 3 features.")
            continue

        # Shared standardization across the common images for this area.
        scaler = StandardScaler()
        X_area = scaler.fit_transform(X_area_raw)

        start = time.time()

        anim_fit = fit_adam_logistic_full(
            X_area,
            y_anim,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            epochs=EPOCHS,
            seed=RANDOM_SEED + 1000 * area_idx + 1,
        )
        scene_fit = fit_adam_logistic_full(
            X_area,
            y_scene,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            epochs=EPOCHS,
            seed=RANDOM_SEED + 1000 * area_idx + 2,
        )

        wA = anim_fit["w"]
        wS = scene_fit["w"]
        bA = anim_fit["b"]
        bS = scene_fit["b"]

        vA = poor_mans_contrast(X_area, y_anim)
        vS = poor_mans_contrast(X_area, y_scene)

        row = {
            "area": area,
            "n_samples_common": int(X_area.shape[0]),
            "n_features": int(X_area.shape[1]),
            "animal_label_count_0": int(np.sum(y_anim == 0)),
            "animal_label_count_1": int(np.sum(y_anim == 1)),
            "scene_label_count_0": int(np.sum(y_scene == 0)),
            "scene_label_count_1": int(np.sum(y_scene == 1)),
            "animal_logistic_accuracy_full": anim_fit["accuracy"],
            "animal_logistic_balanced_accuracy_full": anim_fit["balanced_accuracy"],
            "animal_logistic_auc_full": anim_fit["auc"],
            "scene_logistic_accuracy_full": scene_fit["accuracy"],
            "scene_logistic_balanced_accuracy_full": scene_fit["balanced_accuracy"],
            "scene_logistic_auc_full": scene_fit["auc"],
            "animal_bias": float(bA),
            "scene_bias": float(bS),
            "animal_w_norm": float(np.linalg.norm(wA)),
            "scene_w_norm": float(np.linalg.norm(wS)),
            "animal_contrast_norm": float(np.linalg.norm(vA)),
            "scene_contrast_norm": float(np.linalg.norm(vS)),
            "elapsed_seconds": float(time.time() - start),
        }

        # Logistic rank-diff diagnostics.
        logistic_row, logistic_payload = compute_rank_diff_diagnostics(wA, wS, "logistic")
        row.update(logistic_row)

        # Poor-man contrast rank-diff diagnostics as a useful comparison.
        contrast_row, contrast_payload = compute_rank_diff_diagnostics(vA, vS, "contrast")
        row.update(contrast_row)

        prefix = area
        payload[f"{prefix}__w_animal"] = wA
        payload[f"{prefix}__w_scene"] = wS
        payload[f"{prefix}__v_animal_contrast"] = vA
        payload[f"{prefix}__v_scene_contrast"] = vS
        payload[f"{prefix}__area_feature_indices_clean"] = np.where(area_mask)[0]

        for k, val in logistic_payload.items():
            payload[f"{prefix}__{k}"] = val
        for k, val in contrast_payload.items():
            payload[f"{prefix}__{k}"] = val

        summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows).sort_values("area")
        summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
        np.savez_compressed(OUT_WEIGHTS_NPZ, **payload)

        plot_area_profiles(
            area,
            wA,
            wS,
            PLOT_DIR / f"{area}_logistic_rank_profiles_and_diffs.png",
        )
        plot_area_scatter(
            area,
            wA,
            wS,
            PLOT_DIR / f"{area}_logistic_first_difference_scatter.png",
        )

        print(
            f"{area:>7s} | "
            f"logistic angle={row['logistic_raw_wA_vs_wS_angle_deg']:7.2f}° "
            f"axis={row['logistic_raw_wA_vs_wS_axis_angle_deg']:6.2f}° "
            f"spearman={row['logistic_spearman_wA_wS']: .3f} | "
            f"scene-texture-along-animal axis="
            f"{row['logistic_signed_order_scene_texture_along_animal_axis_angle_deg']:6.2f}° | "
            f"animal-texture-along-scene axis="
            f"{row['logistic_signed_order_animal_texture_along_scene_axis_angle_deg']:6.2f}°"
        )
        print(f"[SAVED] {OUT_SUMMARY_CSV}")
        print(f"[SAVED] {OUT_WEIGHTS_NPZ}")

    summary_df = pd.DataFrame(summary_rows).sort_values("area")
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
    np.savez_compressed(OUT_WEIGHTS_NPZ, **payload)
    plot_summary_bars(summary_df, PLOT_DIR)

    config = {
        "brain_area_path": str(BRAIN_AREA_PATH),
        "neural_path": str(NEURAL_PATH),
        "label_files": {k: str(v) for k, v in LABEL_FILES.items()},
        "outdir": str(OUTDIR),
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "random_seed": RANDOM_SEED,
        "device": DEVICE,
        "n_common_images": int(np.sum(common_mask)),
        "areas": areas,
        "method_note": (
            "For each area, animal and scene logistic axes are fit on the same standardized "
            "common image matrix. Signed order means argsort(w), from negative to positive. "
            "Absolute order means argsort(-abs(w)), from largest absolute gain to smallest. "
            "First differences after sorting are rank-ordered/order-statistic derivatives, "
            "not anatomical spatial derivatives."
        ),
        "main_interpretation_columns": {
            "logistic_spearman_wA_wS": "Rank correlation between animal and scene logistic weights.",
            "logistic_raw_wA_vs_wS_axis_angle_deg": "Global logistic axis angle ignoring sign.",
            "logistic_signed_order_scene_texture_along_animal_axis_angle_deg": "Whether scene first-difference texture follows animal first-difference spacing along animal-ranked neurons.",
            "logistic_signed_order_animal_texture_along_scene_axis_angle_deg": "Whether animal first-difference texture follows scene first-difference spacing along scene-ranked neurons.",
            "logistic_signed_order_animal_self_texture_A_vs_S_order_axis_angle_deg": "Whether the animal axis has similar texture under animal ordering and scene ordering.",
            "logistic_signed_order_scene_self_texture_S_vs_A_order_axis_angle_deg": "Whether the scene axis has similar texture under scene ordering and animal ordering.",
        },
    }

    with open(OUT_CONFIG_JSON, "w") as f:
        json.dump(config, f, indent=2)

    print()
    print("=" * 88)
    print("FINAL SUMMARY")
    print("=" * 88)

    display_cols = [
        "area",
        "n_features",
        "logistic_raw_wA_vs_wS_angle_deg",
        "logistic_raw_wA_vs_wS_axis_angle_deg",
        "logistic_spearman_wA_wS",
        "logistic_signed_order_scene_texture_along_animal_axis_angle_deg",
        "logistic_signed_order_animal_texture_along_scene_axis_angle_deg",
        "logistic_signed_order_animal_self_texture_A_vs_S_order_axis_angle_deg",
        "logistic_signed_order_scene_self_texture_S_vs_A_order_axis_angle_deg",
    ]
    existing_display_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[existing_display_cols].to_string(index=False))

    print()
    print(f"[DONE] Summary CSV: {OUT_SUMMARY_CSV}")
    print(f"[DONE] Arrays NPZ:   {OUT_WEIGHTS_NPZ}")
    print(f"[DONE] Config JSON:  {OUT_CONFIG_JSON}")
    print(f"[DONE] Plots dir:    {PLOT_DIR}")


if __name__ == "__main__":
    main()
