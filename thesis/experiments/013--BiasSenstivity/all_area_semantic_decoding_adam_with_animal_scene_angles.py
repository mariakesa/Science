#!/usr/bin/env python3
"""
All brain area Adam logistic decoders for:

1. Animals vs everything else
2. Scenes vs everything else

Runs leave-one-out decoding separately for every brain area.

NEW GEOMETRY DIAGNOSTICS
------------------------

For every LOO fold, this script computes:

    signed_boundary_distance = -b / ||w||
    abs_boundary_distance    = |b| / ||w||

Interpretation:

    signed_boundary_distance:
        Signed location of the decision boundary relative to the standardized
        training-set origin, along the learned model direction.

    abs_boundary_distance:
        Amount of standardized movement along the learned model direction needed
        to overcome the data-independent bias term.

Because StandardScaler is fit inside each LOO fold, x=0 means:
    "the mean training neural response for that fold and brain area."

Expected data:
    X:          images x neurons OR neurons x images
    labels:     length n_images, values {-1, 0, 1}
    brain_area: length n_neurons, area name per neuron

Label convention:
    -1 = exclude / unlabeled
     0 = everything else
     1 = target class
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
)


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

OUTDIR = Path("/home/maria/Science/results/all_area_semantic_decoding_adam_with_boundary_ratio")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY_CSV = OUTDIR / "all_area_semantic_decoding_summary_with_boundary_ratio.csv"
OUT_PREDICTIONS_NPZ = OUTDIR / "all_area_semantic_decoding_predictions_with_boundary_ratio.npz"
OUT_CONFIG_JSON = OUTDIR / "config.json"

OUT_ANGLE_CSV = OUTDIR / "animal_scene_population_direction_angles.csv"
OUT_ANGLE_PNG = OUTDIR / "animal_scene_population_direction_angles.png"
OUT_AXIS_ANGLE_PNG = OUTDIR / "animal_scene_population_axis_angles.png"


# =============================================================================
# Adam logistic settings
# =============================================================================

LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 3000

RANDOM_SEED = 0
EPS = 1e-12

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Utilities
# =============================================================================

def sigmoid_np(z: np.ndarray | float) -> np.ndarray | float:
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def safe_nanmean(x: np.ndarray) -> float:
    if np.all(np.isnan(x)):
        return float("nan")
    return float(np.nanmean(x))


def safe_nanstd(x: np.ndarray) -> float:
    if np.all(np.isnan(x)):
        return float("nan")
    return float(np.nanstd(x))


def vector_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def cosine_similarity(u: np.ndarray, v: np.ndarray, eps: float = EPS) -> float:
    """
    Cosine between two vectors.

    Returns NaN if either vector has near-zero norm.
    """
    nu = vector_norm(u)
    nv = vector_norm(v)
    if nu <= eps or nv <= eps:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def angle_degrees_from_cosine(cosine: float) -> float:
    """
    Signed direction angle in degrees.

    0 deg   = same direction
    90 deg  = orthogonal directions
    180 deg = opposite directions
    """
    if not np.isfinite(cosine):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def axis_angle_degrees_from_cosine(cosine: float) -> float:
    """
    Axis angle in degrees, ignoring vector sign.

    0 deg  = same line, even if opposite signs
    90 deg = orthogonal axes

    This is useful because a decoder/contrast direction can flip sign depending
    on which class is called positive.
    """
    if not np.isfinite(cosine):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(abs(cosine), 0.0, 1.0))))


def contrast_direction(X_std: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Difference-of-means contrast direction in standardized neural space:

        mean(target class 1) - mean(everything else class 0)

    This is the poor-man's classifier / semantic contrast vector.
    """
    y = np.asarray(y).astype(np.int64).ravel()
    if len(np.unique(y)) != 2:
        raise ValueError("contrast_direction requires both labels 0 and 1.")
    return X_std[y == 1].mean(axis=0) - X_std[y == 0].mean(axis=0)


def load_labels(path: Path) -> np.ndarray:
    """
    Handles either:
      - plain labels.npy array
      - dict saved as npy with key "labels"
    """
    obj = np.load(path, allow_pickle=True)

    if isinstance(obj, np.ndarray) and obj.shape == () and isinstance(obj.item(), dict):
        d = obj.item()
        if "labels" not in d:
            raise ValueError(f"Label dict at {path} does not contain key 'labels'.")
        labels = d["labels"]
    else:
        labels = obj

    labels = np.asarray(labels).astype(np.int64).ravel()

    print(f"Loaded labels from: {path}")
    print(f"Label shape: {labels.shape}")
    unique, counts = np.unique(labels, return_counts=True)
    print("Raw label counts including possible -1:")
    print(dict(zip(unique.tolist(), counts.tolist())))

    bad = set(unique.tolist()) - {-1, 0, 1}
    if bad:
        raise ValueError(f"Unexpected label values in {path}: {bad}. Expected only -1, 0, 1.")

    return labels


def load_brain_areas(path: Path) -> np.ndarray:
    areas = np.load(path, allow_pickle=True)
    areas = np.asarray(areas).astype(str).ravel()

    print(f"Loaded brain areas from: {path}")
    print(f"Brain area shape: {areas.shape}")

    unique, counts = np.unique(areas, return_counts=True)
    print("Brain area counts:")
    for a, c in zip(unique, counts):
        print(f"  {a}: {c}")

    return areas


def load_neural_matrix(path: Path) -> np.ndarray:
    X = np.load(path, allow_pickle=True)
    X = np.asarray(X, dtype=np.float64)

    print(f"Loaded neural matrix from: {path}")
    print(f"Raw X shape: {X.shape}")

    return X


def align_x_to_labels(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Returns X as images x neurons/features.
    """
    if X.shape[0] == len(y):
        print("[INFO] X already looks like images x features.")
        return X

    if X.shape[1] == len(y):
        print("[INFO] Transposing X to images x features.")
        return X.T

    raise ValueError(
        f"Cannot align X with labels. X={X.shape}, labels={y.shape}. "
        "Expected one X axis to match number of labels."
    )


def clean_features_global(X: np.ndarray, brain_area: np.ndarray):
    """
    Remove non-finite and constant features once, keeping brain_area aligned.
    """
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

    X_clean = X[:, good_cols]
    brain_area_clean = brain_area[good_cols]

    print()
    print("=" * 80)
    print("Global feature cleaning")
    print("=" * 80)
    print(f"X clean shape: {X_clean.shape}")
    print(f"Removed bad/nonconstant features: {np.sum(~good_cols)}")

    return X_clean, brain_area_clean, good_cols


# =============================================================================
# Torch logistic regression
# =============================================================================

class TorchLogisticRegression(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


def fit_adam_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    epochs: int = EPOCHS,
    seed: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_t = torch.tensor(X_train.astype(np.float32), device=DEVICE)
    y_t = torch.tensor(y_train.astype(np.float32), device=DEVICE).view(-1, 1)

    model = TorchLogisticRegression(X_train.shape[1]).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

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
        w = model.linear.weight.detach().cpu().numpy().ravel().astype(np.float64)
        b = float(model.linear.bias.detach().cpu().numpy()[0])

    return w, b


# =============================================================================
# LOO decoder
# =============================================================================

def run_loo_decoder_for_area(
    X_area: np.ndarray,
    y: np.ndarray,
    *,
    task_name: str,
    area_name: str,
    print_every: int = 10,
):
    n, d = X_area.shape

    if d == 0:
        raise ValueError(f"No features found for area {area_name}.")

    if len(np.unique(y)) != 2:
        raise ValueError(
            f"{task_name}/{area_name}: y does not contain both classes after filtering."
        )

    logits = np.zeros(n, dtype=np.float64)
    probs = np.zeros(n, dtype=np.float64)
    preds = np.zeros(n, dtype=np.int64)

    # -------------------------------------------------------------------------
    # New geometry arrays
    # -------------------------------------------------------------------------
    w_norms = np.zeros(n, dtype=np.float64)
    biases = np.zeros(n, dtype=np.float64)

    signed_boundary_distances = np.zeros(n, dtype=np.float64)
    abs_boundary_distances = np.zeros(n, dtype=np.float64)

    # Useful extra diagnostic:
    # signed_margin_of_test_point = (w^T x_test + b) / ||w||
    # labeled_margin_of_test_point = y_signed * signed_margin
    signed_margins = np.zeros(n, dtype=np.float64)
    labeled_margins = np.zeros(n, dtype=np.float64)

    print()
    print("#" * 80)
    print("Running LOO Adam decoder")
    print(f"Task: {task_name}")
    print(f"Area: {area_name}")
    print("#" * 80)
    print(f"X_area shape: {X_area.shape}")
    print(f"Label counts [0, 1]: {np.bincount(y, minlength=2)}")
    print(f"Device: {DEVICE}")

    start = time.time()

    for test_idx in range(n):
        train_mask = np.arange(n) != test_idx

        X_train_raw = X_area[train_mask]
        y_train = y[train_mask]
        X_test_raw = X_area[~train_mask]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        w, b = fit_adam_logistic(
            X_train,
            y_train,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            epochs=EPOCHS,
            seed=10_000 + test_idx,
        )

        w_norm = float(np.linalg.norm(w))
        logit = float(X_test[0] @ w + b)
        prob = float(sigmoid_np(logit))
        pred = int(prob >= 0.5)

        if w_norm <= EPS:
            signed_boundary_distance = np.nan
            abs_boundary_distance = np.nan
            signed_margin = np.nan
            labeled_margin = np.nan
        else:
            signed_boundary_distance = -b / w_norm
            abs_boundary_distance = abs(b) / w_norm

            signed_margin = logit / w_norm

            # Convert labels {0,1} to {-1,+1}
            y_signed = 1 if y[test_idx] == 1 else -1
            labeled_margin = y_signed * signed_margin

        logits[test_idx] = logit
        probs[test_idx] = prob
        preds[test_idx] = pred

        w_norms[test_idx] = w_norm
        biases[test_idx] = b
        signed_boundary_distances[test_idx] = signed_boundary_distance
        abs_boundary_distances[test_idx] = abs_boundary_distance
        signed_margins[test_idx] = signed_margin
        labeled_margins[test_idx] = labeled_margin

        if (test_idx + 1) % print_every == 0 or test_idx == 0 or test_idx == n - 1:
            running_bal_acc = balanced_accuracy_score(y[: test_idx + 1], preds[: test_idx + 1])

            print(
                f"[{task_name} | {area_name} | LOO {test_idx + 1:03d}/{n}] "
                f"true={y[test_idx]} prob={prob:.4f} pred={pred} "
                f"running_bal_acc={running_bal_acc:.4f} "
                f"||w||={w_norm:.4f} b={b:.4f} "
                f"-b/||w||={signed_boundary_distance:.4f} "
                f"|b|/||w||={abs_boundary_distance:.4f}"
            )

    elapsed = time.time() - start

    acc = accuracy_score(y, preds)
    bal_acc = balanced_accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)
    cm = confusion_matrix(y, preds, labels=[0, 1])

    correct = preds == y

    # -------------------------------------------------------------------------
    # Summary stats for the geometry diagnostics
    # -------------------------------------------------------------------------
    mean_w_norm = safe_nanmean(w_norms)
    std_w_norm = safe_nanstd(w_norms)

    mean_bias = safe_nanmean(biases)
    std_bias = safe_nanstd(biases)

    mean_signed_boundary_distance = safe_nanmean(signed_boundary_distances)
    std_signed_boundary_distance = safe_nanstd(signed_boundary_distances)

    mean_abs_boundary_distance = safe_nanmean(abs_boundary_distances)
    std_abs_boundary_distance = safe_nanstd(abs_boundary_distances)

    median_abs_boundary_distance = float(np.nanmedian(abs_boundary_distances))
    median_signed_boundary_distance = float(np.nanmedian(signed_boundary_distances))

    mean_signed_margin = safe_nanmean(signed_margins)
    std_signed_margin = safe_nanstd(signed_margins)

    mean_labeled_margin = safe_nanmean(labeled_margins)
    std_labeled_margin = safe_nanstd(labeled_margins)

    min_labeled_margin = float(np.nanmin(labeled_margins))
    q05_labeled_margin = float(np.nanquantile(labeled_margins, 0.05))
    q50_labeled_margin = float(np.nanquantile(labeled_margins, 0.50))
    q95_labeled_margin = float(np.nanquantile(labeled_margins, 0.95))

    print()
    print("=" * 80)
    print(f"{task_name} / {area_name} LOO summary")
    print("=" * 80)
    print(f"Features:                         {d}")
    print(f"Accuracy:                         {acc:.4f}")
    print(f"Balanced accuracy:                {bal_acc:.4f}")
    print(f"AUC:                              {auc:.4f}")
    print(f"Elapsed seconds:                  {elapsed:.1f}")
    print()
    print("Boundary-ratio diagnostics:")
    print(f"Mean ||w||:                       {mean_w_norm:.6f}")
    print(f"Std  ||w||:                       {std_w_norm:.6f}")
    print(f"Mean bias b:                      {mean_bias:.6f}")
    print(f"Std  bias b:                      {std_bias:.6f}")
    print(f"Mean signed -b/||w||:             {mean_signed_boundary_distance:.6f}")
    print(f"Std  signed -b/||w||:             {std_signed_boundary_distance:.6f}")
    print(f"Median signed -b/||w||:           {median_signed_boundary_distance:.6f}")
    print(f"Mean abs |b|/||w||:               {mean_abs_boundary_distance:.6f}")
    print(f"Std  abs |b|/||w||:               {std_abs_boundary_distance:.6f}")
    print(f"Median abs |b|/||w||:             {median_abs_boundary_distance:.6f}")
    print()
    print("Held-out margin diagnostics:")
    print(f"Mean signed test margin:          {mean_signed_margin:.6f}")
    print(f"Std  signed test margin:          {std_signed_margin:.6f}")
    print(f"Mean labeled test margin:         {mean_labeled_margin:.6f}")
    print(f"Std  labeled test margin:         {std_labeled_margin:.6f}")
    print(f"Min labeled test margin:          {min_labeled_margin:.6f}")
    print(f"Q05 labeled test margin:          {q05_labeled_margin:.6f}")
    print(f"Q50 labeled test margin:          {q50_labeled_margin:.6f}")
    print(f"Q95 labeled test margin:          {q95_labeled_margin:.6f}")
    print()
    print("Confusion matrix rows=true [0, 1], cols=pred [0, 1]:")
    print(cm)

    return {
        "task_name": task_name,
        "area_name": area_name,
        "n_samples": n,
        "n_features": d,
        "label_count_0": int(np.sum(y == 0)),
        "label_count_1": int(np.sum(y == 1)),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "auc": float(auc),
        "confusion_matrix": cm,
        "logits": logits,
        "probs": probs,
        "preds": preds,
        "correct": correct.astype(np.int64),
        "elapsed_seconds": float(elapsed),

        # Per-fold geometry arrays
        "w_norms": w_norms,
        "biases": biases,
        "signed_boundary_distances": signed_boundary_distances,
        "abs_boundary_distances": abs_boundary_distances,
        "signed_margins": signed_margins,
        "labeled_margins": labeled_margins,

        # Geometry summaries
        "mean_w_norm": mean_w_norm,
        "std_w_norm": std_w_norm,
        "mean_bias": mean_bias,
        "std_bias": std_bias,
        "mean_signed_boundary_distance": mean_signed_boundary_distance,
        "std_signed_boundary_distance": std_signed_boundary_distance,
        "median_signed_boundary_distance": median_signed_boundary_distance,
        "mean_abs_boundary_distance": mean_abs_boundary_distance,
        "std_abs_boundary_distance": std_abs_boundary_distance,
        "median_abs_boundary_distance": median_abs_boundary_distance,

        # Held-out margin summaries
        "mean_signed_margin": mean_signed_margin,
        "std_signed_margin": std_signed_margin,
        "mean_labeled_margin": mean_labeled_margin,
        "std_labeled_margin": std_labeled_margin,
        "min_labeled_margin": min_labeled_margin,
        "q05_labeled_margin": q05_labeled_margin,
        "q50_labeled_margin": q50_labeled_margin,
        "q95_labeled_margin": q95_labeled_margin,
    }



# =============================================================================
# Animal-vs-scene direction geometry
# =============================================================================

def compute_animal_scene_angle_diagnostics(
    X_clean_all: np.ndarray,
    brain_area_clean: np.ndarray,
    labels_by_task: dict[str, np.ndarray],
    all_areas: list[str],
):
    """
    Computes angles between animal and scene directions inside each brain area.

    For each area, we fit ONE StandardScaler on the shared set of images that has
    valid labels for both tasks. Then both contrast vectors and both logistic
    weight vectors live in the same standardized feature coordinate system.

    Directions computed:
      1. contrast direction:
             mean(X | task label 1) - mean(X | task label 0)

      2. full-data logistic direction:
             Adam logistic weight vector trained on all shared valid images

    For each pair we report:
      - cosine
      - signed angle in degrees, 0..180
      - axis angle in degrees, 0..90, ignoring sign
    """
    required = ["animals_vs_everything", "scenes_vs_everything"]
    missing = [k for k in required if k not in labels_by_task]
    if missing:
        raise ValueError(f"Missing required label arrays for angle diagnostics: {missing}")

    y_anim_raw = labels_by_task["animals_vs_everything"]
    y_scene_raw = labels_by_task["scenes_vs_everything"]

    if len(y_anim_raw) != X_clean_all.shape[0] or len(y_scene_raw) != X_clean_all.shape[0]:
        raise ValueError(
            "Angle diagnostics require both label arrays to have one value per image."
        )

    shared_mask = (y_anim_raw != -1) & (y_scene_raw != -1)

    print()
    print("=" * 80)
    print("Animal-vs-scene population direction angle diagnostics")
    print("=" * 80)
    print(
        "Using shared valid stimulus set so animal and scene directions are "
        "measured in the same standardized coordinate system."
    )
    print(f"Shared valid images: {int(shared_mask.sum())} / {len(shared_mask)}")

    rows = []

    for area_name in all_areas:
        area_mask = brain_area_clean == area_name
        X_area_raw = X_clean_all[shared_mask][:, area_mask]
        y_anim = y_anim_raw[shared_mask].astype(np.int64)
        y_scene = y_scene_raw[shared_mask].astype(np.int64)

        row = {
            "area": area_name,
            "n_shared_images": int(shared_mask.sum()),
            "n_features": int(area_mask.sum()),
            "animal_label_count_0": int(np.sum(y_anim == 0)),
            "animal_label_count_1": int(np.sum(y_anim == 1)),
            "scene_label_count_0": int(np.sum(y_scene == 0)),
            "scene_label_count_1": int(np.sum(y_scene == 1)),
        }

        if X_area_raw.shape[1] == 0:
            rows.append(row | {"error": "no_features"})
            continue

        if len(np.unique(y_anim)) != 2 or len(np.unique(y_scene)) != 2:
            rows.append(row | {"error": "one_task_missing_a_class"})
            continue

        scaler = StandardScaler()
        X_area = scaler.fit_transform(X_area_raw)

        # ------------------------------------------------------------------
        # Poor-man / difference-of-means semantic directions
        # ------------------------------------------------------------------
        v_anim = contrast_direction(X_area, y_anim)
        v_scene = contrast_direction(X_area, y_scene)

        contrast_cos = cosine_similarity(v_anim, v_scene)
        contrast_angle = angle_degrees_from_cosine(contrast_cos)
        contrast_axis_angle = axis_angle_degrees_from_cosine(contrast_cos)

        # ------------------------------------------------------------------
        # Full-data Adam logistic directions, trained in the same standardized
        # space. This is not LOO; it is a geometry diagnostic.
        # ------------------------------------------------------------------
        w_anim, b_anim = fit_adam_logistic(
            X_area,
            y_anim,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            epochs=EPOCHS,
            seed=123_001,
        )
        w_scene, b_scene = fit_adam_logistic(
            X_area,
            y_scene,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            epochs=EPOCHS,
            seed=123_002,
        )

        logistic_cos = cosine_similarity(w_anim, w_scene)
        logistic_angle = angle_degrees_from_cosine(logistic_cos)
        logistic_axis_angle = axis_angle_degrees_from_cosine(logistic_cos)

        rows.append(
            row
            | {
                "error": "",
                "contrast_cosine": contrast_cos,
                "contrast_angle_deg": contrast_angle,
                "contrast_axis_angle_deg": contrast_axis_angle,
                "contrast_anim_norm": vector_norm(v_anim),
                "contrast_scene_norm": vector_norm(v_scene),
                "contrast_dot": float(np.dot(v_anim, v_scene)),
                "logistic_cosine": logistic_cos,
                "logistic_angle_deg": logistic_angle,
                "logistic_axis_angle_deg": logistic_axis_angle,
                "logistic_anim_norm": vector_norm(w_anim),
                "logistic_scene_norm": vector_norm(w_scene),
                "logistic_dot": float(np.dot(w_anim, w_scene)),
                "logistic_anim_bias": float(b_anim),
                "logistic_scene_bias": float(b_scene),
            }
        )

        print(
            f"{area_name:>8s} | "
            f"contrast angle={contrast_angle:7.2f}° "
            f"axis={contrast_axis_angle:6.2f}° "
            f"cos={contrast_cos: .3f} || "
            f"logistic angle={logistic_angle:7.2f}° "
            f"axis={logistic_axis_angle:6.2f}° "
            f"cos={logistic_cos: .3f}"
        )

    angle_df = pd.DataFrame(rows)
    angle_df = angle_df.sort_values("contrast_axis_angle_deg", ascending=True)
    angle_df.to_csv(OUT_ANGLE_CSV, index=False)

    print()
    print(f"[SAVED] {OUT_ANGLE_CSV}")

    plot_angle_diagnostics(angle_df)

    return angle_df


def plot_angle_diagnostics(angle_df: pd.DataFrame):
    """
    Saves two compact bar plots:
      1. signed animal-scene direction angle, where 180 means opposite directions
      2. axis angle, where 0 means same line/opposite signs allowed
    """
    df = angle_df.copy()
    df = df[df["error"].fillna("") == ""].copy()
    if df.empty:
        print("[WARN] No valid angle rows to plot.")
        return

    df = df.sort_values("contrast_angle_deg", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.38

    ax.bar(x - width / 2, df["contrast_angle_deg"], width, label="contrast")
    ax.bar(x + width / 2, df["logistic_angle_deg"], width, label="logistic")

    ax.axhline(90, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["area"], rotation=45, ha="right")
    ax.set_ylabel("Animal vs scene signed direction angle (degrees)")
    ax.set_title("Animal contrast direction vs scene contrast direction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_ANGLE_PNG, dpi=200)
    plt.close(fig)

    df = df.sort_values("contrast_axis_angle_deg", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.38

    ax.bar(x - width / 2, df["contrast_axis_angle_deg"], width, label="contrast")
    ax.bar(x + width / 2, df["logistic_axis_angle_deg"], width, label="logistic")

    ax.axhline(45, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["area"], rotation=45, ha="right")
    ax.set_ylabel("Animal vs scene axis angle (degrees, sign ignored)")
    ax.set_title("Do animal and scene contrasts use the same population axis?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_AXIS_ANGLE_PNG, dpi=200)
    plt.close(fig)

    print(f"[SAVED] {OUT_ANGLE_PNG}")
    print(f"[SAVED] {OUT_AXIS_ANGLE_PNG}")


# =============================================================================
# Main
# =============================================================================

def main():
    print()
    print("=" * 80)
    print("All-area semantic decoding with Adam logistic LOO + boundary ratio")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"LR={LR}, WEIGHT_DECAY={WEIGHT_DECAY}, EPOCHS={EPOCHS}")

    X_raw = load_neural_matrix(NEURAL_PATH)
    brain_area = load_brain_areas(BRAIN_AREA_PATH)

    # Use the first label file only to orient X.
    first_task = next(iter(LABEL_FILES))
    first_label_path = LABEL_FILES[first_task]
    if not first_label_path.exists():
        raise FileNotFoundError(
            f"Could not find first label file: {first_label_path}\n"
            "Edit LABEL_FILES at the top of this script."
        )

    y_for_alignment = load_labels(first_label_path)
    X = align_x_to_labels(X_raw, y_for_alignment)

    if X.shape[1] != len(brain_area):
        raise ValueError(
            f"After alignment, X features={X.shape[1]} but brain_area length={len(brain_area)}."
        )

    # Clean features based on all images before label-specific filtering.
    X_clean_all, brain_area_clean, good_cols = clean_features_global(X, brain_area)

    all_areas = sorted(np.unique(brain_area_clean).tolist())

    print()
    print("=" * 80)
    print("Areas to decode")
    print("=" * 80)
    for area in all_areas:
        print(f"  {area}: {np.sum(brain_area_clean == area)} features")

    # -------------------------------------------------------------------------
    # Cross-task geometry: animal contrast direction vs scene contrast direction
    # -------------------------------------------------------------------------
    labels_by_task_for_angles = {
        task_name: load_labels(label_path)
        for task_name, label_path in LABEL_FILES.items()
    }
    angle_df = compute_animal_scene_angle_diagnostics(
        X_clean_all=X_clean_all,
        brain_area_clean=brain_area_clean,
        labels_by_task=labels_by_task_for_angles,
        all_areas=all_areas,
    )

    summary_rows = []
    prediction_payload = {}

    for task_name, label_path in LABEL_FILES.items():
        if not label_path.exists():
            raise FileNotFoundError(
                f"Could not find label file for task '{task_name}': {label_path}\n"
                "Edit LABEL_FILES at the top of this script."
            )

        print()
        print("=" * 80)
        print(f"Task: {task_name}")
        print("=" * 80)

        y_raw = load_labels(label_path)

        if len(y_raw) != X_clean_all.shape[0]:
            raise ValueError(
                f"Label length mismatch for task {task_name}: "
                f"labels={len(y_raw)}, X images={X_clean_all.shape[0]}"
            )

        labeled_mask = y_raw != -1
        original_indices = np.where(labeled_mask)[0]

        X_task = X_clean_all[labeled_mask]
        y_task = y_raw[labeled_mask].astype(np.int64)

        print()
        print("After excluding label -1:")
        print(f"  X_task shape: {X_task.shape}")
        print(f"  y_task shape: {y_task.shape}")
        print(f"  Label counts [0, 1]: {np.bincount(y_task, minlength=2)}")

        if len(np.unique(y_task)) != 2:
            raise ValueError(
                f"Task {task_name} does not contain both classes after excluding -1."
            )

        prediction_payload[f"{task_name}__y"] = y_task
        prediction_payload[f"{task_name}__original_indices"] = original_indices

        for area_name in all_areas:
            area_mask = brain_area_clean == area_name
            X_area = X_task[:, area_mask]

            result = run_loo_decoder_for_area(
                X_area,
                y_task,
                task_name=task_name,
                area_name=area_name,
                print_every=10,
            )

            summary_rows.append(
                {
                    "task": task_name,
                    "area": area_name,
                    "n_samples": result["n_samples"],
                    "n_features": result["n_features"],
                    "label_count_0": result["label_count_0"],
                    "label_count_1": result["label_count_1"],
                    "accuracy": result["accuracy"],
                    "balanced_accuracy": result["balanced_accuracy"],
                    "auc": result["auc"],
                    "tn": int(result["confusion_matrix"][0, 0]),
                    "fp": int(result["confusion_matrix"][0, 1]),
                    "fn": int(result["confusion_matrix"][1, 0]),
                    "tp": int(result["confusion_matrix"][1, 1]),
                    "elapsed_seconds": result["elapsed_seconds"],

                    # Boundary-ratio summaries
                    "mean_w_norm": result["mean_w_norm"],
                    "std_w_norm": result["std_w_norm"],
                    "mean_bias": result["mean_bias"],
                    "std_bias": result["std_bias"],
                    "mean_signed_boundary_distance": result["mean_signed_boundary_distance"],
                    "std_signed_boundary_distance": result["std_signed_boundary_distance"],
                    "median_signed_boundary_distance": result["median_signed_boundary_distance"],
                    "mean_abs_boundary_distance": result["mean_abs_boundary_distance"],
                    "std_abs_boundary_distance": result["std_abs_boundary_distance"],
                    "median_abs_boundary_distance": result["median_abs_boundary_distance"],

                    # Held-out geometric margin summaries
                    "mean_signed_margin": result["mean_signed_margin"],
                    "std_signed_margin": result["std_signed_margin"],
                    "mean_labeled_margin": result["mean_labeled_margin"],
                    "std_labeled_margin": result["std_labeled_margin"],
                    "min_labeled_margin": result["min_labeled_margin"],
                    "q05_labeled_margin": result["q05_labeled_margin"],
                    "q50_labeled_margin": result["q50_labeled_margin"],
                    "q95_labeled_margin": result["q95_labeled_margin"],
                }
            )

            prefix = f"{task_name}__{area_name}"

            # Original prediction payload
            prediction_payload[f"{prefix}__logits"] = result["logits"]
            prediction_payload[f"{prefix}__probs"] = result["probs"]
            prediction_payload[f"{prefix}__preds"] = result["preds"]
            prediction_payload[f"{prefix}__correct"] = result["correct"]
            prediction_payload[f"{prefix}__confusion_matrix"] = result["confusion_matrix"]

            # New per-fold geometry payload
            prediction_payload[f"{prefix}__w_norms"] = result["w_norms"]
            prediction_payload[f"{prefix}__biases"] = result["biases"]
            prediction_payload[f"{prefix}__signed_boundary_distances"] = result["signed_boundary_distances"]
            prediction_payload[f"{prefix}__abs_boundary_distances"] = result["abs_boundary_distances"]
            prediction_payload[f"{prefix}__signed_margins"] = result["signed_margins"]
            prediction_payload[f"{prefix}__labeled_margins"] = result["labeled_margins"]

            # Save after every area so partial progress survives.
            summary_df = pd.DataFrame(summary_rows)
            summary_df = summary_df.sort_values(
                ["task", "balanced_accuracy"],
                ascending=[True, False],
            )
            summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

            np.savez_compressed(
                OUT_PREDICTIONS_NPZ,
                **prediction_payload,
            )

            print()
            print(f"[SAVED] {OUT_SUMMARY_CSV}")
            print(f"[SAVED] {OUT_PREDICTIONS_NPZ}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["task", "balanced_accuracy"],
        ascending=[True, False],
    )
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

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
        "areas": all_areas,
        "angle_diagnostics": {
            "out_angle_csv": str(OUT_ANGLE_CSV),
            "out_angle_png": str(OUT_ANGLE_PNG),
            "out_axis_angle_png": str(OUT_AXIS_ANGLE_PNG),
            "contrast_angle_deg": "angle between animal and scene difference-of-means vectors",
            "contrast_axis_angle_deg": "same as contrast_angle_deg but ignores sign/opposite direction",
            "logistic_angle_deg": "angle between full-data Adam logistic weight vectors",
            "logistic_axis_angle_deg": "same as logistic_angle_deg but ignores sign/opposite direction",
        },
        "geometry_diagnostics": {
            "signed_boundary_distance": "-b / ||w||",
            "abs_boundary_distance": "|b| / ||w||",
            "signed_margin": "(w^T x_test + b) / ||w||",
            "labeled_margin": "y_signed * (w^T x_test + b) / ||w||",
            "note": (
                "Because StandardScaler is fit inside each LOO fold, x=0 is the "
                "training-fold mean neural response for that area and task."
            ),
        },
    }

    with open(OUT_CONFIG_JSON, "w") as f:
        json.dump(config, f, indent=2)

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    display_cols = [
        "task",
        "area",
        "n_samples",
        "n_features",
        "balanced_accuracy",
        "auc",
        "mean_abs_boundary_distance",
        "std_abs_boundary_distance",
        "mean_signed_boundary_distance",
        "mean_w_norm",
        "mean_bias",
        "mean_labeled_margin",
    ]

    print(summary_df[display_cols].to_string(index=False))

    print()
    print(f"[DONE] Summary saved to:     {OUT_SUMMARY_CSV}")
    print(f"[DONE] Predictions saved to: {OUT_PREDICTIONS_NPZ}")
    print(f"[DONE] Config saved to:      {OUT_CONFIG_JSON}")


if __name__ == "__main__":
    main()