#!/usr/bin/env python3
"""
Decode animacy from the COMMON direction between animate and inanimate means.

This is the control version of the poor man's decoder.

Main idea
---------
For each train/test split, compute class means on the TRAINING DATA ONLY:

    mu_anim = mean(X_train[y_train == 1])
    mu_inan = mean(X_train[y_train == 0])

Then define:

    difference axis:    w = mu_anim - mu_inan
    common raw axis:    m = (mu_anim + mu_inan) / 2

The common-axis decoder projects each stimulus response x onto m:

    score_common(x) = x @ m

Then it learns a 1D threshold on the training scores only and predicts the
held-out stimulus from its common-axis score.

Why this matters
----------------
If animacy can be decoded from the common direction, then the animacy result
may partly reflect global/shared visual drive or response magnitude, not only
a contrastive animate-vs-inanimate population pattern.

This script also optionally evaluates:
    - common_orthogonalized: common axis after removing the difference-axis component
    - difference_reference: standard poor man's contrast axis
    - norm_only: ||x|| as a pure global magnitude control

Expected files
--------------
/home/maria/Science/data/
    hybrid_neural_responses_reduced.npy
    google_vit-base-patch16-224_embeddings_logits.pkl

Optional:
    stimulus_ids.npy

Outputs
-------
/home/maria/Science/thesis/experiments/007--PoorMansClassifier/
    animacy_common_direction_decoder/
        loo_predictions.csv
        axis_summary_full_data.csv
        permutation_null.csv
        summary.json
        full_data_axes.npz
        kept_neuron_indices.csv
        stimulus_presentation_counts.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


# =============================================================================
# Config
# =============================================================================

BASE_DIR = Path("/home/maria/Science/thesis/experiments/007--PoorMansClassifier")
DATA_DIR = Path("/home/maria/Science/data")

OUT_DIR = BASE_DIR / "animacy_common_direction_decoder"
OUT_DIR.mkdir(exist_ok=True, parents=True)

NEURAL_FILE = DATA_DIR / "hybrid_neural_responses_reduced.npy"
VIT_FILE = DATA_DIR / "google_vit-base-patch16-224_embeddings_logits.pkl"
VIT_KEY = "natural_scenes"

N_STIMULI = 118
ANIMATE_TOP1_THRESHOLD = 397

PRESENTATION_ORDER = "block"
STIMULUS_IDS_FILE = DATA_DIR / "stimulus_ids.npy"

RANDOM_SEED = 42
N_PERMUTATIONS = 1000

EPS = 1e-8

# Keep this False if you want raw neural geometry.
STANDARDIZE_NEURONS = False

# Which scalar decoders to evaluate.
# The requested one is "common_raw".
DECODER_NAMES = [
    "common_raw",
    "common_orthogonalized",
    "difference_reference",
    "norm_only",
]


# =============================================================================
# Loading
# =============================================================================

def load_neural_presentations() -> np.ndarray:
    """
    Load neural matrix and return shape:

        presentations x neurons

    Supported raw orientations:

        neurons x presentations
        presentations x neurons
    """
    if not NEURAL_FILE.exists():
        raise FileNotFoundError(f"Missing neural file: {NEURAL_FILE}")

    X_raw = np.asarray(np.load(NEURAL_FILE, allow_pickle=True))

    print(f"[INFO] Raw neural shape: {X_raw.shape}")

    if X_raw.ndim != 2:
        raise ValueError(f"Expected 2D neural matrix, got {X_raw.shape}")

    n0, n1 = X_raw.shape

    if n0 > n1 and n1 % N_STIMULI == 0:
        print("[INFO] Interpreting raw neural matrix as neurons x presentations.")
        X_pres = X_raw.T
    elif n1 > n0 and n0 % N_STIMULI == 0:
        print("[INFO] Interpreting raw neural matrix as presentations x neurons.")
        X_pres = X_raw
    else:
        raise ValueError(
            f"Could not infer orientation from neural shape {X_raw.shape}. "
            "Expected something like (39209, 118), (39209, 5900), "
            "(118, 39209), or (5900, 39209)."
        )

    X_pres = X_pres.astype(np.float32, copy=False)
    print(f"[INFO] Presentation-level neural shape: {X_pres.shape}")

    return X_pres


def load_vit_natural_scenes_logits() -> np.ndarray:
    """
    Load ViT logits for natural scenes.

    Expected output shape:

        stimuli x ImageNet-logit dimensions = (118, 1000)
    """
    if not VIT_FILE.exists():
        raise FileNotFoundError(f"Missing ViT file: {VIT_FILE}")

    obj = np.load(VIT_FILE, allow_pickle=True)

    if hasattr(obj, "keys"):
        keys = list(obj.keys())
        if VIT_KEY not in keys:
            raise KeyError(f"Key {VIT_KEY!r} not found. Available keys: {keys}")
        logits = np.asarray(obj[VIT_KEY])

    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        item = obj.item()
        if not isinstance(item, dict):
            raise TypeError(f"Expected object array containing dict, got {type(item)}")
        if VIT_KEY not in item:
            raise KeyError(f"Key {VIT_KEY!r} not found. Available keys: {list(item.keys())}")
        logits = np.asarray(item[VIT_KEY])

    else:
        raise TypeError(f"Unsupported ViT object type: {type(obj)}")

    if logits.ndim != 2:
        raise ValueError(f"Expected 2D ViT logits, got {logits.shape}")

    if logits.shape[0] != N_STIMULI:
        raise ValueError(f"Expected {N_STIMULI} rows, got {logits.shape[0]}")

    print(f"[INFO] ViT logits shape: {logits.shape}")
    return logits.astype(np.float32, copy=False)


def make_animacy_labels_from_vit_logits(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    1 = animate
    0 = inanimate

    Rule:
        ImageNet top-1 index <= 397 => animate
    """
    top1 = np.argmax(logits, axis=1)
    y = (top1 <= ANIMATE_TOP1_THRESHOLD).astype(int)

    print("[INFO] Derived animate/inanimate labels from ViT top-1.")
    print(f"[INFO] Inanimate count: {int((y == 0).sum())}")
    print(f"[INFO] Animate count:   {int((y == 1).sum())}")

    return y, top1


# =============================================================================
# Presentation averaging
# =============================================================================

def make_presentation_stimulus_ids(n_presentations: int) -> np.ndarray:
    """
    Return vector of length n_presentations containing stimulus IDs 0..117.
    """
    if STIMULUS_IDS_FILE.exists():
        stim_ids = np.load(STIMULUS_IDS_FILE, allow_pickle=True).astype(int).ravel()

        if len(stim_ids) != n_presentations:
            raise ValueError(
                f"{STIMULUS_IDS_FILE} has length {len(stim_ids)}, "
                f"but neural data has {n_presentations} presentations."
            )

        if stim_ids.min() < 0 or stim_ids.max() >= N_STIMULI:
            raise ValueError(
                f"Stimulus IDs must be in [0, {N_STIMULI - 1}], "
                f"got min={stim_ids.min()}, max={stim_ids.max()}."
            )

        print(f"[INFO] Loaded explicit stimulus IDs from {STIMULUS_IDS_FILE}")
        return stim_ids

    if n_presentations % N_STIMULI != 0:
        raise ValueError(
            f"n_presentations={n_presentations} is not divisible by {N_STIMULI}."
        )

    repeats = n_presentations // N_STIMULI

    if PRESENTATION_ORDER == "block":
        stim_ids = np.repeat(np.arange(N_STIMULI), repeats)
    elif PRESENTATION_ORDER == "cycle":
        stim_ids = np.tile(np.arange(N_STIMULI), repeats)
    else:
        raise ValueError("PRESENTATION_ORDER must be either 'block' or 'cycle'.")

    print(
        f"[WARN] No explicit {STIMULUS_IDS_FILE.name} found. "
        f"Assuming PRESENTATION_ORDER={PRESENTATION_ORDER!r}. "
        f"Repeats per stimulus={repeats}."
    )

    return stim_ids.astype(int)


def average_presentations_by_stimulus(X_pres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Average presentation-level neural matrix to stimulus-level matrix.

    Input:
        X_pres: presentations x neurons

    Output:
        X_avg: stimuli x neurons
        counts: presentations per stimulus
    """
    n_presentations, n_neurons = X_pres.shape
    stim_ids = make_presentation_stimulus_ids(n_presentations)

    X_avg = np.zeros((N_STIMULI, n_neurons), dtype=np.float32)
    counts = np.zeros(N_STIMULI, dtype=int)

    for stim_id in range(N_STIMULI):
        mask = stim_ids == stim_id
        counts[stim_id] = int(mask.sum())

        if counts[stim_id] == 0:
            raise ValueError(f"Stimulus {stim_id} has zero presentations.")

        X_avg[stim_id] = X_pres[mask].mean(axis=0)

    print("[INFO] Averaged neural responses by stimulus.")
    print(f"[INFO] Stimulus-averaged neural shape: {X_avg.shape}")
    print(f"[INFO] Presentations per stimulus: min={counts.min()}, max={counts.max()}")

    pd.DataFrame(
        {
            "stimulus_index": np.arange(N_STIMULI),
            "n_presentations": counts,
        }
    ).to_csv(OUT_DIR / "stimulus_presentation_counts.csv", index=False)

    return X_avg, counts


# =============================================================================
# Cleaning and optional standardization
# =============================================================================

def clean_features_all_data(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove neurons that are non-finite anywhere or have zero variance across stimuli.

    This is unsupervised cleaning.
    It does not use labels.
    """
    finite = np.isfinite(X).all(axis=0)
    var = np.nanvar(X, axis=0)
    nonzero_var = var > 0

    keep = finite & nonzero_var
    kept_original_indices = np.where(keep)[0]

    removed = X.shape[1] - int(keep.sum())
    if removed:
        print(f"[WARN] Removing {removed} non-finite or zero-variance neurons.")

    X_clean = X[:, keep].astype(np.float32, copy=False)

    pd.DataFrame(
        {
            "clean_feature_index": np.arange(len(kept_original_indices)),
            "original_neuron_index": kept_original_indices,
        }
    ).to_csv(OUT_DIR / "kept_neuron_indices.csv", index=False)

    print(f"[INFO] Clean stimulus-level neural shape: {X_clean.shape}")

    return X_clean, kept_original_indices


def standardize_columns(X: np.ndarray, eps: float = EPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score columns.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std > eps, std, 1.0)

    X_z = ((X - mean) / std_safe).astype(np.float32, copy=False)
    return X_z, mean.astype(np.float32), std_safe.astype(np.float32)


# =============================================================================
# Axis construction
# =============================================================================

def compute_train_axes(X_train: np.ndarray, y_train: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute all axes from training data only.
    """
    if np.sum(y_train == 0) == 0 or np.sum(y_train == 1) == 0:
        raise ValueError("Training labels must contain both classes.")

    mu_inan = X_train[y_train == 0].mean(axis=0)
    mu_anim = X_train[y_train == 1].mean(axis=0)

    difference = (mu_anim - mu_inan).astype(np.float32, copy=False)
    common = ((mu_anim + mu_inan) / 2.0).astype(np.float32, copy=False)

    # Remove the component of common that lies along the difference axis.
    denom = float(difference @ difference)
    if denom > EPS:
        common_orth = common - (float(common @ difference) / denom) * difference
    else:
        common_orth = common.copy()

    return {
        "mu_inanimate": mu_inan.astype(np.float32, copy=False),
        "mu_animate": mu_anim.astype(np.float32, copy=False),
        "difference_reference": difference,
        "common_raw": common,
        "common_orthogonalized": common_orth.astype(np.float32, copy=False),
    }


def safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def cosine(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), eps)
    return float((a @ b) / denom)


# =============================================================================
# 1D threshold classifier
# =============================================================================

def fit_best_threshold_1d(
    scores: np.ndarray,
    y: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = balanced_accuracy_score,
) -> dict[str, float | int]:
    """
    Fit the best 1D threshold and polarity using training data only.

    Prediction rule:

        polarity = +1: predict 1 if score >= threshold else 0
        polarity = -1: predict 1 if score <= threshold else 0

    The threshold is selected to maximize the requested metric on training data.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(y, dtype=int).ravel()

    if len(np.unique(y)) != 2:
        raise ValueError("Need both classes to fit a binary threshold.")

    order = np.argsort(scores)
    s_sorted = scores[order]

    # Candidate thresholds include edges and midpoints.
    mids = (s_sorted[:-1] + s_sorted[1:]) / 2.0
    thresholds = np.concatenate(
        [
            np.array([s_sorted[0] - 1e-6]),
            mids,
            np.array([s_sorted[-1] + 1e-6]),
        ]
    )

    best = {
        "threshold": float(thresholds[0]),
        "polarity": 1,
        "train_metric": -np.inf,
    }

    for threshold in thresholds:
        for polarity in (1, -1):
            if polarity == 1:
                pred = (scores >= threshold).astype(int)
            else:
                pred = (scores <= threshold).astype(int)

            value = float(metric_fn(y, pred))

            if value > float(best["train_metric"]):
                best = {
                    "threshold": float(threshold),
                    "polarity": int(polarity),
                    "train_metric": value,
                }

    return best


def predict_threshold_1d(scores: np.ndarray, threshold: float, polarity: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).ravel()

    if polarity == 1:
        return (scores >= threshold).astype(int)

    if polarity == -1:
        return (scores <= threshold).astype(int)

    raise ValueError(f"polarity must be +1 or -1, got {polarity}")


def score_from_decoder_name(
    X: np.ndarray,
    axes: dict[str, np.ndarray],
    decoder_name: str,
) -> np.ndarray:
    """
    Compute scalar scores for one decoder.
    """
    if decoder_name == "norm_only":
        return np.linalg.norm(X, axis=1)

    if decoder_name not in axes:
        raise KeyError(f"Unknown decoder {decoder_name!r}")

    axis = axes[decoder_name]
    return X @ axis


# =============================================================================
# Cross-validation
# =============================================================================

def loo_decode(X: np.ndarray, y: np.ndarray, decoder_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leave-one-out decoding.

    For each held-out stimulus:
        1. compute axes on the other 117 stimuli
        2. project train and held-out point onto the selected axis
        3. fit threshold on train scores only
        4. predict held-out stimulus
    """
    rows = []
    n = X.shape[0]

    for test_idx in range(n):
        train_idx = np.array([i for i in range(n) if i != test_idx], dtype=int)

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[[test_idx]]
        y_test = int(y[test_idx])

        axes = compute_train_axes(X_train, y_train)

        # Diagnostics for this fold.
        common_raw = axes["common_raw"]
        diff = axes["difference_reference"]
        common_orth = axes["common_orthogonalized"]

        fold_axis_diagnostics = {
            "common_raw_norm": safe_norm(common_raw),
            "difference_norm": safe_norm(diff),
            "common_orthogonalized_norm": safe_norm(common_orth),
            "common_dot_difference": float(common_raw @ diff),
            "common_cosine_difference": cosine(common_raw, diff),
            "mean_animate_norm": safe_norm(axes["mu_animate"]),
            "mean_inanimate_norm": safe_norm(axes["mu_inanimate"]),
        }

        for decoder_name in decoder_names:
            train_scores = score_from_decoder_name(X_train, axes, decoder_name)
            test_score = score_from_decoder_name(X_test, axes, decoder_name)

            threshold_info = fit_best_threshold_1d(train_scores, y_train)
            y_pred = int(
                predict_threshold_1d(
                    test_score,
                    threshold=float(threshold_info["threshold"]),
                    polarity=int(threshold_info["polarity"]),
                )[0]
            )

            rows.append(
                {
                    "test_stimulus_index": int(test_idx),
                    "true_label": y_test,
                    "predicted_label": y_pred,
                    "correct": int(y_pred == y_test),
                    "decoder": decoder_name,
                    "test_score": float(test_score[0]),
                    "threshold": float(threshold_info["threshold"]),
                    "polarity": int(threshold_info["polarity"]),
                    "train_balanced_accuracy_at_threshold": float(threshold_info["train_metric"]),
                    **fold_axis_diagnostics,
                }
            )

    pred_df = pd.DataFrame(rows)

    metric_rows = []
    for decoder_name, g in pred_df.groupby("decoder"):
        y_true = g["true_label"].to_numpy(int)
        y_pred = g["predicted_label"].to_numpy(int)
        scores = g["test_score"].to_numpy(float)

        # AUC can flip sign depending on threshold polarity; report raw AUC and max(AUC, 1-AUC).
        try:
            auc = float(roc_auc_score(y_true, scores))
            auc_abs = max(auc, 1.0 - auc)
        except ValueError:
            auc = np.nan
            auc_abs = np.nan

        metric_rows.append(
            {
                "decoder": decoder_name,
                "loo_accuracy": float(accuracy_score(y_true, y_pred)),
                "loo_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "loo_roc_auc_raw_scores": auc,
                "loo_roc_auc_direction_free": auc_abs,
                "n_correct": int(np.sum(y_true == y_pred)),
                "n_total": int(len(y_true)),
                "mean_common_cosine_difference_across_folds": float(
                    g["common_cosine_difference"].mean()
                ),
                "mean_common_dot_difference_across_folds": float(
                    g["common_dot_difference"].mean()
                ),
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values("decoder").reset_index(drop=True)
    return pred_df, metrics_df


def permutation_test_loo(
    X: np.ndarray,
    y: np.ndarray,
    decoder_names: list[str],
    n_permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Shuffle labels and rerun the full LOO procedure.

    This is slower than a fixed-axis permutation because the axis is recomputed
    inside every fold for every shuffled label vector, which is the right null
    for this decoder.
    """
    rng = np.random.default_rng(random_seed)

    observed_pred, observed_metrics = loo_decode(X, y, decoder_names)

    rows = []
    observed_map = {
        row["decoder"]: row["loo_balanced_accuracy"]
        for row in observed_metrics.to_dict(orient="records")
    }

    for decoder_name in decoder_names:
        print(f"[PERM] Decoder: {decoder_name}")

        null_bal_acc = np.zeros(n_permutations, dtype=np.float32)
        null_acc = np.zeros(n_permutations, dtype=np.float32)

        for b in range(n_permutations):
            y_perm = rng.permutation(y)

            _, perm_metrics = loo_decode(X, y_perm, [decoder_name])
            null_bal_acc[b] = float(perm_metrics.loc[0, "loo_balanced_accuracy"])
            null_acc[b] = float(perm_metrics.loc[0, "loo_accuracy"])

            if (b + 1) % 100 == 0:
                print(f"    {b + 1}/{n_permutations}")

        observed = float(observed_map[decoder_name])
        p_value = float((1 + np.sum(null_bal_acc >= observed)) / (n_permutations + 1))

        rows.append(
            {
                "decoder": decoder_name,
                "observed_loo_balanced_accuracy": observed,
                "null_balanced_accuracy_mean": float(null_bal_acc.mean()),
                "null_balanced_accuracy_std": float(null_bal_acc.std(ddof=1)),
                "null_balanced_accuracy_95pct": float(np.quantile(null_bal_acc, 0.95)),
                "null_balanced_accuracy_99pct": float(np.quantile(null_bal_acc, 0.99)),
                "p_value_greater_equal": p_value,
                "n_permutations": int(n_permutations),
            }
        )

        print(
            f"[PERM] {decoder_name}: observed bal acc={observed:.4f}, "
            f"null={null_bal_acc.mean():.4f} ± {null_bal_acc.std(ddof=1):.4f}, "
            f"p={p_value:.6f}"
        )

    return pd.DataFrame(rows)


# =============================================================================
# Full-data axis summary
# =============================================================================

def save_full_data_axes_and_summary(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    axes = compute_train_axes(X, y)

    np.savez_compressed(
        OUT_DIR / "full_data_axes.npz",
        common_raw=axes["common_raw"],
        common_orthogonalized=axes["common_orthogonalized"],
        difference_reference=axes["difference_reference"],
        mu_animate=axes["mu_animate"],
        mu_inanimate=axes["mu_inanimate"],
    )

    common_raw = axes["common_raw"]
    common_orth = axes["common_orthogonalized"]
    difference = axes["difference_reference"]
    mu_anim = axes["mu_animate"]
    mu_inan = axes["mu_inanimate"]

    rows = [
        {
            "quantity": "common_raw",
            "norm": safe_norm(common_raw),
            "dot_with_difference": float(common_raw @ difference),
            "cosine_with_difference": cosine(common_raw, difference),
        },
        {
            "quantity": "common_orthogonalized",
            "norm": safe_norm(common_orth),
            "dot_with_difference": float(common_orth @ difference),
            "cosine_with_difference": cosine(common_orth, difference),
        },
        {
            "quantity": "difference_reference",
            "norm": safe_norm(difference),
            "dot_with_difference": float(difference @ difference),
            "cosine_with_difference": 1.0,
        },
        {
            "quantity": "mu_animate",
            "norm": safe_norm(mu_anim),
            "dot_with_difference": float(mu_anim @ difference),
            "cosine_with_difference": cosine(mu_anim, difference),
        },
        {
            "quantity": "mu_inanimate",
            "norm": safe_norm(mu_inan),
            "dot_with_difference": float(mu_inan @ difference),
            "cosine_with_difference": cosine(mu_inan, difference),
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "axis_summary_full_data.csv", index=False)
    return df


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    np.random.seed(RANDOM_SEED)

    print("=" * 80)
    print("Loading neural data")
    print("=" * 80)

    X_pres = load_neural_presentations()

    print("=" * 80)
    print("Averaging presentations by stimulus")
    print("=" * 80)

    X_avg, presentation_counts = average_presentations_by_stimulus(X_pres)

    print("=" * 80)
    print("Loading ViT logits and animacy labels")
    print("=" * 80)

    vit_logits = load_vit_natural_scenes_logits()
    y_animacy, top1 = make_animacy_labels_from_vit_logits(vit_logits)

    if X_avg.shape[0] != vit_logits.shape[0]:
        raise ValueError(
            f"Neural rows {X_avg.shape[0]} do not match ViT rows {vit_logits.shape[0]}."
        )

    print("=" * 80)
    print("Cleaning neural features")
    print("=" * 80)

    X_clean, kept_original_neuron_indices = clean_features_all_data(X_avg)

    if STANDARDIZE_NEURONS:
        print("=" * 80)
        print("Standardizing neurons across stimuli")
        print("=" * 80)
        X_clean, neuron_mean, neuron_std = standardize_columns(X_clean)
        np.savez_compressed(
            OUT_DIR / "neuron_standardization_stats.npz",
            neuron_mean=neuron_mean,
            neuron_std=neuron_std,
        )

    print("=" * 80)
    print("Saving full-data common/difference axis diagnostics")
    print("=" * 80)

    axis_summary_df = save_full_data_axes_and_summary(X_clean, y_animacy)
    print(axis_summary_df.round(6))

    print("=" * 80)
    print("Running leakage-safe LOO decoding")
    print("=" * 80)

    loo_pred_df, loo_metrics_df = loo_decode(X_clean, y_animacy, DECODER_NAMES)

    loo_pred_df.to_csv(OUT_DIR / "loo_predictions.csv", index=False)
    loo_metrics_df.to_csv(OUT_DIR / "loo_metrics.csv", index=False)

    print("[RESULT] LOO metrics:")
    print(loo_metrics_df.round(4))

    print("=" * 80)
    print("Running permutation null")
    print("=" * 80)

    perm_df = permutation_test_loo(
        X=X_clean,
        y=y_animacy,
        decoder_names=DECODER_NAMES,
        n_permutations=N_PERMUTATIONS,
        random_seed=RANDOM_SEED,
    )

    perm_df.to_csv(OUT_DIR / "permutation_null.csv", index=False)

    print("[RESULT] Permutation null:")
    print(perm_df.round(4))

    summary = {
        "experiment": "animacy_common_direction_decoder",
        "description": (
            "Leakage-safe leave-one-out decoding of animate/inanimate labels from "
            "the common direction m=(mu_animate+mu_inanimate)/2. Axes are recomputed "
            "inside each training fold. A 1D threshold and polarity are fitted on "
            "training projections only. The script also reports an orthogonalized "
            "common-axis control, the standard difference-axis reference, and a "
            "norm-only global magnitude control."
        ),
        "neural_file": str(NEURAL_FILE),
        "vit_file": str(VIT_FILE),
        "vit_key": VIT_KEY,
        "n_stimuli": int(N_STIMULI),
        "clean_shape": list(X_clean.shape),
        "n_clean_neurons": int(X_clean.shape[1]),
        "n_animate": int(np.sum(y_animacy == 1)),
        "n_inanimate": int(np.sum(y_animacy == 0)),
        "standardize_neurons": bool(STANDARDIZE_NEURONS),
        "common_axis_formula": "m = (mean(X_train[y=1]) + mean(X_train[y=0])) / 2",
        "difference_axis_formula": "w = mean(X_train[y=1]) - mean(X_train[y=0])",
        "common_orthogonalized_formula": "m_orth = m - ((m @ w) / (w @ w)) * w",
        "score_common_raw": "score = x @ m",
        "thresholding": "best 1D threshold and polarity selected on training scores only",
        "cv": "leave-one-out over stimuli",
        "decoders": DECODER_NAMES,
        "loo_metrics": loo_metrics_df.to_dict(orient="records"),
        "axis_summary_full_data": axis_summary_df.to_dict(orient="records"),
        "n_permutations": int(N_PERMUTATIONS),
        "presentation_order_assumption": PRESENTATION_ORDER,
        "used_explicit_stimulus_ids": bool(STIMULUS_IDS_FILE.exists()),
        "min_presentations_per_stimulus": int(presentation_counts.min()),
        "max_presentations_per_stimulus": int(presentation_counts.max()),
        "outputs": {
            "loo_predictions": str(OUT_DIR / "loo_predictions.csv"),
            "loo_metrics": str(OUT_DIR / "loo_metrics.csv"),
            "axis_summary_full_data": str(OUT_DIR / "axis_summary_full_data.csv"),
            "permutation_null": str(OUT_DIR / "permutation_null.csv"),
            "full_data_axes": str(OUT_DIR / "full_data_axes.npz"),
            "summary": str(OUT_DIR / "summary.json"),
        },
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("Full summary")
    print("=" * 80)
    print(json.dumps(summary, indent=2))

    print("=" * 80)
    print(f"Done. Results saved to: {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
