#!/usr/bin/env python3
"""
Animacy poor man's decoder with global-norm / gain controls.

Question
--------
Is animate-vs-inanimate decoding driven by a real population-pattern contrast,
or mostly by global response magnitude?

This script compares:

1. difference_raw
   Poor man's decoder on raw neural vectors.

2. norm_only
   Decoder using only ||x||_2 as a scalar feature.

3. difference_row_l2
   Poor man's decoder after row-normalizing every stimulus vector:
       x_i <- x_i / ||x_i||
   This removes global population response magnitude per stimulus.

4. difference_residualize_norm
   Poor man's decoder after regressing global norm out of every neuron:
       X[:, j] ~ a_j + b_j * ||x||
   and decoding from residuals.
   This removes linear neuron-wise dependence on total response magnitude.

5. difference_norm_matched
   Optional matched-subset control:
   choose animate/inanimate stimuli with overlapping norms and run LOO only
   inside this matched subset.

Outputs
-------
/home/maria/Science/thesis/experiments/007--PoorMansClassifier/
    animacy_norm_gain_controls/
        loo_metrics.csv
        fold_predictions.csv
        full_data_axis_diagnostics.csv
        matched_subset_stimuli.csv
        summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


# =============================================================================
# Config
# =============================================================================

BASE_DIR = Path("/home/maria/Science/thesis/experiments/007--PoorMansClassifier")
DATA_DIR = Path("/home/maria/Science/data")

OUT_DIR = BASE_DIR / "animacy_norm_gain_controls"
OUT_DIR.mkdir(exist_ok=True, parents=True)

NEURAL_FILE = DATA_DIR / "hybrid_neural_responses_reduced.npy"
VIT_FILE = DATA_DIR / "google_vit-base-patch16-224_embeddings_logits.pkl"
VIT_KEY = "natural_scenes"

N_STIMULI = 118
ANIMATE_TOP1_THRESHOLD = 397

PRESENTATION_ORDER = "block"
STIMULUS_IDS_FILE = DATA_DIR / "stimulus_ids.npy"

EPS = 1e-8
RANDOM_SEED = 42

# Set True only if you really want neuron-wise z-scoring before everything.
# I recommend False for this specific geometry/gain control.
STANDARDIZE_NEURONS = False

# Matched subset control.
# "overlap" keeps only stimuli whose norm lies in the overlap of animate and inanimate norm ranges.
# Then it greedily balances counts by trimming the larger class to the smaller class.
RUN_NORM_MATCHED_CONTROL = True


# =============================================================================
# Loading
# =============================================================================

def load_neural_presentations() -> np.ndarray:
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
            f"Could not infer orientation from neural shape {X_raw.shape}."
        )

    X_pres = X_pres.astype(np.float32, copy=False)
    print(f"[INFO] Presentation-level neural shape: {X_pres.shape}")
    return X_pres


def load_vit_natural_scenes_logits() -> np.ndarray:
    if not VIT_FILE.exists():
        raise FileNotFoundError(f"Missing ViT file: {VIT_FILE}")

    obj = np.load(VIT_FILE, allow_pickle=True)

    if hasattr(obj, "keys"):
        if VIT_KEY not in obj:
            raise KeyError(f"Key {VIT_KEY!r} not found in {VIT_FILE}")
        logits = np.asarray(obj[VIT_KEY])
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        item = obj.item()
        if not isinstance(item, dict):
            raise TypeError(f"Expected object-array dict, got {type(item)}")
        if VIT_KEY not in item:
            raise KeyError(f"Key {VIT_KEY!r} not found in object dict")
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

    return X_avg, counts


# =============================================================================
# Cleaning / transforms
# =============================================================================

def clean_features_all_data(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Unsupervised cleaning: remove non-finite or zero-variance neurons.
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
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std > eps, std, 1.0)
    X_z = ((X - mean) / std_safe).astype(np.float32, copy=False)
    return X_z, mean.astype(np.float32), std_safe.astype(np.float32)


def row_l2_normalize(X: np.ndarray, eps: float = EPS) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(X, axis=1)
    X_unit = X / np.maximum(norms[:, None], eps)
    return X_unit.astype(np.float32, copy=False), norms.astype(np.float32)


def residualize_against_scalar_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    z_train: np.ndarray,
    z_test: np.ndarray,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Regress scalar z out of every feature using training data only.

        X[:, j] = a_j + b_j z + residual

    Then apply the same fitted a_j, b_j to test data.

    This is leakage-safe.
    """
    z_train = z_train.astype(np.float64)
    z_test = z_test.astype(np.float64)

    z_mean = float(z_train.mean())
    z_centered = z_train - z_mean

    denom = float(np.dot(z_centered, z_centered))
    if denom < eps:
        # If z has no variation in train, residualization cannot do anything.
        X_train_centered = X_train - X_train.mean(axis=0, keepdims=True)
        X_test_centered = X_test - X_train.mean(axis=0, keepdims=True)
        return X_train_centered.astype(np.float32), X_test_centered.astype(np.float32)

    X_mean = X_train.mean(axis=0)
    X_train_centered = X_train - X_mean[None, :]

    beta = (z_centered[:, None] * X_train_centered).sum(axis=0) / denom
    alpha = X_mean - beta * z_mean

    X_train_resid = X_train - (alpha[None, :] + z_train[:, None] * beta[None, :])
    X_test_resid = X_test - (alpha[None, :] + z_test[:, None] * beta[None, :])

    return X_train_resid.astype(np.float32), X_test_resid.astype(np.float32)


# =============================================================================
# Poor man's decoder machinery
# =============================================================================

@dataclass
class AxisResult:
    axis: np.ndarray
    midpoint: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray


def fit_difference_axis(X_train: np.ndarray, y_train: np.ndarray) -> AxisResult:
    mu0 = X_train[y_train == 0].mean(axis=0)
    mu1 = X_train[y_train == 1].mean(axis=0)

    axis = mu1 - mu0
    midpoint = 0.5 * (mu1 + mu0)

    return AxisResult(
        axis=axis.astype(np.float32),
        midpoint=midpoint.astype(np.float32),
        mu0=mu0.astype(np.float32),
        mu1=mu1.astype(np.float32),
    )


def project_on_axis(X: np.ndarray, axis: np.ndarray, midpoint: np.ndarray | None = None) -> np.ndarray:
    if midpoint is None:
        return X @ axis
    return (X - midpoint[None, :]) @ axis


def best_threshold_from_train_scores(
    scores: np.ndarray,
    y: np.ndarray,
) -> tuple[float, int, float]:
    """
    Choose threshold and polarity on training data.

    Rule:
        pred = 1 if polarity * score >= threshold else 0

    Returns:
        threshold, polarity, train_balanced_accuracy
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)

    if len(np.unique(y)) != 2:
        raise ValueError("Training labels must contain both classes.")

    candidates = np.unique(scores)

    if len(candidates) == 1:
        thresholds = np.array([candidates[0]])
    else:
        mids = 0.5 * (candidates[:-1] + candidates[1:])
        thresholds = np.r_[candidates[0] - EPS, mids, candidates[-1] + EPS]

    best = {
        "threshold": float(thresholds[0]),
        "polarity": 1,
        "bal_acc": -np.inf,
    }

    for polarity in (1, -1):
        signed = polarity * scores

        for threshold in thresholds:
            pred = (signed >= threshold).astype(int)
            bal_acc = balanced_accuracy_score(y, pred)

            if bal_acc > best["bal_acc"]:
                best = {
                    "threshold": float(threshold),
                    "polarity": int(polarity),
                    "bal_acc": float(bal_acc),
                }

    return best["threshold"], best["polarity"], best["bal_acc"]


def signed_scores_direction_free(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    For AUC reporting only: orient scores so larger tends to mean class 1.
    Uses labels, so this is diagnostic, not a deployed classifier.
    """
    auc = roc_auc_score(y, scores)
    if auc < 0.5:
        return -scores
    return scores


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    scores = np.asarray(scores, dtype=float)

    out: dict[str, float | int] = {
        "loo_accuracy": float(accuracy_score(y_true, y_pred)),
        "loo_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n_correct": int((y_true == y_pred).sum()),
        "n_total": int(len(y_true)),
    }

    try:
        out["loo_roc_auc_raw_scores"] = float(roc_auc_score(y_true, scores))
        out["loo_roc_auc_direction_free"] = float(
            roc_auc_score(y_true, signed_scores_direction_free(scores, y_true))
        )
    except ValueError:
        out["loo_roc_auc_raw_scores"] = float("nan")
        out["loo_roc_auc_direction_free"] = float("nan")

    return out


# =============================================================================
# Norm matching
# =============================================================================

def make_norm_matched_subset(y: np.ndarray, norms: np.ndarray) -> np.ndarray:
    """
    Keep stimuli in the overlapping norm range, then greedily balance class counts.

    This is deliberately simple and transparent.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    norm0 = norms[y == 0]
    norm1 = norms[y == 1]

    lo = max(float(norm0.min()), float(norm1.min()))
    hi = min(float(norm0.max()), float(norm1.max()))

    in_overlap = (norms >= lo) & (norms <= hi)

    idx0 = np.where(in_overlap & (y == 0))[0]
    idx1 = np.where(in_overlap & (y == 1))[0]

    n = min(len(idx0), len(idx1))
    if n < 3:
        raise ValueError(
            f"Too few matched samples after norm overlap: n0={len(idx0)}, n1={len(idx1)}"
        )

    # Greedy-ish trim:
    # sort each class by norm and take evenly spaced samples so the retained class norm ranges are comparable.
    def evenly_spaced_take(indices: np.ndarray, n_take: int) -> np.ndarray:
        sorted_idx = indices[np.argsort(norms[indices])]
        if len(sorted_idx) == n_take:
            return sorted_idx
        positions = np.linspace(0, len(sorted_idx) - 1, n_take).round().astype(int)
        return sorted_idx[positions]

    idx0_keep = evenly_spaced_take(idx0, n)
    idx1_keep = evenly_spaced_take(idx1, n)

    matched = np.sort(np.r_[idx0_keep, idx1_keep])

    pd.DataFrame(
        {
            "stimulus_index": matched,
            "animacy_label": y[matched],
            "raw_norm": norms[matched],
        }
    ).to_csv(OUT_DIR / "matched_subset_stimuli.csv", index=False)

    print(
        f"[MATCH] Raw norm overlap: [{lo:.6g}, {hi:.6g}], "
        f"kept n0={n}, n1={n}, total={2*n}"
    )

    return matched


# =============================================================================
# LOO experiments
# =============================================================================

def run_loo_decoders(X_raw: np.ndarray, y: np.ndarray, subset: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all LOO decoders.

    For row-L2 and norm-residualized controls, all transform parameters are
    computed inside each training fold when needed.
    """
    if subset is None:
        indices = np.arange(len(y))
        subset_name = "all_stimuli"
    else:
        indices = np.asarray(subset, dtype=int)
        subset_name = "norm_matched_subset"

    rows_pred = []

    decoders = [
        "difference_raw",
        "norm_only",
        "difference_row_l2",
        "difference_residualize_norm",
    ]

    for test_i in indices:
        train_idx = indices[indices != test_i]
        test_idx = np.array([test_i])

        y_train = y[train_idx]
        y_test = y[test_idx]

        if len(np.unique(y_train)) != 2:
            raise ValueError(f"Training fold for test stimulus {test_i} lacks both classes.")

        X_train_raw = X_raw[train_idx]
        X_test_raw = X_raw[test_idx]

        raw_norm_train = np.linalg.norm(X_train_raw, axis=1)
        raw_norm_test = np.linalg.norm(X_test_raw, axis=1)

        # ---------------------------------------------------------------------
        # 1. Raw difference PMC
        # ---------------------------------------------------------------------
        axis_result = fit_difference_axis(X_train_raw, y_train)
        train_scores = project_on_axis(X_train_raw, axis_result.axis, axis_result.midpoint)
        test_score = project_on_axis(X_test_raw, axis_result.axis, axis_result.midpoint)

        threshold, polarity, train_bal_acc = best_threshold_from_train_scores(train_scores, y_train)
        pred = int((polarity * test_score[0]) >= threshold)

        rows_pred.append(
            {
                "subset": subset_name,
                "decoder": "difference_raw",
                "test_stimulus_index": int(test_i),
                "y_true": int(y_test[0]),
                "score": float(test_score[0]),
                "polarity": int(polarity),
                "threshold": float(threshold),
                "y_pred": pred,
                "train_balanced_accuracy": float(train_bal_acc),
                "raw_norm_test": float(raw_norm_test[0]),
            }
        )

        # ---------------------------------------------------------------------
        # 2. Norm-only scalar decoder
        # ---------------------------------------------------------------------
        train_scores = raw_norm_train
        test_score = raw_norm_test

        threshold, polarity, train_bal_acc = best_threshold_from_train_scores(train_scores, y_train)
        pred = int((polarity * test_score[0]) >= threshold)

        rows_pred.append(
            {
                "subset": subset_name,
                "decoder": "norm_only",
                "test_stimulus_index": int(test_i),
                "y_true": int(y_test[0]),
                "score": float(test_score[0]),
                "polarity": int(polarity),
                "threshold": float(threshold),
                "y_pred": pred,
                "train_balanced_accuracy": float(train_bal_acc),
                "raw_norm_test": float(raw_norm_test[0]),
            }
        )

        # ---------------------------------------------------------------------
        # 3. Row-L2 normalized difference PMC
        # ---------------------------------------------------------------------
        X_train_l2, _ = row_l2_normalize(X_train_raw)
        # Important: normalize test row by its own norm. This is okay; it uses no label.
        X_test_l2, _ = row_l2_normalize(X_test_raw)

        axis_result = fit_difference_axis(X_train_l2, y_train)
        train_scores = project_on_axis(X_train_l2, axis_result.axis, axis_result.midpoint)
        test_score = project_on_axis(X_test_l2, axis_result.axis, axis_result.midpoint)

        threshold, polarity, train_bal_acc = best_threshold_from_train_scores(train_scores, y_train)
        pred = int((polarity * test_score[0]) >= threshold)

        rows_pred.append(
            {
                "subset": subset_name,
                "decoder": "difference_row_l2",
                "test_stimulus_index": int(test_i),
                "y_true": int(y_test[0]),
                "score": float(test_score[0]),
                "polarity": int(polarity),
                "threshold": float(threshold),
                "y_pred": pred,
                "train_balanced_accuracy": float(train_bal_acc),
                "raw_norm_test": float(raw_norm_test[0]),
            }
        )

        # ---------------------------------------------------------------------
        # 4. Norm-residualized difference PMC
        # ---------------------------------------------------------------------
        X_train_resid, X_test_resid = residualize_against_scalar_train_test(
            X_train=X_train_raw,
            X_test=X_test_raw,
            z_train=raw_norm_train,
            z_test=raw_norm_test,
        )

        axis_result = fit_difference_axis(X_train_resid, y_train)
        train_scores = project_on_axis(X_train_resid, axis_result.axis, axis_result.midpoint)
        test_score = project_on_axis(X_test_resid, axis_result.axis, axis_result.midpoint)

        threshold, polarity, train_bal_acc = best_threshold_from_train_scores(train_scores, y_train)
        pred = int((polarity * test_score[0]) >= threshold)

        rows_pred.append(
            {
                "subset": subset_name,
                "decoder": "difference_residualize_norm",
                "test_stimulus_index": int(test_i),
                "y_true": int(y_test[0]),
                "score": float(test_score[0]),
                "polarity": int(polarity),
                "threshold": float(threshold),
                "y_pred": pred,
                "train_balanced_accuracy": float(train_bal_acc),
                "raw_norm_test": float(raw_norm_test[0]),
            }
        )

    pred_df = pd.DataFrame(rows_pred)

    metric_rows = []
    for decoder, g in pred_df.groupby("decoder", sort=False):
        metrics = evaluate_predictions(
            y_true=g["y_true"].values,
            y_pred=g["y_pred"].values,
            scores=g["score"].values,
        )

        metric_rows.append(
            {
                "subset": subset_name,
                "decoder": decoder,
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    return metrics_df, pred_df


def full_data_axis_diagnostics(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    mu0 = X[y == 0].mean(axis=0)
    mu1 = X[y == 1].mean(axis=0)

    common = 0.5 * (mu1 + mu0)
    difference = mu1 - mu0

    diff_norm_sq = float(np.dot(difference, difference))
    common_proj_on_diff = (float(np.dot(common, difference)) / max(diff_norm_sq, EPS)) * difference
    common_orth = common - common_proj_on_diff

    raw_norms = np.linalg.norm(X, axis=1)

    quantities = {
        "mu_inanimate": mu0,
        "mu_animate": mu1,
        "common_raw": common,
        "difference": difference,
        "common_orthogonalized_to_difference": common_orth,
    }

    rows = []

    for name, v in quantities.items():
        norm = float(np.linalg.norm(v))
        dot = float(np.dot(v, difference))
        cos = dot / max(norm * float(np.linalg.norm(difference)), EPS)

        rows.append(
            {
                "quantity": name,
                "norm": norm,
                "dot_with_difference": dot,
                "cosine_with_difference": cos,
            }
        )

    rows.append(
        {
            "quantity": "stimulus_raw_norm_mean_inanimate",
            "norm": float(raw_norms[y == 0].mean()),
            "dot_with_difference": np.nan,
            "cosine_with_difference": np.nan,
        }
    )

    rows.append(
        {
            "quantity": "stimulus_raw_norm_mean_animate",
            "norm": float(raw_norms[y == 1].mean()),
            "dot_with_difference": np.nan,
            "cosine_with_difference": np.nan,
        }
    )

    return pd.DataFrame(rows)


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
    print("Full-data diagnostics")
    print("=" * 80)
    diag_df = full_data_axis_diagnostics(X_clean, y_animacy)
    diag_df.to_csv(OUT_DIR / "full_data_axis_diagnostics.csv", index=False)
    print(diag_df.round(6))

    all_metrics = []
    all_preds = []

    print("=" * 80)
    print("Running LOO controls on all stimuli")
    print("=" * 80)
    metrics_df, pred_df = run_loo_decoders(X_clean, y_animacy, subset=None)
    all_metrics.append(metrics_df)
    all_preds.append(pred_df)

    print(metrics_df.round(4))

    matched_subset = None

    if RUN_NORM_MATCHED_CONTROL:
        print("=" * 80)
        print("Constructing norm-matched subset and rerunning LOO controls")
        print("=" * 80)

        raw_norms = np.linalg.norm(X_clean, axis=1)
        matched_subset = make_norm_matched_subset(y_animacy, raw_norms)

        metrics_matched_df, pred_matched_df = run_loo_decoders(
            X_clean,
            y_animacy,
            subset=matched_subset,
        )

        all_metrics.append(metrics_matched_df)
        all_preds.append(pred_matched_df)

        print(metrics_matched_df.round(4))

    final_metrics_df = pd.concat(all_metrics, ignore_index=True)
    final_pred_df = pd.concat(all_preds, ignore_index=True)

    final_metrics_df.to_csv(OUT_DIR / "loo_metrics.csv", index=False)
    final_pred_df.to_csv(OUT_DIR / "fold_predictions.csv", index=False)

    summary = {
        "experiment": "animacy_norm_gain_controls",
        "description": (
            "Tests whether animate-inanimate poor man's decoding is driven by global "
            "population response norm/gain. Compares raw difference decoder, norm-only scalar "
            "decoder, row-L2-normalized difference decoder, norm-residualized difference decoder, "
            "and optional norm-matched subset controls."
        ),
        "neural_file": str(NEURAL_FILE),
        "vit_file": str(VIT_FILE),
        "vit_key": VIT_KEY,
        "n_stimuli": int(N_STIMULI),
        "presentation_level_shape": list(X_pres.shape),
        "stimulus_averaged_shape": list(X_avg.shape),
        "clean_shape": list(X_clean.shape),
        "n_clean_neurons": int(X_clean.shape[1]),
        "standardize_neurons": bool(STANDARDIZE_NEURONS),
        "presentation_order_assumption": PRESENTATION_ORDER,
        "used_explicit_stimulus_ids": bool(STIMULUS_IDS_FILE.exists()),
        "min_presentations_per_stimulus": int(presentation_counts.min()),
        "max_presentations_per_stimulus": int(presentation_counts.max()),
        "animate_count": int((y_animacy == 1).sum()),
        "inanimate_count": int((y_animacy == 0).sum()),
        "run_norm_matched_control": bool(RUN_NORM_MATCHED_CONTROL),
        "matched_subset_size": None if matched_subset is None else int(len(matched_subset)),
        "outputs": {
            "loo_metrics": str(OUT_DIR / "loo_metrics.csv"),
            "fold_predictions": str(OUT_DIR / "fold_predictions.csv"),
            "full_data_axis_diagnostics": str(OUT_DIR / "full_data_axis_diagnostics.csv"),
            "matched_subset_stimuli": str(OUT_DIR / "matched_subset_stimuli.csv"),
            "kept_neuron_indices": str(OUT_DIR / "kept_neuron_indices.csv"),
        },
        "metrics": final_metrics_df.to_dict(orient="records"),
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("Final metrics")
    print("=" * 80)
    print(final_metrics_df.round(4))

    print("=" * 80)
    print(f"Done. Results saved to: {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
