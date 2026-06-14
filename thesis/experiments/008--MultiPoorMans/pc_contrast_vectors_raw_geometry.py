#!/usr/bin/env python3
"""
Raw-geometry ViT-PC contrast vectors for all-neuron poor man's analysis.

Goal
----
1. Load stimulus-level neural responses X:

       X: stimuli x neurons

2. Load ViT logits Z:

       Z: stimuli x ImageNet-logit dimensions

3. Run PCA on ViT logits and take the first 10 PC score directions.

4. For each PC k, median-split stimuli into:

       low-PC-k stimuli
       high-PC-k stimuli

5. Compute the raw neural contrast vector:

       w_k = mean(X_high_k) - mean(X_low_k)

No row L2-normalization.
No class-template normalization.
No normalization of class means before subtraction.

The raw contrast vectors are saved directly. Normalization is used only when
computing cosine similarities between already-computed contrast vectors.

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
    vit_pc_raw_contrast_vectors/
        vit_pc_scores_and_labels.csv
        vit_pc_contrast_summary.csv
        vit_pc_contrast_vectors_raw.npy
        vit_pc_contrast_vectors_unit_for_cosine.npy
        vit_pc_contrast_cosine_matrix.csv
        vit_pc_contrast_raw_dot_matrix.csv
        vit_pc_alignment_with_animacy.csv
        vit_pc_contrast_norm_permutation_null.csv
        kept_neuron_indices.csv
        stimulus_presentation_counts.csv
        summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA


# =============================================================================
# Config
# =============================================================================

BASE_DIR = Path("/home/maria/Science/thesis/experiments/007--PoorMansClassifier")
DATA_DIR = Path("/home/maria/Science/data")

OUT_DIR = BASE_DIR / "vit_pc_raw_contrast_vectors"
OUT_DIR.mkdir(exist_ok=True, parents=True)

NEURAL_FILE = DATA_DIR / "hybrid_neural_responses_reduced.npy"
VIT_FILE = DATA_DIR / "google_vit-base-patch16-224_embeddings_logits.pkl"
VIT_KEY = "natural_scenes"

N_STIMULI = 118
N_VIT_PCS = 10

# Used only for the optional animacy-reference contrast.
ANIMATE_TOP1_THRESHOLD = 397

PRESENTATION_ORDER = "block"
STIMULUS_IDS_FILE = DATA_DIR / "stimulus_ids.npy"

RANDOM_SEED = 42
N_PERMUTATIONS = 1000

EPS = 1e-8

# Optional:
# False = preserve original neuron coordinate scale.
# True = z-score each neuron across stimuli before computing contrasts.
# This changes the neural-space metric, so keep False for raw geometry.
STANDARDIZE_NEURONS = False

# PCA(logits) in sklearn centers columns automatically.
# If True, additionally z-score each ImageNet logit dimension before PCA.
# Default False means PCs are of the raw centered logit geometry.
STANDARDIZE_LOGITS_BEFORE_PCA = False


# =============================================================================
# Loading
# =============================================================================

def load_neural_presentations() -> np.ndarray:
    """
    Load neural matrix and return shape:

        presentations x neurons

    Expected raw shape can be:

        neurons x presentations = (39209, 118)

    or:

        presentations x neurons = (118, 39209)
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
            "Expected something like (39209, 118) or (118, 39209)."
        )

    X_pres = X_pres.astype(np.float32, copy=False)
    print(f"[INFO] Presentation-level neural shape: {X_pres.shape}")

    return X_pres


def load_vit_natural_scenes_logits() -> np.ndarray:
    """
    Load ViT logits for natural scenes.

    Expected output shape:

        (118, 1000)
    """
    if not VIT_FILE.exists():
        raise FileNotFoundError(f"Missing ViT file: {VIT_FILE}")

    obj = np.load(VIT_FILE, allow_pickle=True)

    if hasattr(obj, "keys"):
        keys = list(obj.keys())

        if VIT_KEY not in keys:
            raise KeyError(
                f"Key {VIT_KEY!r} not found in {VIT_FILE}. "
                f"Available keys: {keys}"
            )

        logits = np.asarray(obj[VIT_KEY])

    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        item = obj.item()

        if not isinstance(item, dict):
            raise TypeError(f"Expected object array containing dict, got {type(item)}")

        if VIT_KEY not in item:
            raise KeyError(
                f"Key {VIT_KEY!r} not found in object dict. "
                f"Available keys: {list(item.keys())}"
            )

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
    Optional reference labels.

        1 = animate
        0 = inanimate

    Rule:

        top1 <= 397 => animate
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


def average_presentations_by_stimulus(
    X_pres: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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

    This is an unsupervised cleaning step.
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
# ViT PCA labels
# =============================================================================

def fit_vit_pca_and_make_labels(
    logits: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PCA]:
    """
    Fit PCA on ViT logits and median-split each PC score vector.

    Returns
    -------
    pc_scores:
        stimuli x n_components
    pc_labels:
        stimuli x n_components, where 1 means high-PC group
    pc_medians:
        n_components
    explained_variance_ratio:
        n_components
    pca:
        fitted sklearn PCA object
    """
    Z = logits

    if STANDARDIZE_LOGITS_BEFORE_PCA:
        print("[INFO] Z-scoring logit dimensions before PCA.")
        Z, logit_mean, logit_std = standardize_columns(Z)
        np.savez_compressed(
            OUT_DIR / "logit_standardization_stats.npz",
            logit_mean=logit_mean,
            logit_std=logit_std,
        )

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pc_scores = pca.fit_transform(Z).astype(np.float32, copy=False)

    pc_labels = np.zeros_like(pc_scores, dtype=int)
    pc_medians = np.zeros(n_components, dtype=np.float32)

    # Rank-based median split gives exactly balanced groups when n is even.
    for k in range(n_components):
        scores = pc_scores[:, k]
        order = np.argsort(scores)
        low_idx = order[: len(scores) // 2]
        high_idx = order[len(scores) // 2 :]

        labels = np.zeros(len(scores), dtype=int)
        labels[high_idx] = 1

        pc_labels[:, k] = labels
        pc_medians[k] = np.median(scores).astype(np.float32)

        print(
            f"[INFO] PC{k + 1}: low={int((labels == 0).sum())}, "
            f"high={int((labels == 1).sum())}, "
            f"EVR={pca.explained_variance_ratio_[k]:.6f}"
        )

    label_df = pd.DataFrame({"stimulus_index": np.arange(N_STIMULI)})

    for k in range(n_components):
        label_df[f"vit_pc{k + 1}_score"] = pc_scores[:, k]
        label_df[f"vit_pc{k + 1}_high_label"] = pc_labels[:, k]

    label_df.to_csv(OUT_DIR / "vit_pc_scores_and_labels.csv", index=False)

    np.save(OUT_DIR / "vit_pca_components.npy", pca.components_.astype(np.float32))
    np.save(OUT_DIR / "vit_pc_scores.npy", pc_scores)
    np.save(OUT_DIR / "vit_pc_labels.npy", pc_labels)

    return (
        pc_scores,
        pc_labels,
        pc_medians,
        pca.explained_variance_ratio_.astype(np.float32),
        pca,
    )


# =============================================================================
# Contrast vectors
# =============================================================================

def contrast_vector_for_binary_labels(
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    """
    Compute raw poor man's contrast:

        w = mean(X[y == 1]) - mean(X[y == 0])

    No normalization.
    """
    n_low = int(np.sum(y == 0))
    n_high = int(np.sum(y == 1))

    if n_low == 0 or n_high == 0:
        raise ValueError("Binary labels must contain both classes.")

    mu_low = X[y == 0].mean(axis=0)
    mu_high = X[y == 1].mean(axis=0)

    w = (mu_high - mu_low).astype(np.float32, copy=False)

    return {
        "w": w,
        "mu_low": mu_low.astype(np.float32, copy=False),
        "mu_high": mu_high.astype(np.float32, copy=False),
        "n_low": n_low,
        "n_high": n_high,
        "contrast_norm": float(np.linalg.norm(w)),
        "mean_low_norm": float(np.linalg.norm(mu_low)),
        "mean_high_norm": float(np.linalg.norm(mu_high)),
    }


def unit_rows(W: np.ndarray, eps: float = EPS) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize rows of W. Used only for direction comparison after raw contrasts exist.
    """
    norms = np.linalg.norm(W, axis=1)
    W_unit = W / np.maximum(norms[:, None], eps)

    return W_unit.astype(np.float32, copy=False), norms.astype(np.float32, copy=False)


def cosine_matrix_from_contrasts(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise cosine similarities between raw contrast vectors.
    """
    W_unit, norms = unit_rows(W)
    C = W_unit @ W_unit.T

    return C.astype(np.float32, copy=False), norms


def raw_dot_matrix_from_contrasts(W: np.ndarray) -> np.ndarray:
    """
    Raw dot products. This mixes direction alignment and contrast magnitude.
    """
    return (W @ W.T).astype(np.float32, copy=False)


def permutation_test_contrast_norms(
    X: np.ndarray,
    pc_labels: np.ndarray,
    n_permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    For each PC split, compare ||w_k|| to shuffled balanced labels.

    This tests whether that PC's high/low split has a stronger raw neural
    mean difference than an arbitrary balanced split.
    """
    rng = np.random.default_rng(random_seed)
    rows = []

    n_pcs = pc_labels.shape[1]

    observed_norms = []
    for k in range(n_pcs):
        observed_norms.append(
            float(contrast_vector_for_binary_labels(X, pc_labels[:, k])["contrast_norm"])
        )

    for k in range(n_pcs):
        y = pc_labels[:, k]
        observed = observed_norms[k]

        null = np.zeros(n_permutations, dtype=np.float32)

        for b in range(n_permutations):
            y_perm = rng.permutation(y)
            null[b] = float(contrast_vector_for_binary_labels(X, y_perm)["contrast_norm"])

        p_value = float((1 + np.sum(null >= observed)) / (n_permutations + 1))

        rows.append(
            {
                "pc": k + 1,
                "observed_contrast_norm": observed,
                "null_mean": float(null.mean()),
                "null_std": float(null.std(ddof=1)),
                "null_95pct": float(np.quantile(null, 0.95)),
                "null_99pct": float(np.quantile(null, 0.99)),
                "p_value_greater_equal": p_value,
            }
        )

        print(
            f"[PERM] PC{k + 1}: observed ||w||={observed:.6g}, "
            f"null={null.mean():.6g} ± {null.std(ddof=1):.6g}, "
            f"p={p_value:.6f}"
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
    print("Loading ViT logits")
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
    print("Running PCA on ViT logits and median-splitting first PCs")
    print("=" * 80)

    pc_scores, pc_labels, pc_medians, explained_variance_ratio, pca = (
        fit_vit_pca_and_make_labels(vit_logits, n_components=N_VIT_PCS)
    )

    print("=" * 80)
    print("Computing raw neural contrast vectors")
    print("=" * 80)

    contrast_vectors = np.zeros((N_VIT_PCS, X_clean.shape[1]), dtype=np.float32)
    mean_low_vectors = np.zeros_like(contrast_vectors)
    mean_high_vectors = np.zeros_like(contrast_vectors)

    summary_rows = []

    for k in range(N_VIT_PCS):
        result = contrast_vector_for_binary_labels(X_clean, pc_labels[:, k])

        w = result["w"]
        mu_low = result["mu_low"]
        mu_high = result["mu_high"]

        assert isinstance(w, np.ndarray)
        assert isinstance(mu_low, np.ndarray)
        assert isinstance(mu_high, np.ndarray)

        contrast_vectors[k] = w
        mean_low_vectors[k] = mu_low
        mean_high_vectors[k] = mu_high

        summary_rows.append(
            {
                "pc": k + 1,
                "explained_variance_ratio": float(explained_variance_ratio[k]),
                "pc_score_median": float(pc_medians[k]),
                "n_low": int(result["n_low"]),
                "n_high": int(result["n_high"]),
                "contrast_norm": float(result["contrast_norm"]),
                "mean_low_norm": float(result["mean_low_norm"]),
                "mean_high_norm": float(result["mean_high_norm"]),
            }
        )

        print(
            f"[CONTRAST] PC{k + 1}: ||w||={float(result['contrast_norm']):.6g}, "
            f"EVR={float(explained_variance_ratio[k]):.6f}"
        )

    contrast_summary_df = pd.DataFrame(summary_rows)
    contrast_summary_df.to_csv(OUT_DIR / "vit_pc_contrast_summary.csv", index=False)

    np.save(OUT_DIR / "vit_pc_contrast_vectors_raw.npy", contrast_vectors)
    np.save(OUT_DIR / "vit_pc_mean_low_vectors_raw.npy", mean_low_vectors)
    np.save(OUT_DIR / "vit_pc_mean_high_vectors_raw.npy", mean_high_vectors)

    print("=" * 80)
    print("Comparing contrast directions")
    print("=" * 80)

    cosine_matrix, contrast_norms = cosine_matrix_from_contrasts(contrast_vectors)
    raw_dot_matrix = raw_dot_matrix_from_contrasts(contrast_vectors)

    W_unit, _ = unit_rows(contrast_vectors)
    np.save(OUT_DIR / "vit_pc_contrast_vectors_unit_for_cosine.npy", W_unit)

    pc_names = [f"PC{k + 1}" for k in range(N_VIT_PCS)]

    pd.DataFrame(cosine_matrix, index=pc_names, columns=pc_names).to_csv(
        OUT_DIR / "vit_pc_contrast_cosine_matrix.csv"
    )

    pd.DataFrame(raw_dot_matrix, index=pc_names, columns=pc_names).to_csv(
        OUT_DIR / "vit_pc_contrast_raw_dot_matrix.csv"
    )

    print("[INFO] Contrast cosine matrix:")
    print(pd.DataFrame(cosine_matrix, index=pc_names, columns=pc_names).round(3))

    print("=" * 80)
    print("Comparing PC contrasts to animacy contrast")
    print("=" * 80)

    animacy_result = contrast_vector_for_binary_labels(X_clean, y_animacy)
    w_animacy = animacy_result["w"]
    assert isinstance(w_animacy, np.ndarray)

    w_animacy_unit, animacy_norm_arr = unit_rows(w_animacy[None, :])
    w_animacy_unit = w_animacy_unit[0]
    animacy_norm = float(animacy_norm_arr[0])

    align_rows = []

    for k in range(N_VIT_PCS):
        cos_to_animacy = float(W_unit[k] @ w_animacy_unit)
        corr_pc_score_animacy = float(np.corrcoef(pc_scores[:, k], y_animacy)[0, 1])

        align_rows.append(
            {
                "pc": k + 1,
                "cosine_to_animacy_contrast": cos_to_animacy,
                "absolute_cosine_to_animacy_contrast": abs(cos_to_animacy),
                "corr_pc_score_with_animacy_label": corr_pc_score_animacy,
                "pc_contrast_norm": float(contrast_norms[k]),
                "animacy_contrast_norm": animacy_norm,
                "explained_variance_ratio": float(explained_variance_ratio[k]),
            }
        )

    alignment_df = pd.DataFrame(align_rows)
    alignment_df.to_csv(OUT_DIR / "vit_pc_alignment_with_animacy.csv", index=False)

    print(alignment_df.round(4))

    print("=" * 80)
    print("Running permutation tests for raw contrast norms")
    print("=" * 80)

    perm_df = permutation_test_contrast_norms(
        X=X_clean,
        pc_labels=pc_labels,
        n_permutations=N_PERMUTATIONS,
        random_seed=RANDOM_SEED,
    )

    perm_df.to_csv(OUT_DIR / "vit_pc_contrast_norm_permutation_null.csv", index=False)

    summary = {
        "experiment": "vit_pc_raw_neural_contrast_vectors",
        "description": (
            "PCA is fitted to ViT logits. The first 10 PC score vectors are "
            "median-split into high/low stimulus groups. For each split, the "
            "raw all-neuron poor man's contrast vector is computed as "
            "mean(high) - mean(low). No population-vector L2 normalization, "
            "no template normalization, and no class-mean normalization are used. "
            "Unit-normalized contrast vectors are saved only for pairwise cosine "
            "direction comparisons after raw contrasts have been computed."
        ),
        "neural_file": str(NEURAL_FILE),
        "vit_file": str(VIT_FILE),
        "vit_key": VIT_KEY,
        "n_stimuli": int(N_STIMULI),
        "n_vit_pcs": int(N_VIT_PCS),
        "presentation_level_shape": list(X_pres.shape),
        "stimulus_averaged_shape": list(X_avg.shape),
        "clean_shape": list(X_clean.shape),
        "n_clean_neurons": int(X_clean.shape[1]),
        "standardize_neurons": bool(STANDARDIZE_NEURONS),
        "standardize_logits_before_pca": bool(STANDARDIZE_LOGITS_BEFORE_PCA),
        "row_l2_normalization": False,
        "class_template_l2_normalization": False,
        "class_mean_l2_normalization_before_subtraction": False,
        "contrast_formula": "w_k = mean(X[PC_k high]) - mean(X[PC_k low])",
        "median_split_method": "rank-based median split, exactly balanced because N_STIMULI is even",
        "presentation_order_assumption": PRESENTATION_ORDER,
        "used_explicit_stimulus_ids": bool(STIMULUS_IDS_FILE.exists()),
        "min_presentations_per_stimulus": int(presentation_counts.min()),
        "max_presentations_per_stimulus": int(presentation_counts.max()),
        "explained_variance_ratio_first_pcs": explained_variance_ratio.tolist(),
        "contrast_summary": contrast_summary_df.to_dict(orient="records"),
        "animacy_contrast_norm": animacy_norm,
        "n_permutations": int(N_PERMUTATIONS),
        "outputs": {
            "raw_contrast_vectors": str(OUT_DIR / "vit_pc_contrast_vectors_raw.npy"),
            "unit_contrast_vectors_for_cosine_only": str(
                OUT_DIR / "vit_pc_contrast_vectors_unit_for_cosine.npy"
            ),
            "contrast_summary": str(OUT_DIR / "vit_pc_contrast_summary.csv"),
            "cosine_matrix": str(OUT_DIR / "vit_pc_contrast_cosine_matrix.csv"),
            "raw_dot_matrix": str(OUT_DIR / "vit_pc_contrast_raw_dot_matrix.csv"),
            "alignment_with_animacy": str(OUT_DIR / "vit_pc_alignment_with_animacy.csv"),
            "permutation_norm_null": str(
                OUT_DIR / "vit_pc_contrast_norm_permutation_null.csv"
            ),
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
