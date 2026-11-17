"""Global and reference-projected PCA computation."""

from typing import List, Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA


def compute_pca(
    X: np.ndarray,
    dataset_tags: List[str],
    unique_tags: List[str],
    pca_ref: str = "combined",
    n_components: int = 3,
) -> Tuple[np.ndarray, PCA, np.ndarray, Optional[str]]:
    """Compute 3-component PCA (with safe padding when n_samples < 3 or n_features < 3)."""
    n_samples, n_features = X.shape
    actual_comp = max(1, min(n_components, n_samples, n_features))
    pca_model = PCA(n_components=actual_comp)
    method_note: Optional[str] = None

    use_ref = bool(
        pca_ref
        and (
            pca_ref.lower() in ["first", "1", "ref", "reference"]
            or pca_ref.lower() != "combined"
        )
    )
    if use_ref and unique_tags:
        target_tag = unique_tags[0]
        if pca_ref.lower() not in ["first", "1", "ref", "reference"]:
            for t in unique_tags:
                if pca_ref.lower() in t.lower():
                    target_tag = t
                    break
        fit_mask = np.array([t == target_tag for t in dataset_tags])
        if np.sum(fit_mask) >= max(2, actual_comp):
            pca_model.fit(X[fit_mask])
            X_pca_raw = pca_model.transform(X)
            print(
                f"\n[Reference PCA Projection] Fitted PCA model strictly on reference dataset "
                f"'{target_tag}' ({np.sum(fit_mask)} frames) and projected all datasets into "
                f"'{target_tag}'s PCA coordinate space."
            )
            method_note = (
                f"Fitted PCA on reference dataset '{target_tag}' ({np.sum(fit_mask)} frames) "
                f"and projected all datasets into reference PCA space."
            )
        else:
            X_pca_raw = pca_model.fit_transform(X)
    else:
        X_pca_raw = pca_model.fit_transform(X)

    var_raw = pca_model.explained_variance_ratio_ * 100.0

    # Pad to n_components columns so downstream PCA_1..PCA_3 always exist even for n_frames == 2
    if actual_comp < n_components:
        X_pca = np.zeros((n_samples, n_components), dtype=X_pca_raw.dtype)
        X_pca[:, :actual_comp] = X_pca_raw
        var_explained = np.zeros(n_components, dtype=np.float64)
        var_explained[:actual_comp] = var_raw
    else:
        X_pca = X_pca_raw
        var_explained = var_raw

    return X_pca, pca_model, var_explained, method_note
