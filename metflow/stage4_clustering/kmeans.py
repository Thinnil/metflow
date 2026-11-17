"""PCA + KMeans clustering with Silhouette auto-k and small-sample safety."""

from typing import Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_kmeans(X_pca: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """Run KMeans on PCA coordinates; auto-detect k if k <= 0 and clamp to n_samples."""
    n_samples = len(X_pca)
    if n_samples == 0:
        raise ValueError("Cannot run KMeans on 0 samples.")

    auto_det = False
    if k <= 0:
        print("\n[ PCA + KMeans ] Auto-detecting optimal 'k' using Silhouette Analysis...")
        auto_det = True
        if n_samples < 3:
            best_k = min(2, n_samples)
            print(f"  -> Too few samples ({n_samples}) for Silhouette; using k = {best_k}")
        else:
            best_score = -1.0
            best_k = 2
            max_test_k = min(15, n_samples - 1)
            for k_test in range(2, max_test_k + 1):
                labels_test = KMeans(n_clusters=k_test, random_state=42, n_init="auto").fit_predict(X_pca)
                if len(set(labels_test)) < 2:
                    continue
                score = silhouette_score(X_pca, labels_test)
                if score > best_score:
                    best_score = score
                    best_k = k_test
            print(f"  -> Optimal clusters detected: k = {best_k} (Silhouette Score: {best_score:.3f})")
    else:
        best_k = min(k, n_samples)
        if best_k != k:
            print(f"\n[ PCA + KMeans ] Clamped k={k} to n_samples={n_samples}.")
        else:
            print(f"\n[ PCA + KMeans ] Running with explicit k={best_k}...")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_pca)
    return labels, kmeans.cluster_centers_, best_k, auto_det
