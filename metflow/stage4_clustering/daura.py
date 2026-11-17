"""Vectorized Daura / GROMOS greedy RMSD clustering."""

from typing import List, Tuple
import numpy as np


def run_daura(
    rmsd_matrix: np.ndarray,
    rmsd_cutoff: float,
    k: int,
) -> Tuple[np.ndarray, List[int]]:
    """Run vectorized Daura greedy clustering; k <= 0 allows unlimited clusters."""
    n_samples = rmsd_matrix.shape[0]
    if n_samples == 0:
        raise ValueError("Cannot run Daura clustering on 0 frames.")

    print(f"\n[ Daura ] Running Vectorized Daura Clustering with RMSD Cutoff {rmsd_cutoff} Å...")
    unclustered = np.ones(n_samples, dtype=bool)
    centroids: List[int] = []
    max_clusters = k if k > 0 else n_samples

    neighborhoods = rmsd_matrix < rmsd_cutoff
    while unclustered.any():
        counts = neighborhoods.dot(unclustered).astype(int)
        counts[~unclustered] = -1
        best_c = int(np.argmax(counts))
        best_h = neighborhoods[best_c] & unclustered
        centroids.append(best_c)
        unclustered[best_h] = False
        if len(centroids) >= max_clusters:
            break

    labels = np.argmin(rmsd_matrix[:, centroids], axis=1)
    return labels, centroids
