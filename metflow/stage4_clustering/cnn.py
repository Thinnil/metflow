"""CNN (DBSCAN) density clustering and Stage 4 orchestration."""

import os
from typing import List
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from metflow.models import StageResult, TrajectoryContext
from metflow.stage4_clustering.daura import run_daura
from metflow.stage4_clustering.kmeans import run_kmeans
from metflow.stage4_clustering.pairwise import compute_pairwise_rmsd, make_centroid_extractor


def run_cnn(rmsd_matrix: np.ndarray, eps: float, min_samples: int = 3) -> np.ndarray:
    """Run DBSCAN density clustering on precomputed pairwise RMSD matrix."""
    print(f"\n[ CNN ] Running Optimized DBSCAN with eps={eps}...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1)
    return dbscan.fit_predict(rmsd_matrix)


def run_stage4(ctx: TrajectoryContext) -> StageResult:
    """Execute Stage 4: pairwise RMSD, requested clustering methods, and centroid export."""
    config = ctx.config
    if config.cluster == "none":
        return StageResult(
            stage_name="stage4_clustering",
            success=True,
            exit_code=0,
            metrics={"skipped": True},
        )

    out_dir = ctx.out_dir
    valid_frames = ctx.valid_frames
    X = ctx.X
    X_pca = ctx.X_pca
    dataset_tags = ctx.dataset_tags
    rmsd_to_ref = ctx.rmsd_to_ref
    artifacts: List[str] = []

    print("\n--- Starting Clustering Modules ---")
    ctx.methodology.append("\n--- Clustering Analysis ---")

    methods_to_run = ["pca_kmeans", "daura", "cnn"] if config.cluster == "all" else [config.cluster]
    allowed_methods = {"pca_kmeans", "daura", "cnn"}
    for m in methods_to_run:
        if m not in allowed_methods:
            raise ValueError(f"Unknown clustering method: {m}")

    if config.cluster in ["daura", "cnn", "all"]:
        print("Calculating pairwise distance matrix...")
        ctx.rmsd_matrix = compute_pairwise_rmsd(X, ctx.coords_array.shape[1])

    extract_c = make_centroid_extractor(
        ctx.merged_ref_path, ctx.merged_xtc_path, valid_frames, ctx.frame_universe_map
    )

    try:
        for m in methods_to_run:
            method_dir = os.path.join(out_dir, m)
            os.makedirs(method_dir, exist_ok=True)

            if m == "pca_kmeans":
                labels, centers, best_k, auto_det = run_kmeans(X_pca, config.k)
                ctx.df_data["PCA_KMeans_Cluster"] = labels
                for cid in range(best_k):
                    pts = X_pca[labels == cid]
                    if len(pts) > 0:
                        c_idx = int(np.where(labels == cid)[0][np.argmin(np.linalg.norm(pts - centers[cid], axis=1))])
                        artifacts.append(extract_c("pca_kmeans", c_idx, cid, method_dir))
                ctx.methodology.append(
                    f"PCA+KMeans (k={best_k}) used."
                    + (" Optimal 'k' detected via Silhouette Score." if auto_det else "")
                )
            elif m == "daura":
                labels, centroids = run_daura(ctx.rmsd_matrix, config.rmsd, config.k)
                ctx.df_data["Daura_Cluster"] = labels
                for cid, c_idx in enumerate(centroids):
                    artifacts.append(extract_c("daura", c_idx, cid, method_dir))
                ctx.methodology.append(
                    f"Daura greedy clustering (cutoff {config.rmsd} Å) extracted {len(centroids)} centroids."
                )
            elif m == "cnn":
                labels = run_cnn(ctx.rmsd_matrix, config.cnn_eps, min_samples=3)
                ctx.df_data["CNN_Cluster"] = labels
                for cid in set(labels):
                    if cid == -1:
                        continue
                    idxs = np.where(labels == cid)[0]
                    sums = ctx.rmsd_matrix[np.ix_(idxs, idxs)].sum(axis=1)
                    center = int(idxs[np.argmin(sums)])
                    artifacts.append(extract_c("cnn", center, int(cid), method_dir))
                ctx.methodology.append(
                    f"CNN/DBSCAN: Density-based clustering (eps={config.cnn_eps}) extracted "
                    f"{len(set(labels) - {-1})} clusters."
                )
            else:
                raise ValueError(f"Unknown clustering method: {m}")

            df_method = pd.DataFrame({
                "Frame_Index": np.arange(len(valid_frames)),
                "Dataset": dataset_tags,
                "RMSD_to_Ref": rmsd_to_ref,
                "PCA_1": X_pca[:, 0],
                "PCA_2": X_pca[:, 1],
                "PCA_3": X_pca[:, 2],
                "Cluster_ID": labels,
            })
            csv_path = os.path.join(method_dir, f"{m}_metrics.csv")
            df_method.to_csv(csv_path, index=False)
            artifacts.append(csv_path)
    finally:
        if hasattr(extract_c, "close"):
            try:
                extract_c.close()
            except Exception:
                pass

    return StageResult(
        stage_name="stage4_clustering",
        success=True,
        exit_code=0,
        artifacts=artifacts,
        metrics={"methods": methods_to_run},
    )
