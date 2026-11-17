"""Static matplotlib/seaborn plots, t-SNE, RMSD heatmaps, and PCA loadings."""

import os
from typing import Any, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE


def _get_dataset_palette(n_ds: int):
    if n_ds <= 10:
        return sns.color_palette("tab10", max(1, n_ds))
    if n_ds <= 20:
        return sns.color_palette("tab20", n_ds)
    return sns.color_palette("husl", n_ds)


def generate_comparison_plots(
    df: pd.DataFrame,
    out_dir: str,
) -> Tuple[List[str], List[Any], Any]:
    """Generate PCA dataset comparison, energy landscape, and 1D KDE/hist plots."""
    artifacts: List[str] = []
    unique_datasets = list(pd.unique(df["Dataset"]))
    n_ds = len(unique_datasets)
    palette = _get_dataset_palette(n_ds)

    pc1_col_name = [c for c in df.columns if c.startswith("PCA_1")][0]
    pc2_col_name = [c for c in df.columns if c.startswith("PCA_2")][0]

    # 1. PCA Dataset Origin Comparison
    plt.figure(figsize=(10, 8))
    for i, ds in enumerate(unique_datasets):
        sub = df[df["Dataset"] == ds]
        plt.scatter(
            sub[pc1_col_name],
            sub[pc2_col_name],
            label=f"{ds} (n={len(sub)})",
            color=palette[i % len(palette)],
            alpha=0.6,
            s=15,
        )
    plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Multi-Trajectory PCA Projections by Origin Dataset")
    plt.xlabel(pc1_col_name)
    plt.ylabel(pc2_col_name)
    pca_cmp_path = os.path.join(out_dir, "pca_dataset_comparison.png")
    plt.savefig(pca_cmp_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA Dataset Comparison Plot to {pca_cmp_path}")
    artifacts.append(pca_cmp_path)

    # 2. Energy Landscape Plot
    if "OpenMM_Energy_kcal_mol" in df.columns:
        en = df["OpenMM_Energy_kcal_mol"].dropna()
        if len(en) > 0:
            plt.figure(figsize=(10, 8))
            vmax = np.percentile(en, 95)
            vmin = np.min(en)
            scatter = plt.scatter(
                df[pc1_col_name],
                df[pc2_col_name],
                c=df["OpenMM_Energy_kcal_mol"],
                cmap="coolwarm",
                alpha=0.7,
                s=15,
                vmin=vmin,
                vmax=vmax,
            )
            plt.colorbar(scatter, label="Energy (kcal/mol)")
            plt.title("Free Energy Landscape (PCA colored by OpenMM Energy)")
            plt.xlabel(pc1_col_name)
            plt.ylabel(pc2_col_name)
            en_plot_path = os.path.join(out_dir, "energy_landscape_plot.png")
            plt.savefig(en_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved Energy Landscape Plot to {en_plot_path}")
            artifacts.append(en_plot_path)

    # 3. PCA 1D Distributions partitioned by origin dataset
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for i, ds in enumerate(unique_datasets):
        sub = df[df["Dataset"] == ds]
        color = palette[i % len(palette)]
        lbl = f"{ds} (n={len(sub)})"
        if len(sub) > 1:
            try:
                sns.kdeplot(data=sub[pc1_col_name], ax=ax1, label=lbl, color=color, fill=True, alpha=0.25, warn_singular=False)
                sns.kdeplot(data=sub[pc2_col_name], ax=ax2, label=lbl, color=color, fill=True, alpha=0.25, warn_singular=False)
            except Exception:
                ax1.hist(sub[pc1_col_name], bins=30, density=True, alpha=0.4, color=color, label=lbl)
                ax2.hist(sub[pc2_col_name], bins=30, density=True, alpha=0.4, color=color, label=lbl)
        else:
            ax1.hist(sub[pc1_col_name], bins=30, density=True, alpha=0.4, color=color, label=lbl)
            ax2.hist(sub[pc2_col_name], bins=30, density=True, alpha=0.4, color=color, label=lbl)

    ax1.set_title("Conformational Density along PC1 by Dataset")
    ax1.set_xlabel(pc1_col_name)
    ax1.set_ylabel("Density")
    ax1.legend(title="Dataset")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Conformational Density along PC2 by Dataset")
    ax2.set_xlabel(pc2_col_name)
    ax2.set_ylabel("Density")
    ax2.legend(title="Dataset")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    dist_path = os.path.join(out_dir, "pca_1d_distributions.png")
    plt.savefig(dist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA 1D Distributions to {dist_path}")
    artifacts.append(dist_path)

    return artifacts, unique_datasets, palette


def generate_cluster_plots(
    df: pd.DataFrame,
    out_dir: str,
    var_explained: np.ndarray,
) -> List[str]:
    """Generate 2D PCA cluster scatter plots for each clustering method present."""
    artifacts: List[str] = []
    pc1_col = f"PCA_1 ({var_explained[0]:.1f}%)"
    pc2_col = f"PCA_2 ({var_explained[1]:.1f}%)"
    if pc1_col not in df.columns:
        pc1_col = [c for c in df.columns if c.startswith("PCA_1")][0]
    if pc2_col not in df.columns:
        pc2_col = [c for c in df.columns if c.startswith("PCA_2")][0]

    for cluster_col, title, out_name in [
        ("PCA_KMeans_Cluster", "K-Means Clusters", "pca_kmeans_plot.png"),
        ("Daura_Cluster", "Daura Clusters", "daura_plot.png"),
        ("CNN_Cluster", "CNN/DBSCAN Clusters", "cnn_plot.png"),
    ]:
        if cluster_col in df.columns:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(df[pc1_col], df[pc2_col], c=df[cluster_col], cmap="tab20", alpha=0.6, s=10)
            plt.colorbar(scatter, label=title)
            plt.title(f"PCA Projection ({title})")
            plt.xlabel(f"Principal Component 1 ({var_explained[0]:.1f}%)")
            plt.ylabel(f"Principal Component 2 ({var_explained[1]:.1f}%)")
            out_path = os.path.join(out_dir, out_name)
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved 2D Cluster Graph to {out_path}")
            artifacts.append(out_path)
    return artifacts


def generate_advanced_plots(
    df: pd.DataFrame,
    X: np.ndarray,
    rmsd_matrix: Optional[np.ndarray],
    out_dir: str,
    unique_datasets: List[str],
    palette: Any,
    methodology: List[str],
) -> List[str]:
    """Generate t-SNE projections (with safe perplexity) and RMSD cluster heatmaps."""
    artifacts: List[str] = []
    print("\n--- Generating Advanced Non-linear Projections & Heatmaps ---")
    methodology.append("\n--- Advanced Cluster Visualizations ---")

    n_samples = len(X)
    if n_samples >= 3:
        perplexity = min(30, max(1, (n_samples - 1) // 3))
        print(f"Calculating t-SNE projection (perplexity={perplexity}, this may take a minute)...")
        tsne_model = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
        X_tsne = tsne_model.fit_transform(X)
        df["tSNE_1"] = X_tsne[:, 0]
        df["tSNE_2"] = X_tsne[:, 1]

        plt.figure(figsize=(10, 8))
        for i, ds in enumerate(unique_datasets):
            sub = df[df["Dataset"] == ds]
            plt.scatter(
                sub["tSNE_1"],
                sub["tSNE_2"],
                label=f"{ds} (n={len(sub)})",
                color=palette[i % len(palette)],
                alpha=0.6,
                s=15,
            )
        plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title("Multi-Trajectory t-SNE Projections by Origin Dataset")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        tsne_cmp_path = os.path.join(out_dir, "tsne_dataset_comparison.png")
        plt.savefig(tsne_cmp_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved t-SNE Dataset Comparison Plot to {tsne_cmp_path}")
        artifacts.append(tsne_cmp_path)

        for cluster_col, title, out_name in [
            ("Daura_Cluster", "Daura Clusters", "tsne_daura_plot.png"),
            ("CNN_Cluster", "CNN/DBSCAN Clusters", "tsne_cnn_plot.png"),
        ]:
            if cluster_col in df.columns:
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(df["tSNE_1"], df["tSNE_2"], c=df[cluster_col], cmap="tab20", alpha=0.6, s=10)
                plt.colorbar(scatter, label=title)
                plt.title(f"t-SNE Projection ({title})")
                plt.xlabel("t-SNE Dimension 1")
                plt.ylabel("t-SNE Dimension 2")
                out_path = os.path.join(out_dir, out_name)
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Saved t-SNE Graph to {out_path}")
                artifacts.append(out_path)
                methodology.append(f"Generated t-SNE non-linear projection for {title}.")
    else:
        print(f"⚠️ Skipping t-SNE: need at least 3 samples (got {n_samples}).")

    if rmsd_matrix is not None:
        print("Generating Cluster RMSD Heatmaps...")
        for cluster_col, title, out_name in [
            ("Daura_Cluster", "Daura", "daura_rmsd_heatmap.png"),
            ("CNN_Cluster", "CNN/DBSCAN", "cnn_rmsd_heatmap.png"),
        ]:
            if cluster_col in df.columns:
                sorted_idx = np.argsort(df[cluster_col].values)
                step = max(1, len(sorted_idx) // 1000)
                downsampled_idx = sorted_idx[::step]
                sorted_dist = rmsd_matrix[np.ix_(downsampled_idx, downsampled_idx)]

                plt.figure(figsize=(9, 8))
                plt.imshow(sorted_dist, cmap="viridis_r", aspect="auto", interpolation="none")
                plt.colorbar(label="RMSD (Å)")
                plt.title(f"RMSD Pairwise Distance Matrix (Sorted by {title} Clusters)")
                plt.xlabel("Frames (Sorted by Cluster)")
                plt.ylabel("Frames (Sorted by Cluster)")
                out_path = os.path.join(out_dir, out_name)
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Saved RMSD Heatmap to {out_path}")
                artifacts.append(out_path)
                methodology.append(f"Generated RMSD pairwise distance block heatmap for {title} clusters.")

    return artifacts


def generate_pca_loadings(
    pca_model: Any,
    ref_bb_atoms: Any,
    out_dir: str,
) -> List[str]:
    """Save pca_loadings.csv and pca_loadings_plot.png."""
    n_atoms = len(ref_bb_atoms)
    resids = ref_bb_atoms.resids
    components = pca_model.components_

    atom_contrib_pc1 = np.sqrt(np.sum(components[0].reshape(n_atoms, 3) ** 2, axis=1))
    if components.shape[0] > 1:
        atom_contrib_pc2 = np.sqrt(np.sum(components[1].reshape(n_atoms, 3) ** 2, axis=1))
    else:
        atom_contrib_pc2 = np.zeros(n_atoms, dtype=np.float64)

    unique_resids, res_inv = np.unique(resids, return_inverse=True)
    res_contrib_pc1 = np.bincount(res_inv, weights=atom_contrib_pc1)
    res_contrib_pc2 = np.bincount(res_inv, weights=atom_contrib_pc2)

    all_resids = np.arange(unique_resids.min(), unique_resids.max() + 1)
    pc1_full = np.full(len(all_resids), np.nan)
    pc2_full = np.full(len(all_resids), np.nan)

    idx_arr = unique_resids - unique_resids.min()
    pc1_full[idx_arr] = res_contrib_pc1
    pc2_full[idx_arr] = res_contrib_pc2

    csv_path = os.path.join(out_dir, "pca_loadings.csv")
    pd.DataFrame({
        "Residue": all_resids,
        "PC1_Contribution": pc1_full,
        "PC2_Contribution": pc2_full,
    }).to_csv(csv_path, index=False)
    print(f"Saved PCA Loadings data to {csv_path}")

    plot_path = os.path.join(out_dir, "pca_loadings_plot.png")
    plt.figure(figsize=(12, 5))
    plt.plot(all_resids, pc1_full, label="PC1 Contribution", alpha=0.8, color="blue", marker=".", markersize=4)
    plt.plot(all_resids, pc2_full, label="PC2 Contribution", alpha=0.8, color="orange", marker=".", markersize=4)
    plt.xlabel("Residue Number")
    plt.ylabel("Loading Magnitude (RMS)")
    plt.title("Per-Residue Contribution to Principal Components")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA Loadings Plot to {plot_path}")

    return [csv_path, plot_path]
