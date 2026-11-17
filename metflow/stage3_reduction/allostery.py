"""Advanced allostery (DCCM, Cavity Sub-PCA, Loop RMSD) and Stage 3 orchestration."""

import os
from typing import List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

from metflow.cli import parse_res_string
from metflow.models import StageResult, TrajectoryContext
from metflow.stage1_ingestion.indexing import precompute_atom_indexer
from metflow.stage3_reduction.kabsch import export_merged_trajectory
from metflow.stage3_reduction.pca import compute_pca


def compute_dccm(ca_coords: np.ndarray) -> np.ndarray:
    """Compute Dynamic Cross-Correlation Matrix (N_ca x N_ca) from aligned C-alpha coords."""
    n_frames, n_ca, _ = ca_coords.shape
    mean_coords = np.mean(ca_coords, axis=0)
    fluct = ca_coords - mean_coords
    fluct_flat = fluct.transpose(1, 0, 2).reshape(n_ca, -1)
    cov = (fluct_flat @ fluct_flat.T) / max(1, n_frames)
    var = np.diag(cov)
    std = np.sqrt(np.maximum(var, 1e-12))
    return cov / np.outer(std, std)


def compute_cavity_subpca(
    cav_coords: np.ndarray,
    loop_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2-comp Sub-PCA on cavity coordinates and Loop RMSD to mean."""
    n_frames = cav_coords.shape[0]
    X_cav = cav_coords.reshape(n_frames, -1)
    n_comp = max(1, min(2, X_cav.shape[0], X_cav.shape[1]))
    sub_pca_model = PCA(n_components=n_comp)
    cav_pca_raw = sub_pca_model.fit_transform(X_cav)
    sub_var_raw = sub_pca_model.explained_variance_ratio_ * 100.0

    if n_comp < 2:
        cav_pca = np.zeros((n_frames, 2), dtype=cav_pca_raw.dtype)
        cav_pca[:, :n_comp] = cav_pca_raw
        sub_var_explained = np.zeros(2, dtype=np.float64)
        sub_var_explained[:n_comp] = sub_var_raw
    else:
        cav_pca = cav_pca_raw
        sub_var_explained = sub_var_raw

    mean_loop = np.mean(loop_coords, axis=0)
    loop_rmsd = np.sqrt(np.mean(np.sum((loop_coords - mean_loop) ** 2, axis=2), axis=1))
    return cav_pca, sub_var_explained, loop_rmsd, sub_pca_model.components_


def run_allostery(ctx: TrajectoryContext) -> List[str]:
    """Run DCCM and optional Cavity Sub-PCA + Loop RMSD analysis."""
    import MDAnalysis as mda

    config = ctx.config
    if not config.is_allostery:
        return []

    out_dir = ctx.out_dir
    valid_frames = ctx.valid_frames
    artifacts: List[str] = []

    print("\n--- Running Advanced Allostery Metrics ---")
    ctx.methodology.append("\n--- Advanced Allostery Metrics ---")

    print("Calculating Dynamic Cross-Correlation Matrix (DCCM)...")
    dccm = compute_dccm(ctx.coords_array)

    dccm_path = os.path.join(out_dir, "dccm_heatmap.png")
    plt.figure(figsize=(8, 7))
    sns.heatmap(dccm, cmap="coolwarm", center=0, xticklabels=20, yticklabels=20)
    plt.title("Dynamic Cross-Correlation Matrix (DCCM)")
    plt.xlabel("Residue Index")
    plt.ylabel("Residue Index")
    plt.savefig(dccm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved DCCM Heatmap to {dccm_path}")
    ctx.methodology.append("Generated Dynamic Cross-Correlation Matrix (DCCM).")
    artifacts.append(dccm_path)

    if config.cavity_res and config.loop_res:
        cav_res = parse_res_string(config.cavity_res)
        loop_res = parse_res_string(config.loop_res)
        cav_str = " ".join(str(x) for x in cav_res)

        cav_indexers = []
        loop_indexers = []
        for idx, u_ds in enumerate(ctx.universes):
            c_atoms = []
            for r_num in cav_res:
                r_idx = r_num - 1
                if 0 <= r_idx < len(ctx.ds_residues_homologous[idx]):
                    t_res = ctx.ds_residues_homologous[idx][r_idx]
                    ca_m = t_res.atoms.select_atoms("name CA Ca ca CA* C-alpha Calpha")
                    c_atoms.append(ca_m[0] if len(ca_m) > 0 else t_res.atoms[0])
                else:
                    fb = u_ds.select_atoms(f"resid {r_num} and name CA")
                    if len(fb) > 0:
                        c_atoms.append(fb[0])
            l_atoms = []
            for r_num in loop_res:
                r_idx = r_num - 1
                if 0 <= r_idx < len(ctx.ds_residues_homologous[idx]):
                    t_res = ctx.ds_residues_homologous[idx][r_idx]
                    ca_m = t_res.atoms.select_atoms("name CA Ca ca CA* C-alpha Calpha")
                    l_atoms.append(ca_m[0] if len(ca_m) > 0 else t_res.atoms[0])
                else:
                    fb = u_ds.select_atoms(f"resid {r_num} and name CA")
                    if len(fb) > 0:
                        l_atoms.append(fb[0])
            cav_indexers.append(precompute_atom_indexer(mda.AtomGroup(c_atoms) if c_atoms else u_ds.atoms[:0]))
            loop_indexers.append(precompute_atom_indexer(mda.AtomGroup(l_atoms) if l_atoms else u_ds.atoms[:0]))

        def _indexer_len(idxer) -> int:
            return max(0, idxer.stop - idxer.start) if isinstance(idxer, slice) else len(idxer)

        n_cav_atoms = _indexer_len(cav_indexers[0]) if cav_indexers else 0
        n_loop_atoms = _indexer_len(loop_indexers[0]) if loop_indexers else 0
        cav_coords = np.empty((len(valid_frames), n_cav_atoms, 3), dtype=np.float32)
        loop_coords = np.empty((len(valid_frames), n_loop_atoms, 3), dtype=np.float32)

        for idx, u_ds in enumerate(ctx.universes):
            ds_start = ctx.dataset_offsets[idx]
            ds_end = ctx.dataset_offsets[idx + 1]
            ds_valid_mask = (valid_frames >= ds_start) & (valid_frames < ds_end)
            if not np.any(ds_valid_mask):
                continue

            ds_valid_out_indices = np.where(ds_valid_mask)[0]
            ds_valid_local_frames = valid_frames[ds_valid_mask] - ds_start
            c_idxer = cav_indexers[idx]
            l_idxer = loop_indexers[idx]
            scale = ctx.dataset_scales[idx]
            ptr = 0
            n_needed = len(ds_valid_local_frames)

            for f_i, ts in enumerate(u_ds.trajectory):
                if ptr >= n_needed:
                    break
                if f_i == ds_valid_local_frames[ptr]:
                    out_i = ds_valid_out_indices[ptr]
                    if n_cav_atoms > 0:
                        cav_coords[out_i] = ts.positions[c_idxer]
                    if n_loop_atoms > 0:
                        loop_coords[out_i] = ts.positions[l_idxer]
                    ptr += 1

            if scale != 1.0:
                if n_cav_atoms > 0:
                    cav_coords[ds_valid_out_indices] *= scale
                if n_loop_atoms > 0:
                    loop_coords[ds_valid_out_indices] *= scale

        if cav_coords.shape[1] > 0 and loop_coords.shape[1] > 0:
            print("Calculating Targeted Sub-PCA for Allosteric Linkage...")
            cav_pca, sub_var_explained, loop_rmsd, sub_components = compute_cavity_subpca(cav_coords, loop_coords)

            subpca_plot_path = os.path.join(out_dir, "allostery_subpca_plot.png")
            plt.figure(figsize=(9, 7))
            scatter = plt.scatter(cav_pca[:, 0], cav_pca[:, 1], c=loop_rmsd, cmap="viridis", alpha=0.7, s=15)
            plt.colorbar(scatter, label="Loop RMSD to Mean (Å)")
            plt.title("Cavity PCA colored by Loop Fluctuation (Allostery Plot)")
            plt.xlabel(f"Cavity PC1 ({sub_var_explained[0]:.1f}%)")
            plt.ylabel(f"Cavity PC2 ({sub_var_explained[1]:.1f}%)")
            plt.savefig(subpca_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved Allostery Plot to {subpca_plot_path}")
            artifacts.append(subpca_plot_path)
            ctx.methodology.append(
                f"Targeted Sub-PCA applied to Cavity (res {cav_res}), colored by Loop RMSD (res {loop_res}). "
                f"Variance: PC1={sub_var_explained[0]:.1f}%, PC2={sub_var_explained[1]:.1f}%"
            )

            sub_n_atoms = cav_coords.shape[1]
            u_first, l_first = ctx.frame_universe_map[valid_frames[0]]
            u_first.trajectory[l_first]
            sub_resids = u_first.select_atoms(f"resid {cav_str} and name CA").resids
            if len(sub_resids) == sub_n_atoms:
                sub_atom_contrib_pc1 = np.sqrt(np.sum(sub_components[0].reshape(sub_n_atoms, 3) ** 2, axis=1))
                sub_atom_contrib_pc2 = (
                    np.sqrt(np.sum(sub_components[1].reshape(sub_n_atoms, 3) ** 2, axis=1))
                    if sub_components.shape[0] > 1
                    else np.zeros(sub_n_atoms)
                )

                sub_unique_resids, sub_res_inv = np.unique(sub_resids, return_inverse=True)
                sub_res_contrib_pc1 = np.bincount(sub_res_inv, weights=sub_atom_contrib_pc1)
                sub_res_contrib_pc2 = np.bincount(sub_res_inv, weights=sub_atom_contrib_pc2)

                loadings_path = os.path.join(out_dir, "allostery_subpca_loadings.png")
                plt.figure(figsize=(8, 4))
                width = 0.35
                x = np.arange(len(sub_unique_resids))
                plt.bar(x - width / 2, sub_res_contrib_pc1, width, label="PC1 Contribution", alpha=0.8, color="blue")
                plt.bar(x + width / 2, sub_res_contrib_pc2, width, label="PC2 Contribution", alpha=0.8, color="orange")
                plt.xticks(x, sub_unique_resids)
                plt.xlabel("Cavity Residue Number")
                plt.ylabel("Loading Magnitude (RMS)")
                plt.title("Per-Residue Contribution to Cavity Sub-PCA")
                plt.legend()
                plt.grid(True, alpha=0.3, axis="y")
                plt.savefig(loadings_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Saved Sub-PCA Loadings Plot to {loadings_path}")
                artifacts.append(loadings_path)

    return artifacts


def run_stage3(ctx: TrajectoryContext) -> StageResult:
    """Execute Stage 3: streaming Kabsch export, global/ref PCA, and allostery."""
    coords_array, rmsd_to_ref = export_merged_trajectory(ctx)
    ctx.coords_array = coords_array
    ctx.rmsd_to_ref = rmsd_to_ref

    X = coords_array.reshape(len(ctx.valid_frames), -1)
    ctx.X = X

    X_pca, pca_model, var_explained, method_note = compute_pca(
        X, ctx.dataset_tags, ctx.unique_tags, pca_ref=ctx.config.pca_ref, n_components=3
    )
    ctx.X_pca = X_pca
    ctx.pca_model = pca_model
    ctx.var_explained = var_explained
    if method_note:
        ctx.methodology.append(method_note)
    ctx.methodology.append(
        f"PCA Variance Explained: PC1={var_explained[0]:.1f}%, PC2={var_explained[1]:.1f}%, PC3={var_explained[2]:.1f}%"
    )

    ctx.df_data = {
        "Frame_Index": np.arange(len(ctx.valid_frames)),
        "Dataset": ctx.dataset_tags,
        "RMSD_to_Reference": rmsd_to_ref,
        "PCA_1": X_pca[:, 0],
        "PCA_2": X_pca[:, 1],
        "PCA_3": X_pca[:, 2],
    }

    artifacts = [ctx.merged_xtc_path, ctx.unified_xtc_path]
    artifacts.extend(run_allostery(ctx))

    return StageResult(
        stage_name="stage3_reduction",
        success=True,
        exit_code=0,
        artifacts=artifacts,
        metrics={
            "n_frames": int(len(ctx.valid_frames)),
            "pca_var_explained": [float(v) for v in var_explained],
        },
    )
