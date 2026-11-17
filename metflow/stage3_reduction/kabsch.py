"""Batched Kabsch superposition and streaming unified trajectory export."""

import os
from typing import Tuple
import numpy as np

from metflow.models import TrajectoryContext
from metflow.stage1_ingestion.loader import make_alias


def batched_kabsch(
    P_raw: np.ndarray,
    Q_ref: np.ndarray,
    out: np.ndarray = None,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Superpose (N, A, 3) frames in P_raw onto centered (A, 3) reference Q_ref."""
    n_samples = len(P_raw)
    if n_samples == 0:
        return P_raw
    if out is None:
        out = np.empty_like(P_raw)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = P_raw[start:end]
        c_mean = chunk.mean(axis=1, keepdims=True)
        c_centered = chunk - c_mean
        H = np.matmul(c_centered.swapaxes(1, 2), Q_ref)
        U, _, Vt = np.linalg.svd(H)
        R = np.matmul(U, Vt)
        dets = (
            R[:, 0, 0] * (R[:, 1, 1] * R[:, 2, 2] - R[:, 1, 2] * R[:, 2, 1])
            - R[:, 0, 1] * (R[:, 1, 0] * R[:, 2, 2] - R[:, 1, 2] * R[:, 2, 0])
            + R[:, 0, 2] * (R[:, 1, 0] * R[:, 2, 1] - R[:, 1, 1] * R[:, 2, 0])
        )
        neg_mask = dets < 0
        if np.any(neg_mask):
            U[neg_mask, :, 2] *= -1
            R[neg_mask] = np.matmul(U[neg_mask], Vt[neg_mask])
        np.matmul(c_centered, R, out=out[start:end])
    return out


def export_merged_trajectory(
    ctx: TrajectoryContext,
    stream_chunk: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stream-align consensus heavy atoms, write merged_trajectory.xtc, and compute RMSD."""
    import MDAnalysis as mda

    out_dir = ctx.out_dir
    valid_frames = ctx.valid_frames
    u_consensus = ctx.u_consensus
    ref_consensus_mean = ctx.ref_consensus_mean
    pca_atom_indices = ctx.pca_atom_indices

    merged_xtc_path = os.path.join(out_dir, "merged_trajectory.xtc")
    unified_xtc_path = os.path.join(out_dir, "unified_trajectory.xtc")
    ctx.merged_xtc_path = merged_xtc_path
    ctx.unified_xtc_path = unified_xtc_path

    print("\n--- Exporting Physical Unified Trajectory ---")
    print(
        f"Writing {len(valid_frames)} aligned frames ({len(u_consensus.atoms)} atoms per frame) "
        f"to {merged_xtc_path}..."
    )

    coords_array = np.empty((len(valid_frames), len(pca_atom_indices), 3), dtype=np.float32)
    n_consensus_atoms = u_consensus.atoms.n_atoms
    c_batch_buf = np.empty((stream_chunk, n_consensus_atoms, 3), dtype=np.float32)
    dest_idx_buf = np.empty(stream_chunk, dtype=np.intp)

    with mda.Writer(merged_xtc_path, n_atoms=n_consensus_atoms) as w:
        def _flush_batch(count: int, scale: float) -> None:
            c_batch = c_batch_buf[:count]
            if scale != 1.0:
                c_batch *= scale
            batched_kabsch(c_batch, ref_consensus_mean, out=c_batch)
            for pos in c_batch:
                u_consensus.atoms.positions = pos
                w.write(u_consensus.atoms)
            coords_array[dest_idx_buf[:count]] = c_batch[:, pca_atom_indices, :]

        for ds_idx, (_, _, _, u_ds, _, _) in enumerate(ctx.temp_universes):
            ds_start = ctx.dataset_offsets[ds_idx]
            ds_end = ctx.dataset_offsets[ds_idx + 1]
            ds_valid_mask = (valid_frames >= ds_start) & (valid_frames < ds_end)
            if not np.any(ds_valid_mask):
                continue

            ds_valid_out_indices = np.where(ds_valid_mask)[0]
            ds_valid_local_frames = valid_frames[ds_valid_mask] - ds_start
            indexer = ctx.ds_consensus_indexers[ds_idx]
            scale = ctx.dataset_scales[ds_idx]

            buf_ptr = 0
            ptr = 0
            n_needed = len(ds_valid_local_frames)

            for f_i, ts in enumerate(u_ds.trajectory):
                if ptr >= n_needed:
                    break
                if f_i == ds_valid_local_frames[ptr]:
                    c_batch_buf[buf_ptr] = ts.positions[indexer]
                    dest_idx_buf[buf_ptr] = ds_valid_out_indices[ptr]
                    buf_ptr += 1
                    ptr += 1

                    if buf_ptr == stream_chunk:
                        _flush_batch(buf_ptr, scale)
                        buf_ptr = 0

            if buf_ptr > 0:
                _flush_batch(buf_ptr, scale)

    print(f"  -> Exported merged trajectory to {merged_xtc_path}")
    make_alias(merged_xtc_path, unified_xtc_path, out_dir)
    print(f"  -> Created unified trajectory alias at {unified_xtc_path}")

    diff_ref = coords_array - ctx.ref_bb
    rmsd_to_ref = np.sqrt(np.einsum("nai,nai->n", diff_ref, diff_ref) / coords_array.shape[1])
    return coords_array, rmsd_to_ref
