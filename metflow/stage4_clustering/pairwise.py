"""Pairwise RMSD calculation and centroid PDB extraction."""

from typing import Any, Callable, List, Tuple
import numpy as np
from sklearn.metrics import pairwise_distances


def compute_pairwise_rmsd(X: np.ndarray, n_atoms: int) -> np.ndarray:
    """Compute pairwise RMSD matrix (N x N) from flattened aligned coordinates."""
    if n_atoms <= 0:
        raise ValueError("n_atoms must be > 0 to compute pairwise RMSD.")
    dists = pairwise_distances(X, metric="euclidean", n_jobs=-1)
    dists /= dists.dtype.type(np.sqrt(n_atoms))
    return dists


def make_centroid_extractor(
    merged_ref_path: str,
    merged_xtc_path: str,
    valid_frames: np.ndarray,
    frame_universe_map: List[Tuple[Any, int]],
) -> Callable[[str, int, int, str], str]:
    """Create a centroid PDB writer backed by the merged trajectory or source universes."""
    import MDAnalysis as mda

    try:
        u_centroid_src = mda.Universe(merged_ref_path, merged_xtc_path)

        def extract_c(pfx: str, c_idx: int, cid: int, out_dir: str) -> str:
            u_centroid_src.trajectory[c_idx]
            out_path = f"{out_dir}/{pfx}_centroid_{cid}.pdb"
            u_centroid_src.atoms.write(out_path)
            return out_path

        extract_c.close = u_centroid_src.trajectory.close  # type: ignore[attr-defined]
        return extract_c
    except Exception:

        def extract_c(pfx: str, c_idx: int, cid: int, out_dir: str) -> str:
            frame_idx = valid_frames[c_idx]
            u_curr, local_idx = frame_universe_map[frame_idx]
            u_curr.trajectory[local_idx]
            out_path = f"{out_dir}/{pfx}_centroid_{cid}.pdb"
            u_curr.atoms.write(out_path)
            return out_path

        extract_c.close = lambda: None  # type: ignore[attr-defined]
        return extract_c
