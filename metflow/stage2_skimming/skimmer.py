"""ConvexHull volume skimming, gate distance filtering, and PyMOL hull visualization."""

from multiprocessing import Pool
import os
from typing import List, Optional, Tuple
import numpy as np
from scipy.spatial import ConvexHull

from metflow.models import StageResult, TrajectoryContext
from metflow.stage1_ingestion.indexing import extract_group_coords, precompute_atom_indexer
from metflow.stage2_skimming.sampling import downsample_per_dataset
from metflow.stage6_visualization.reports import log_methodology


def _calc_vol(coords: np.ndarray) -> float:
    """Compute ConvexHull volume for a single frame's coordinates (0.0 on degenerate hull)."""
    try:
        return float(ConvexHull(coords).volume)
    except Exception:
        return 0.0


def compute_hull_volumes(skim_coords_all: np.ndarray, use_multiprocessing: bool = True) -> np.ndarray:
    """Compute ConvexHull volumes across frames with Windows-safe multiprocessing fallback."""
    if len(skim_coords_all) == 0:
        return np.array([], dtype=np.float64)
    if use_multiprocessing and len(skim_coords_all) >= 64:
        try:
            with Pool() as p:
                return np.array(p.map(_calc_vol, skim_coords_all), dtype=np.float64)
        except Exception:
            pass
    return np.array([_calc_vol(c) for c in skim_coords_all], dtype=np.float64)


def write_visualize_hull_py(
    out_dir: str,
    verts: np.ndarray,
    top_res: int,
    bot_res: int,
) -> str:
    """Write a PyMOL CGO visualization script for the convex hull and gate distance."""
    hull = ConvexHull(verts)
    cgo = ["from pymol.cgo import *", "from pymol import cmd", "obj = []"]
    for s in hull.simplices:
        v1, v2, v3 = verts[s[0]], verts[s[1]], verts[s[2]]
        cgo.append(
            f"obj.extend([BEGIN, TRIANGLES, COLOR, 0.4, 0.8, 0.8, "
            f"VERTEX, {v1[0]:.3f}, {v1[1]:.3f}, {v1[2]:.3f}, "
            f"VERTEX, {v2[0]:.3f}, {v2[1]:.3f}, {v2[2]:.3f}, "
            f"VERTEX, {v3[0]:.3f}, {v3[1]:.3f}, {v3[2]:.3f}, END])"
        )
    cgo.extend([
        "cmd.load_cgo(obj, 'internal_volume_mesh')",
        "cmd.set('cgo_transparency', 0.4, 'internal_volume_mesh')",
        f"cmd.distance('gate_distance', 'resid {top_res} and name CA', 'resid {bot_res} and name CA')",
    ])
    out_path = os.path.join(out_dir, "visualize_hull.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cgo) + "\n")
    return out_path


def skim_frames(ctx: TrajectoryContext) -> Tuple[np.ndarray, Optional[str]]:
    """Filter frames by ConvexHull volume > min_vol and gate distance > 8.0 A."""
    import MDAnalysis as mda

    config = ctx.config
    if not (config.is_skim and config.equator and config.top and config.bottom):
        print("Skipped skimming.")
        log_methodology(ctx, "\n--- Conformational Skimming ---\nNo skimming was applied.")
        return ctx.valid_frames, None

    eq_res = [int(x) for x in config.equator.replace(",", " ").split()]
    top_res = int(config.top.replace(",", ""))
    bot_res = int(config.bottom.replace(",", ""))
    all_skim_res = eq_res + [top_res, bot_res]
    res_str = " ".join(str(x) for x in all_skim_res)

    print("\n--- Starting Tandem Skimming ---")
    skim_coords_list = []
    top_idx_in_s = len(eq_res)
    bot_idx_in_s = len(eq_res) + 1

    for idx, u_ds in enumerate(ctx.universes):
        s_atom_list = []
        for r_num in all_skim_res:
            r_idx = r_num - 1
            if 0 <= r_idx < len(ctx.ds_residues_homologous[idx]):
                target_res = ctx.ds_residues_homologous[idx][r_idx]
                ca_matches = target_res.atoms.select_atoms("name CA Ca ca CA* C-alpha Calpha")
                s_atom_list.append(ca_matches[0] if len(ca_matches) > 0 else target_res.atoms[0])
            else:
                fallback_atoms = u_ds.select_atoms(f"resid {r_num} and name CA")
                if len(fallback_atoms) > 0:
                    s_atom_list.append(fallback_atoms[0])
                else:
                    raise ValueError(
                        f"Residue {r_num} CA not found in dataset '{ctx.temp_universes[idx][2]}'. Aborting skim."
                    )

        s_atoms = mda.AtomGroup(s_atom_list)
        s_coords = extract_group_coords(u_ds, precompute_atom_indexer(s_atoms))
        if ctx.dataset_scales[idx] != 1.0:
            s_coords *= ctx.dataset_scales[idx]
        skim_coords_list.append(s_coords)

    skim_coords_all = (
        skim_coords_list[0]
        if len(skim_coords_list) == 1
        else np.concatenate(skim_coords_list, axis=0)
    )
    heights = np.linalg.norm(
        skim_coords_all[:, top_idx_in_s, :] - skim_coords_all[:, bot_idx_in_s, :], axis=1
    )

    print("Computing volumes with multiprocessing...")
    vols = compute_hull_volumes(skim_coords_all, use_multiprocessing=True)

    passed_mask = (vols > config.min_vol) & (heights > 8.0)
    valid_frames = np.where(passed_mask)[0]
    print(f"Skimming Complete: {len(valid_frames)} / {ctx.n_frames_initial} passed.")
    log_methodology(
        ctx,
        f"\n--- Conformational Skimming ---\n"
        f"Tandem skim (ConvexHull Volume > {config.min_vol} Å³, Gate Dist > 8.0 Å; "
        f"equator={config.equator}, top={config.top}, bottom={config.bottom}). "
        f"Surviving: {len(valid_frames)} / {ctx.n_frames_initial} frames.",
    )

    hull_script: Optional[str] = None
    if len(valid_frames) > 0:
        u_first, l_first = ctx.frame_universe_map[valid_frames[0]]
        u_first.trajectory[l_first]
        skim_atoms_first = u_first.select_atoms(f"resid {res_str} and name CA")
        hull_script = write_visualize_hull_py(ctx.out_dir, skim_atoms_first.positions, top_res, bot_res)
        print(f"Generated PyMOL visualization script: {hull_script}")

    return valid_frames, hull_script


def run_stage2(ctx: TrajectoryContext) -> StageResult:
    """Execute Stage 2: conformational skimming and per-dataset downsampling."""
    artifacts: List[str] = []
    valid_frames, hull_script = skim_frames(ctx)
    if hull_script:
        artifacts.append(hull_script)

    if len(valid_frames) < 2:
        raise ValueError(
            f"Too few valid frames remaining ({len(valid_frames)}) to perform clustering (minimum 2 required)."
        )

    ctx.unique_tags = list(dict.fromkeys(ctx.dataset_tags))
    sampled_frames, filtered_tags, sampling_msg = downsample_per_dataset(
        valid_frames, ctx.dataset_tags, ctx.config.max_frames
    )
    ctx.valid_frames = sampled_frames
    ctx.dataset_tags = filtered_tags
    log_methodology(ctx, sampling_msg)

    return StageResult(
        stage_name="stage2_skimming",
        success=True,
        exit_code=0,
        artifacts=artifacts,
        metrics={"n_valid_frames": int(len(sampled_frames)), "n_datasets": len(ctx.unique_tags)},
    )
