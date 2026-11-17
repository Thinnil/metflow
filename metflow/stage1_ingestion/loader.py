"""Stage 1 loader: dataset resolution, universe initialization, and consensus PDB export."""

import datetime
import os
import shutil
from typing import List, Tuple

import numpy as np

from metflow.cli import parse_list_arg, parse_res_string, save_params
from metflow.models import PipelineConfig, StageResult, TrajectoryContext
from metflow.stage1_ingestion.alignment import (
    build_consensus_atom_groups,
    find_consensus_bounds,
    verify_sequence_homology,
)
from metflow.stage1_ingestion.indexing import (
    PROTEIN_RESNAMES,
    extract_ca_atoms,
    normalize_resname,
    precompute_atom_indexer,
    res_to_1letter,
)


def resolve_dataset_configs(config: PipelineConfig) -> List[Tuple[str, str, str]]:
    """Resolve CLI dataset arguments into a non-empty list of (pdb, xtc, label) tuples."""
    if config.dataset_configs:
        dataset_configs: List[Tuple[str, str, str]] = []
        for entry in config.dataset_configs.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            parts = [p.strip() for p in entry.split(",")]
            if len(parts) >= 2:
                p_pdb = parts[0]
                p_xtc = parts[1]
                p_lbl = (
                    parts[2]
                    if (len(parts) >= 3 and parts[2])
                    else os.path.splitext(os.path.basename(p_pdb))[0]
                )
                dataset_configs.append((p_pdb, p_xtc, p_lbl))
        if not dataset_configs:
            raise ValueError(
                "No valid dataset entries found in --dataset_configs. "
                "Expected 'pdb1,xtc1,label1;pdb2,xtc2,label2'."
            )
        return dataset_configs

    pdb_list = parse_list_arg(config.pdb)
    xtc_list = parse_list_arg(config.xtc)
    if config.multi_datasets.lower() == "true" or (len(pdb_list) > 1 and len(xtc_list) > 1):
        label_list = parse_list_arg(config.dataset_labels)
        if len(pdb_list) != len(xtc_list):
            raise ValueError(
                f"Number of PDB files ({len(pdb_list)}) does not match XTC files ({len(xtc_list)})."
            )
        if len(pdb_list) == 0:
            raise ValueError("Multi-dataset mode enabled but no PDB/XTC files were provided.")
        while len(label_list) < len(pdb_list):
            curr_pdb = pdb_list[len(label_list)]
            default_lbl = os.path.splitext(os.path.basename(curr_pdb))[0] or f"Dataset_{len(label_list) + 1}"
            label_list.append(default_lbl)
        return list(zip(pdb_list, xtc_list, label_list))

    if config.pdb or config.xtc:
        pdb_path = pdb_list[0] if pdb_list else "./bioemu_ensemble/topology.pdb"
        xtc_path = xtc_list[0] if xtc_list else "./bioemu_ensemble/samples.xtc"
        if not os.path.exists(pdb_path) and os.path.exists("topology.pdb"):
            pdb_path = "topology.pdb"
        if not os.path.exists(xtc_path) and os.path.exists("samples.xtc"):
            xtc_path = "samples.xtc"
    else:
        xtc_path = "./bioemu_ensemble/samples.xtc"
        pdb_path = "./bioemu_ensemble/topology.pdb"
        if not os.path.exists(xtc_path) and os.path.exists("samples.xtc"):
            xtc_path = "samples.xtc"
            pdb_path = "topology.pdb"

    default_single_lbl = os.path.splitext(os.path.basename(pdb_path))[0] if pdb_path else "BioEmu_Ensemble"
    if default_single_lbl in ["topology", ""]:
        default_single_lbl = "BioEmu_Ensemble"
    label_list = parse_list_arg(config.dataset_labels)
    ds_label = label_list[0] if label_list else default_single_lbl
    return [(pdb_path, xtc_path, ds_label)]


def make_alias(src: str, dst: str, out_dir: str) -> None:
    """Create a symlink (or copy fallback) inside out_dir."""
    dst_abs = os.path.abspath(dst)
    out_dir_abs = os.path.abspath(out_dir)
    if os.path.abspath(src) == dst_abs:
        return
    try:
        if os.path.commonpath([out_dir_abs, dst_abs]) != out_dir_abs:
            return
    except ValueError:
        return
    if os.path.exists(dst) or os.path.islink(dst):
        try:
            os.remove(dst)
        except Exception:
            pass
    try:
        os.symlink(os.path.basename(src), dst)
    except Exception:
        shutil.copyfile(src, dst)


def run_stage1(ctx: TrajectoryContext) -> StageResult:
    """Execute Stage 1: dataset loading, homology verification, and consensus reference export."""
    import MDAnalysis as mda

    config = ctx.config
    dataset_configs = resolve_dataset_configs(config)
    if not dataset_configs:
        raise ValueError("No datasets configured for analysis.")

    ctx.dataset_configs = dataset_configs
    ctx.master_ref_pdb = dataset_configs[0][0]
    if not os.path.exists(ctx.master_ref_pdb):
        raise FileNotFoundError(f"Could not find master reference topology PDB: {ctx.master_ref_pdb}")

    out_dir = config.out_dir
    ctx.out_dir = out_dir
    os.makedirs(out_dir, exist_ok=True)
    param_path = save_params(config, out_dir, base_dir=config.base_dir)

    ctx.methodology = [f"=== MET Pipeline Methodology ===\nDate: {datetime.datetime.now()}"]
    if config.gen_params:
        ctx.methodology.append(f"Generation Parameters:\n{config.gen_params}")

    temp_universes = []
    prot_res_str = " ".join(PROTEIN_RESNAMES)
    for pdb_p, xtc_p, label in dataset_configs:
        if not os.path.exists(pdb_p):
            raise FileNotFoundError(f"PDB file not found: {pdb_p}")
        if not os.path.exists(xtc_p):
            raise FileNotFoundError(f"XTC file not found: {xtc_p}")
        u_temp = mda.Universe(pdb_p, xtc_p)
        protein_atoms = u_temp.select_atoms(
            f"resname {prot_res_str} and not (name H* or name [0-9]H* or type H or element H)"
        )
        if len(protein_atoms) == 0:
            raise ValueError(f"No protein residues matching PROTEIN_RESNAMES found in {pdb_p}.")
        ca_group = extract_ca_atoms(protein_atoms)
        prot_res = protein_atoms.residues
        temp_universes.append((pdb_p, xtc_p, label, u_temp, ca_group, prot_res))

    ctx.temp_universes = temp_universes
    if config.is_only_energy:
        print("\n--- Option [C]: Fast Energy ONLY Calculation Mode (Consensus-Aligned) ---")

    labels = [t[2] for t in temp_universes]
    dataset_seqs = ["".join(res_to_1letter(rn) for rn in t[5].resnames) for t in temp_universes]
    resname_seqs = [tuple(normalize_resname(rn) for rn in t[5].resnames) for t in temp_universes]
    ctx.dataset_seqs = dataset_seqs

    pair_ident_cache = verify_sequence_homology(dataset_seqs, labels, threshold=0.90)
    dataset_matches, consensus_ref_start, _, consensus_len, seq_summary_text = find_consensus_bounds(
        resname_seqs, labels, dataset_seqs, pair_ident_cache, min_consensus_len=10
    )
    ctx.dataset_matches = dataset_matches
    print(seq_summary_text)
    ctx.methodology.append(seq_summary_text)

    ds_residues_homologous, ds_consensus_groups = build_consensus_atom_groups(
        temp_universes, dataset_matches, consensus_ref_start, consensus_len
    )
    ctx.ds_residues_homologous = ds_residues_homologous
    ctx.ds_consensus_groups = ds_consensus_groups

    ref_atoms = ds_consensus_groups[0]
    u_consensus = mda.Merge(ref_atoms)
    for i, res in enumerate(u_consensus.residues, start=1):
        res.resid = i
    for i, atom in enumerate(u_consensus.atoms, start=1):
        atom.id = i

    ref_consensus_pos = ref_atoms.positions.copy()
    ref_consensus_mean = ref_consensus_pos - ref_consensus_pos.mean(axis=0)
    ref_rg = float(np.mean(np.sqrt(np.sum(ref_consensus_mean ** 2, axis=1))))
    if 0 < ref_rg < 5.0:
        print(
            f"  -> Auto-detected nanometer (nm) coordinate scale for reference (Rg={ref_rg:.2f} nm). "
            "Auto-scaling reference to Ångströms (x10)..."
        )
        ref_consensus_mean *= 10.0

    u_consensus.atoms.positions = ref_consensus_mean
    ctx.u_consensus = u_consensus
    ctx.ref_consensus_mean = ref_consensus_mean

    merged_ref_path = os.path.join(out_dir, "merged_reference.pdb")
    unified_ref_path = os.path.join(out_dir, "unified_reference.pdb")
    u_consensus.atoms.write(merged_ref_path)
    print(f"  -> Exported consensus reference PDB to {merged_ref_path}")
    make_alias(merged_ref_path, unified_ref_path, out_dir)
    print(f"  -> Created unified reference alias at {unified_ref_path}")
    ctx.merged_ref_path = merged_ref_path
    ctx.unified_ref_path = unified_ref_path

    ca_mask = np.array([a.name.strip().upper() == "CA" for a in u_consensus.atoms])
    ca_indices = np.where(ca_mask)[0]
    if config.pca_res:
        pca_res_list = set(parse_res_string(config.pca_res))
        rel_indices = [
            i
            for i in range(len(ca_indices))
            if (i + 1) in pca_res_list or u_consensus.residues[i].resid in pca_res_list
        ]
        if not rel_indices:
            rel_indices = list(range(min(len(pca_res_list), len(ca_indices))))
        pca_atom_indices = ca_indices[rel_indices]
        ctx.methodology.append(
            f"Alignment and Global PCA restricted to {len(rel_indices)} C-alpha residues from: {config.pca_res}"
        )
    else:
        pca_atom_indices = ca_indices
        ctx.methodology.append(f"Alignment and Global PCA used {len(ca_indices)} common C-alpha backbone atoms.")

    ctx.pca_atom_indices = pca_atom_indices
    ctx.ref_bb_atoms = u_consensus.atoms[pca_atom_indices]
    ctx.ref_bb = ref_consensus_mean[pca_atom_indices]

    offset_info = []
    for idx, (_, _, label, _, _, _) in enumerate(temp_universes):
        ref_start, _, ds_start, _ = dataset_matches[idx]
        offset = ref_start - ds_start
        if offset != 0:
            offset_info.append(f"{label} offset {offset:+} detected and aligned")

    offset_str = f" ({', '.join(offset_info)})" if offset_info else ""
    print(f"Aligning frames across {len(dataset_configs)} dataset(s) in memory...")
    print(
        f"  -> Dynamic sequence alignment matched {consensus_len} residues "
        f"({len(u_consensus.atoms)} heavy atoms) across datasets{offset_str}."
    )

    total_frames = sum(len(t[3].trajectory) for t in temp_universes)
    ctx.ds_consensus_indexers = [precompute_atom_indexer(ag) for ag in ds_consensus_groups]
    ctx.universes = []
    ctx.dataset_scales = []
    ctx.dataset_offsets = []
    ctx.dataset_tags = []
    ctx.frame_universe_map = []

    print(f"\n--- Loading and Aligning {len(dataset_configs)} Trajectory Dataset(s) ---")
    curr_frame_offset = 0
    for idx, (pdb_p, xtc_p, label, u_ds, _, _) in enumerate(temp_universes):
        print(f"Loading dataset '{label}' (PDB: {pdb_p}, XTC: {xtc_p})...")
        ctx.universes.append(u_ds)
        n_ds_frames = len(u_ds.trajectory)
        ctx.dataset_offsets.append(curr_frame_offset)

        if n_ds_frames > 0:
            pos0 = u_ds.trajectory[0].positions[ctx.ds_consensus_indexers[idx]]
            ca_centered = pos0 - pos0.mean(axis=0)
            ds_rg = float(np.mean(np.sqrt(np.sum(ca_centered ** 2, axis=1))))
            if 0 < ds_rg < 5.0:
                print(
                    f"  -> Auto-detected nanometer (nm) coordinate scale for dataset '{label}' "
                    f"(Rg={ds_rg:.2f} nm). Auto-scaling to Ångströms (x10)..."
                )
                ctx.dataset_scales.append(10.0)
            else:
                ctx.dataset_scales.append(1.0)
        else:
            ctx.dataset_scales.append(1.0)

        ctx.dataset_tags.extend([label] * n_ds_frames)
        for local_idx in range(n_ds_frames):
            ctx.frame_universe_map.append((u_ds, local_idx))
        curr_frame_offset += n_ds_frames

    ctx.dataset_offsets.append(curr_frame_offset)
    ctx.n_frames_initial = total_frames
    ctx.valid_frames = np.arange(total_frames)
    print(f"Loaded {total_frames} total conformations across {len(dataset_configs)} dataset(s).")
    ctx.methodology.append(f"Loaded {len(dataset_configs)} dataset(s). Total conformations: {total_frames}")

    return StageResult(
        stage_name="stage1_ingestion",
        success=True,
        exit_code=0,
        artifacts=[param_path, merged_ref_path, unified_ref_path],
        metrics={
            "n_datasets": len(dataset_configs),
            "consensus_len": consensus_len,
            "n_consensus_atoms": len(u_consensus.atoms),
            "n_frames_initial": total_frames,
        },
    )
