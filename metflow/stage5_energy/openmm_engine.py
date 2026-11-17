"""OpenMM Amber14/OBC2 frozen-heavy-atom energy evaluation with checkpointing."""

import os
import subprocess
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metflow.models import StageResult, TrajectoryContext
from metflow.stage1_ingestion.indexing import PROTEIN_RESNAMES, precompute_atom_indexer
from metflow.stage5_energy.csv_fusion import fuse_energy_csv, is_valid_energy


def _save_partial_plot(df_partial: pd.DataFrame, out_dir: str) -> str:
    """Generate partial_energy_progress.png on KeyboardInterrupt."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(df_partial["Frame"], df_partial["Energy_kcal_mol"], alpha=0.5, s=10, color="red")
    plt.xlabel("Frame Index")
    plt.ylabel("Potential Energy (kcal/mol)")
    plt.title("Partial Energy Timeline")

    plt.subplot(1, 2, 2)
    valid_partial = df_partial["Energy_kcal_mol"].dropna()
    if len(valid_partial) > 0:
        plt.hist(valid_partial, bins=30, color="blue", alpha=0.7)
    plt.xlabel("Potential Energy (kcal/mol)")
    plt.ylabel("Frequency (Count)")
    plt.title("Partial Energy Distribution")

    plt.tight_layout()
    partial_plot = os.path.join(out_dir, "partial_energy_progress.png")
    plt.savefig(partial_plot, dpi=300)
    plt.close()
    print(f"📊 Generated a progress visualization of your partial results: {partial_plot}")
    return partial_plot


def run_openmm_energy(ctx: TrajectoryContext) -> Tuple[Dict[int, float], bool, List[str]]:
    """Evaluate OpenMM potential energies with 50-frame checkpointing and clean interrupt handling."""
    import MDAnalysis as mda

    config = ctx.config
    out_dir = ctx.out_dir
    valid_frames = ctx.valid_frames
    artifacts: List[str] = []

    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "setuptools==69.0.0"], check=True)
        except Exception:
            pass

    from openmm.app import ForceField, HBonds, Modeller, NoCutoff
    import openmm as mm
    import openmm.unit as unit
    from pdbfixer import PDBFixer
    from tqdm import tqdm

    print("  -> Initializing OpenMM system from clean consensus reference...")
    merged_ref_path = ctx.merged_ref_path or os.path.join(out_dir, "merged_reference.pdb")
    if not os.path.exists(merged_ref_path):
        merged_ref_path = ctx.master_ref_pdb

    prot_res_str = " ".join(PROTEIN_RESNAMES)
    u_ref_clean = mda.Universe(merged_ref_path)
    u_heavy = u_ref_clean.select_atoms(
        f"resname {prot_res_str} and not (name H* or name [0-9]H* or type H or element H)"
    )
    if len(u_heavy) == 0:
        u_heavy = u_ref_clean.select_atoms(
            "not (name H* or name [0-9]H* or type H or element H or "
            "resname WAT or resname HOH or resname SOL or resname NA or resname CL)"
        )

    resnames = list(u_heavy.residues.resnames)
    for i in range(len(resnames)):
        if resnames[i] in ["HIE", "HID", "HIP", "HSD", "HSE", "HSP"]:
            resnames[i] = "HIS"
        elif resnames[i] in ["CYX", "CYM"]:
            resnames[i] = "CYS"
        elif resnames[i] in ["MSE"]:
            resnames[i] = "MET"
    u_heavy.residues.resnames = resnames

    tmp_pdb = os.path.join(out_dir, "tmp_clean_heavy.pdb")
    u_heavy.write(tmp_pdb)

    print("  -> Initializing OpenMM with amber14/obc2 implicit solvent...")
    print("  -> Fixing missing heavy atoms (e.g. OXT) with PDBFixer...")
    fixer = PDBFixer(filename=tmp_pdb)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    u_heavy_res_to_idx = {res: idx for idx, res in enumerate(u_heavy.residues)}
    u_atom_map = {
        (u_heavy_res_to_idx[a.residue], a.name.strip().upper()): i
        for i, a in enumerate(u_heavy.atoms)
    }

    fixed_to_orig = []
    parent_atom = []
    for a in fixer.topology.atoms():
        key = (a.residue.index, a.name.strip().upper())
        if key in u_atom_map:
            fixed_to_orig.append(u_atom_map[key])
            parent_atom.append(-1)
        else:
            fixed_to_orig.append(-1)
            anchor = -1
            for other in a.residue.atoms():
                k2 = (other.residue.index, other.name.strip().upper())
                if k2 in u_atom_map:
                    anchor = u_atom_map[k2]
                    if other.name.strip().upper() in ["CA", "C", "N"]:
                        break
            parent_atom.append(anchor)

    fixed_to_orig_arr = np.array(fixed_to_orig, dtype=int)
    parent_atom_arr = np.array(parent_atom, dtype=int)

    fixed_mask = fixed_to_orig_arr != -1
    anchor_mask = (~fixed_mask) & (parent_atom_arr != -1)
    fallback_mask = (~fixed_mask) & (~anchor_mask)

    fixed_orig_idx = fixed_to_orig_arr[fixed_mask]
    anchor_parent_idx = parent_atom_arr[anchor_mask]

    num_fixer_atoms = fixer.topology.getNumAtoms()
    n_anchor = int(np.sum(anchor_mask))
    n_fallback = int(np.sum(fallback_mask))

    forcefield = ForceField("amber14-all.xml", "implicit/obc2.xml")
    print("  -> Building reference system and compiling context...")
    ref_modeller = Modeller(fixer.topology, fixer.positions)
    ref_modeller.addHydrogens(forcefield)
    system = forcefield.createSystem(ref_modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)

    for a in ref_modeller.topology.atoms():
        if a.element.symbol != "H":
            system.setParticleMass(a.index, 0.0 * unit.dalton)

    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    try:
        platform = mm.Platform.getPlatformByName("CUDA")
        print("  -> 🚀 Using CUDA (GPU) for blazing fast OpenMM calculations!")
    except Exception:
        platform = mm.Platform.getPlatformByName("CPU")
        print("  -> 🐢 Using CPU. (No CUDA detected). Calculations may be slow.")
    simulation = mm.app.Simulation(ref_modeller.topology, system, integrator, platform)

    ref_full_pos = np.array(ref_modeller.positions.value_in_unit(unit.nanometers), dtype=np.float32)
    ref_pos = np.array(fixer.positions.value_in_unit(unit.nanometers), dtype=np.float32)
    ref_pos_anchor = ref_pos[anchor_parent_idx] if n_anchor > 0 else None
    ref_pos_anchor_offset = (ref_pos[anchor_mask] - ref_pos_anchor) if n_anchor > 0 else None

    full_atoms = list(ref_modeller.topology.atoms())
    full_atom_dict = {(a.residue.chain.index, a.residue.index, a.name.strip().upper()): i for i, a in enumerate(full_atoms)}
    heavy_to_full = [
        full_atom_dict.get((a.residue.chain.index, a.residue.index, a.name.strip().upper()), -1)
        for a in fixer.topology.atoms()
    ]
    heavy_to_full_arr = np.array(heavy_to_full, dtype=int)
    valid_h2f_mask = heavy_to_full_arr != -1
    valid_h2f_src = np.where(valid_h2f_mask)[0]
    valid_h2f_dst = heavy_to_full_arr[valid_h2f_mask]

    h_parent_map = {}
    for bond in ref_modeller.topology.bonds():
        a1, a2 = bond[0], bond[1]
        is_h1 = a1.element.symbol == "H"
        is_h2 = a2.element.symbol == "H"
        if is_h1 and not is_h2:
            h_parent_map[a1.index] = a2.index
        elif is_h2 and not is_h1:
            h_parent_map[a2.index] = a1.index

    h_indices = np.array(sorted(h_parent_map.keys()), dtype=int)
    if len(h_indices) > 0:
        h_parents = np.array([h_parent_map[i] for i in h_indices], dtype=int)
        h_offsets = ref_full_pos[h_indices] - ref_full_pos[h_parents]
    else:
        h_parents = np.array([], dtype=int)
        h_offsets = np.zeros((0, 3), dtype=np.float32)

    merged_xtc_path = ctx.merged_xtc_path or os.path.join(out_dir, "merged_trajectory.xtc")
    use_merged_traj = False
    u_openmm = None
    if os.path.exists(merged_ref_path) and os.path.exists(merged_xtc_path):
        try:
            u_openmm = mda.Universe(merged_ref_path, merged_xtc_path)
            use_merged_traj = len(u_openmm.trajectory) == len(valid_frames)
        except Exception:
            use_merged_traj = False

    universe_heavy_info = {}
    if not use_merged_traj:
        for u_obj in set(u for u, _ in ctx.frame_universe_map):
            u_h = u_obj.select_atoms(
                f"resname {prot_res_str} and not (name H* or name [0-9]H* or type H or element H)"
            )
            if len(u_h) == 0:
                u_h = u_obj.select_atoms(
                    "not (name H* or name [0-9]H* or type H or element H or "
                    "resname WAT or resname HOH or resname SOL or resname NA or resname CL)"
                )
            h_indexer = precompute_atom_indexer(u_h)
            if len(u_h) == len(u_heavy):
                map_idx = None
            else:
                curr_map = {(a.resid, a.name): i for i, a in enumerate(u_h)}
                map_idx = np.zeros(len(u_heavy), dtype=int)
                for m_idx, m_atom in enumerate(u_heavy.atoms):
                    key = (m_atom.resid, m_atom.name)
                    map_idx[m_idx] = curr_map.get(key, min(m_idx, len(u_h) - 1))
            universe_heavy_info[id(u_obj)] = (h_indexer, map_idx)

    energy_csv = os.path.join(out_dir, "energy_checkpoint.csv")
    energy_dict: Dict[int, float] = {}
    if os.path.exists(energy_csv):
        try:
            df_resume = pd.read_csv(energy_csv)
            ene_col = "Energy_kcal_mol" if "Energy_kcal_mol" in df_resume.columns else "Energy"
            raw_dict = dict(zip(df_resume["Frame"], df_resume[ene_col]))
            energy_dict = {int(f): v for f, v in raw_dict.items() if is_valid_energy(v)}
            print(f"  -> Found checkpoint! Resuming {len(energy_dict)} valid previously minimized frames...")
        except Exception:
            energy_dict = {}

    relax_steps = config.min_steps if config.min_steps > 0 else 25
    if config.min_steps > 0:
        print(f"  -> Relaxing hydrogens and evaluating energy ({config.min_steps} steps per frame, heavy atoms frozen)...")
    else:
        print("  -> Evaluating potential energy with hydrogen clash-relief (25 steps per frame, heavy atoms frozen)...")

    full_heavy_pos = np.zeros((num_fixer_atoms, 3), dtype=np.float32)
    if n_fallback > 0:
        full_heavy_pos[fallback_mask] = ref_pos[fallback_mask]
    current_full_pos = ref_full_pos.copy()

    try:
        for local_idx, frame_idx in enumerate(tqdm(valid_frames)):
            frame_key = int(frame_idx)
            if frame_key in energy_dict and is_valid_energy(energy_dict[frame_key]):
                continue

            if use_merged_traj:
                ts = u_openmm.trajectory[local_idx]
                heavy_pos_src = ts.positions * 0.1
            else:
                u_curr, raw_local_idx = ctx.frame_universe_map[frame_key]
                ts = u_curr.trajectory[raw_local_idx]
                h_indexer, map_idx = universe_heavy_info[id(u_curr)]
                pos_angstroms = ts.positions[h_indexer]
                pos_centered = pos_angstroms - pos_angstroms.mean(axis=0)
                rms_dist = np.sqrt(np.mean(np.sum(pos_centered ** 2, axis=1)))
                scale = 0.1 if rms_dist >= 5.0 else 1.0
                frame_orig_pos = pos_angstroms * scale
                heavy_pos_src = frame_orig_pos if map_idx is None else frame_orig_pos[map_idx]

            full_heavy_pos[fixed_mask] = heavy_pos_src[fixed_orig_idx]
            if n_anchor > 0:
                full_heavy_pos[anchor_mask] = (
                    heavy_pos_src[anchor_parent_idx]
                    + ref_pos_anchor_offset
                    + np.random.normal(0, 0.001, (n_anchor, 3))
                )

            current_full_pos[valid_h2f_dst] = full_heavy_pos[valid_h2f_src]
            if len(h_indices) > 0:
                current_full_pos[h_indices] = current_full_pos[h_parents] + h_offsets

            try:
                simulation.context.setPositions(current_full_pos * unit.nanometers)
                simulation.minimizeEnergy(maxIterations=relax_steps)
                state = simulation.context.getState(getEnergy=True)
                energy_val = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
                if np.isnan(energy_val) or np.isinf(energy_val):
                    energy_val = np.nan
            except Exception:
                energy_val = np.nan

            energy_dict[frame_key] = energy_val
            if len(energy_dict) % 50 == 0:
                pd.DataFrame(list(energy_dict.items()), columns=["Frame", "Energy_kcal_mol"]).to_csv(
                    energy_csv, index=False
                )
    except KeyboardInterrupt:
        df_partial = pd.DataFrame(list(energy_dict.items()), columns=["Frame", "Energy_kcal_mol"])
        df_partial.to_csv(energy_csv, index=False)
        artifacts.append(energy_csv)
        print(f"\n⚠️  Energy calculation PAUSED by user! Saved checkpoint ({len(energy_dict)} frames) to {energy_csv}")
        if len(df_partial) > 1:
            artifacts.append(_save_partial_plot(df_partial, out_dir))
        print("Run the script again to resume exactly where you left off.")
        return energy_dict, True, artifacts
    finally:
        if u_openmm is not None:
            try:
                u_openmm.trajectory.close()
            except Exception:
                pass

    pd.DataFrame(list(energy_dict.items()), columns=["Frame", "Energy_kcal_mol"]).to_csv(energy_csv, index=False)
    artifacts.append(energy_csv)
    return energy_dict, False, artifacts


def run_stage5(ctx: TrajectoryContext) -> StageResult:
    """Execute Stage 5: optional OpenMM energy calculation and checkpoint CSV fusion."""
    config = ctx.config
    out_dir = ctx.out_dir
    valid_frames = ctx.valid_frames
    artifacts: List[str] = []

    if config.is_calc_energy:
        print("\n--- Calculating Thermodynamic Energy (OpenMM Physical Minimization) ---")
        ctx.methodology.append("\n--- OpenMM Energy Calculation ---")
        try:
            energy_dict, interrupted, energy_artifacts = run_openmm_energy(ctx)
            artifacts.extend(energy_artifacts)
            if interrupted:
                ctx.energy_interrupted = True
                return StageResult(
                    stage_name="stage5_energy",
                    success=True,
                    exit_code=0,
                    artifacts=artifacts,
                    metrics={"interrupted": True, "evaluated_frames": len(energy_dict)},
                )
            energies_kcal = [energy_dict.get(int(f), np.nan) for f in valid_frames]
            ctx.df_data["OpenMM_Energy_kcal_mol"] = energies_kcal
            valid_cnt = sum(1 for e in energies_kcal if pd.notna(e))
            ctx.methodology.append(
                f"Evaluated Thermodynamic Potential Energy (OpenMM, Amber14/OBC2, {config.min_steps}-step "
                f"Minimization, {valid_cnt}/{len(valid_frames)} valid)."
            )
            print(f"✅ OpenMM Energy Evaluation complete ({valid_cnt}/{len(valid_frames)} valid frames).")
        except Exception as exc:
            print(f"❌ Error during OpenMM Energy calculation: {exc}")
            ctx.methodology.append(f"OpenMM Energy Calculation failed: {exc}")

    energy_csv = os.path.join(out_dir, "energy_checkpoint.csv")
    fused_count = fuse_energy_csv(ctx.df_data, valid_frames, energy_csv)
    return StageResult(
        stage_name="stage5_energy",
        success=True,
        exit_code=0,
        artifacts=artifacts,
        metrics={"interrupted": False, "fused_count": fused_count},
    )
