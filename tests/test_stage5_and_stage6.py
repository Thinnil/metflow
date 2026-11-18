"""Tests for Stage 5 (CSV fusion, energy validation) and Stage 6 (dashboard, plots, reports)."""

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from metflow.models import PipelineConfig, TrajectoryContext
from metflow.stage5_energy.csv_fusion import fuse_energy_csv, is_valid_energy
from metflow.stage5_energy.openmm_engine import _write_checkpoint_atomic, run_openmm_energy, run_stage5
from metflow.stage6_visualization.dashboard import generate_dashboard
from metflow.stage6_visualization.plots import (
    generate_advanced_plots,
    generate_cluster_plots,
    generate_comparison_plots,
)
from metflow.stage6_visualization.reports import (
    init_methodology,
    log_methodology,
    write_cluster_summary,
    write_methodology,
)
from scripts.run_pipeline import main as orchestrator_main, run_stages


class TestStage5AndStage6(unittest.TestCase):
    def test_energy_validation_and_csv_fusion(self):
        self.assertTrue(is_valid_energy(-1234.5))
        self.assertFalse(is_valid_energy(np.nan))
        self.assertFalse(is_valid_energy(np.inf))
        self.assertFalse(is_valid_energy(None))
        self.assertFalse(is_valid_energy("not_a_number"))

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "energy_checkpoint.csv")
            pd.DataFrame({"Frame": [0, 2], "Energy_kcal_mol": [-100.0, -200.0]}).to_csv(csv_path, index=False)
            df_data = {"Frame_Index": [0, 1, 2]}
            count = fuse_energy_csv(df_data, np.array([0, 1, 2]), csv_path)
            self.assertEqual(count, 2)
            self.assertEqual(df_data["OpenMM_Energy_kcal_mol"][0], -100.0)
            self.assertTrue(np.isnan(df_data["OpenMM_Energy_kcal_mol"][1]))
            self.assertEqual(df_data["OpenMM_Energy_kcal_mol"][2], -200.0)

    def test_dashboard_and_plots_small_n(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "Frame_Index": np.arange(8),
                "Dataset": ["WT"] * 4 + ["Mut"] * 4,
                "PCA_1 (50.0%)": np.linspace(-1, 1, 8),
                "PCA_2 (30.0%)": np.linspace(1, -1, 8),
                "PCA_3 (10.0%)": np.zeros(8),
                "PCA_KMeans_Cluster": [0, 0, 0, 0, 1, 1, 1, 1],
                "Daura_Cluster": [0, 0, 0, 0, 1, 1, 1, 1],
                "OpenMM_Energy_kcal_mol": np.linspace(-500, -400, 8),
            })

            dash = generate_dashboard(tmpdir, df=df)
            self.assertTrue(os.path.exists(dash))
            with open(dash, "r", encoding="utf-8") as f:
                html = f.read()
            self.assertNotIn("/*EMBEDDED_DATA_PLACEHOLDER*/[]", html)
            self.assertIn("load merged_reference.pdb", html)

            cmp_arts, unique_ds, palette = generate_comparison_plots(df, tmpdir)
            self.assertEqual(len(cmp_arts), 3)
            for p in cmp_arts:
                self.assertTrue(os.path.exists(p))

            cl_arts = generate_cluster_plots(df, tmpdir, np.array([50.0, 30.0, 10.0]))
            self.assertEqual(len(cl_arts), 2)

            # Advanced plots with n_samples=8 (tests safe t-SNE perplexity <= (8-1)//3)
            rng = np.random.default_rng(3)
            X = rng.normal(size=(8, 6))
            rmsd_mat = np.abs(rng.normal(size=(8, 8)))
            adv_arts = generate_advanced_plots(df, X, rmsd_mat, tmpdir, unique_ds, palette, [])
            self.assertGreaterEqual(len(adv_arts), 3)

            sum_path = write_cluster_summary(df, tmpdir, unique_ds)
            meth_path = write_methodology(["Step 1", "Step 2"], tmpdir)
            self.assertTrue(os.path.exists(sum_path))
            self.assertTrue(os.path.exists(meth_path))

    def test_fuse_energy_csv_duplicates_and_invalid_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "energy_checkpoint.csv")
            pd.DataFrame({
                "Frame": ["0", "0", "1", "1", "2", "bad_id", "99"],
                "Energy_kcal_mol": [-150.0, "NaN", np.nan, -250.0, "inf", -300.0, -400.0],
            }).to_csv(csv_path, index=False)

            df_data = {"Frame_Index": [0, 1, 2]}
            count = fuse_energy_csv(df_data, np.array([0, 1, 2]), csv_path)
            self.assertEqual(count, 2)
            self.assertEqual(df_data["OpenMM_Energy_kcal_mol"][0], -150.0)
            self.assertEqual(df_data["OpenMM_Energy_kcal_mol"][1], -250.0)
            self.assertTrue(np.isnan(df_data["OpenMM_Energy_kcal_mol"][2]))

    def test_atomic_checkpoint_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "energy_checkpoint.csv")
            df_ckpt = _write_checkpoint_atomic({0: -100.5, 1: -99.2}, csv_path)
            self.assertTrue(os.path.exists(csv_path))
            self.assertFalse(os.path.exists(f"{csv_path}.tmp"))
            self.assertEqual(len(df_ckpt), 2)

    def test_openmm_engine_mocked_checkpoint_freeze_and_interrupt(self):
        class FakeAtom:
            def __init__(self, idx, name, symbol, resid=1, res_idx=0, chain_idx=0):
                self.index = idx
                self.name = name
                self.resid = resid
                self.element = types.SimpleNamespace(symbol=symbol)
                self.residue = types.SimpleNamespace(
                    index=res_idx,
                    chain=types.SimpleNamespace(index=chain_idx),
                    atoms=lambda: [self],
                )

        class _FakeResidues(list):
            def __init__(self, items, resnames):
                super().__init__(items)
                self.resnames = resnames

        class _FakeRes:
            def __init__(self, index=0, resname="ALA"):
                self.index = index
                self.resname = resname

        class FakeMDAUniverse:
            def __init__(self, n_frames=55):
                res0 = _FakeRes(index=0, resname="ALA")
                atom0 = FakeAtom(0, "CA", "C", resid=1)
                atom0.residue = res0
                self.atoms = [atom0]
                self.residues = _FakeResidues([res0], ["ALA"])
                self.trajectory = _FakeTraj(n_frames)

            def select_atoms(self, _sel):
                return _FakeAtomGroup(self.atoms, self.residues)

        class _FakeTraj:
            def __init__(self, n_frames):
                self._frames = [types.SimpleNamespace(positions=np.zeros((1, 3), dtype=np.float32)) for _ in range(n_frames)]
                self.closed = False

            def __len__(self):
                return len(self._frames)

            def __getitem__(self, idx):
                return self._frames[idx]

            def close(self):
                self.closed = True

        class _FakeAtomGroup:
            def __init__(self, atoms, residues):
                self.atoms = atoms
                self.residues = residues
                self.indices = np.array([a.index for a in atoms], dtype=int)

            def __len__(self):
                return len(self.atoms)

            def __iter__(self):
                return iter(self.atoms)

            def write(self, path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("END\n")

        # Build fake openmm, openmm.app, openmm.unit, pdbfixer, MDAnalysis, tqdm
        fake_atoms_heavy = [FakeAtom(0, "CA", "C")]
        fake_atoms_full = [FakeAtom(0, "CA", "C"), FakeAtom(1, "HA", "H")]

        class FakePDBFixer:
            def __init__(self, filename):
                self.topology = types.SimpleNamespace(
                    atoms=lambda: fake_atoms_heavy,
                    getNumAtoms=lambda: len(fake_atoms_heavy),
                )
                self.positions = types.SimpleNamespace(value_in_unit=lambda _u: np.zeros((1, 3), dtype=np.float32))

            def findMissingResidues(self):
                pass

            def findNonstandardResidues(self):
                pass

            def replaceNonstandardResidues(self):
                pass

            def findMissingAtoms(self):
                pass

            def addMissingAtoms(self):
                pass

        fake_system = MagicMock()
        fake_forcefield = MagicMock()
        fake_forcefield.createSystem.return_value = fake_system

        class FakeModeller:
            def __init__(self, topo, pos):
                self.topology = types.SimpleNamespace(
                    atoms=lambda: fake_atoms_full,
                    bonds=lambda: [(fake_atoms_full[0], fake_atoms_full[1])],
                )
                self.positions = types.SimpleNamespace(value_in_unit=lambda _u: np.zeros((2, 3), dtype=np.float32))

            def addHydrogens(self, _ff):
                pass

        fake_platform = MagicMock()
        fake_platform.getName.return_value = "CPU"

        fake_sim = MagicMock()
        fake_state = MagicMock()
        fake_state.getPotentialEnergy.return_value.value_in_unit.return_value = -42.0
        fake_sim.context.getState.return_value = fake_state

        fake_mm = types.ModuleType("openmm")
        fake_mm.LangevinMiddleIntegrator = MagicMock()
        fake_mm.Platform = MagicMock()
        fake_mm.Platform.getPlatformByName.side_effect = [RuntimeError("No CUDA"), fake_platform]
        fake_mm.app = types.ModuleType("openmm.app")
        fake_mm.app.ForceField = MagicMock(return_value=fake_forcefield)
        fake_mm.app.HBonds = "HBonds"
        fake_mm.app.Modeller = FakeModeller
        fake_mm.app.NoCutoff = "NoCutoff"
        fake_mm.app.Simulation = MagicMock(return_value=fake_sim)

        fake_unit = types.ModuleType("openmm.unit")
        fake_unit.dalton = 1.0
        fake_unit.kelvin = 1.0
        fake_unit.picosecond = 1.0
        fake_unit.picoseconds = 1.0
        fake_unit.nanometers = 1.0
        fake_unit.kilocalories_per_mole = 1.0

        fake_pdbfixer = types.ModuleType("pdbfixer")
        fake_pdbfixer.PDBFixer = FakePDBFixer

        fake_mda = types.ModuleType("MDAnalysis")
        fake_mda.Universe = lambda *args, **kwargs: FakeMDAUniverse(n_frames=55)

        fake_tqdm = types.ModuleType("tqdm")
        fake_tqdm.tqdm = lambda it, **kwargs: it

        modules_patch = {
            "openmm": fake_mm,
            "openmm.app": fake_mm.app,
            "openmm.unit": fake_unit,
            "pdbfixer": fake_pdbfixer,
            "MDAnalysis": fake_mda,
            "tqdm": fake_tqdm,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_pdb = os.path.join(tmpdir, "merged_reference.pdb")
            xtc_path = os.path.join(tmpdir, "merged_trajectory.xtc")
            with open(ref_pdb, "w", encoding="utf-8") as f:
                f.write("END\n")
            with open(xtc_path, "w", encoding="utf-8") as f:
                f.write("")

            cfg = PipelineConfig(out_dir=tmpdir, calc_energy="true", min_steps=0)
            ctx = TrajectoryContext(
                config=cfg,
                out_dir=tmpdir,
                merged_ref_path=ref_pdb,
                merged_xtc_path=xtc_path,
                valid_frames=np.arange(55),
                df_data={"Frame_Index": np.arange(55)},
            )
            init_methodology(ctx, "Energy Test")

            with patch.dict(sys.modules, modules_patch):
                res = run_stage5(ctx)

            self.assertTrue(res.success)
            # Heavy atom (index 0, C) was frozen; Hydrogen (index 1, H) was NOT frozen
            fake_system.setParticleMass.assert_called_once_with(0, 0.0)
            # Check relax_steps logged as 25-step (never 0-step)
            meth_path = os.path.join(tmpdir, "methodology.txt")
            with open(meth_path, "r", encoding="utf-8") as f:
                meth_txt = f.read()
            self.assertIn("25-step Minimization", meth_txt)
            self.assertNotIn("0-step Minimization", meth_txt)

            # Test KeyboardInterrupt handling mid-loop
            fake_mm.Platform.getPlatformByName.side_effect = [RuntimeError("No CUDA"), fake_platform]
            call_counter = {"n": 0}

            def interrupt_on_third(*args, **kwargs):
                call_counter["n"] += 1
                if call_counter["n"] >= 3:
                    raise KeyboardInterrupt()

            fake_sim.minimizeEnergy.side_effect = interrupt_on_third
            os.remove(os.path.join(tmpdir, "energy_checkpoint.csv"))
            ctx2 = TrajectoryContext(
                config=cfg,
                out_dir=tmpdir,
                merged_ref_path=ref_pdb,
                merged_xtc_path=xtc_path,
                valid_frames=np.arange(55),
                df_data={"Frame_Index": np.arange(55)},
            )
            with patch.dict(sys.modules, modules_patch):
                energy_dict, interrupted, arts, _, _ = run_openmm_energy(ctx2)
            self.assertTrue(interrupted)
            self.assertEqual(len(energy_dict), 2)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "energy_checkpoint.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "partial_energy_progress.png")))

    def test_live_methodology_incremental_and_session_appending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = PipelineConfig(out_dir=tmpdir)
            ctx = TrajectoryContext(config=cfg, out_dir=tmpdir)
            meth_path = init_methodology(ctx, "First Run")
            log_methodology(ctx, "Stage 1 complete.")

            # Verify flushed immediately before any Stage 6 call
            with open(meth_path, "r", encoding="utf-8") as f:
                content1 = f.read()
            self.assertIn("Mode: First Run", content1)
            self.assertIn("Stage 1 complete.", content1)

            # Second session appends instead of overwriting
            ctx2 = TrajectoryContext(config=cfg, out_dir=tmpdir)
            init_methodology(ctx2, "Second Run")
            log_methodology(ctx2, "Stage 2 complete.")
            with open(meth_path, "r", encoding="utf-8") as f:
                content2 = f.read()
            self.assertIn("Mode: First Run", content2)
            self.assertIn("Stage 1 complete.", content2)
            self.assertIn("Mode: Second Run", content2)
            self.assertIn("Stage 2 complete.", content2)

            # Error logging in run_stages
            def fail_stage(_c):
                raise RuntimeError("kaboom")

            rc = run_stages(ctx2, [("FailStage", fail_stage)])
            self.assertEqual(rc, 1)
            with open(meth_path, "r", encoding="utf-8") as f:
                content3 = f.read()
            self.assertIn("Pipeline error during FailStage: kaboom", content3)

            # --only_dashboard true appends Dashboard Only session
            pd.DataFrame({"Frame_Index": [0, 1], "Dataset": ["A", "A"], "PCA_1": [0.1, 0.2], "PCA_2": [0.3, 0.4]}).to_csv(
                os.path.join(tmpdir, "global_cluster_metrics.csv"), index=False
            )
            rc_dash = orchestrator_main(["--only_dashboard", "true", "--out_dir", tmpdir], base_dir=tmpdir, check_deps=False)
            self.assertEqual(rc_dash, 0)
            with open(meth_path, "r", encoding="utf-8") as f:
                content4 = f.read()
            self.assertIn("Mode: Dashboard Only", content4)

    def test_cluster_summary_and_dashboard_accessibility(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty df guard
            empty_df = pd.DataFrame(columns=["Frame_Index", "Dataset"])
            empty_sum = write_cluster_summary(empty_df, tmpdir, [], config=PipelineConfig())
            with open(empty_sum, "r", encoding="utf-8") as f:
                self.assertIn("No frames available", f.read())

            # Full df with CNN noise (-1), RMSD, and Energy
            df = pd.DataFrame({
                "Frame_Index": np.arange(6),
                "Dataset": ["WT"] * 3 + ["Mut"] * 3,
                "RMSD_to_Reference": [1.0, 1.2, 1.4, 2.0, 2.2, 2.4],
                "OpenMM_Energy_kcal_mol": [-100.0, -110.0, np.nan, -90.0, -95.0, -105.0],
                "PCA_1 (50.0%)": np.linspace(-1, 1, 6),
                "PCA_2 (30.0%)": np.linspace(1, -1, 6),
                "CNN_Cluster": [0, 0, -1, 1, 1, -1],
            })
            cfg = PipelineConfig(k=4, rmsd=1.5, cnn_eps=2.5, cluster="cnn")
            sum_path = write_cluster_summary(df, tmpdir, ["WT", "Mut"], config=cfg)
            with open(sum_path, "r", encoding="utf-8") as f:
                summary_txt = f.read()
            self.assertIn("Daura RMSD Cutoff: 1.5 Å", summary_txt)
            self.assertIn("CNN/DBSCAN eps: 2.5 Å", summary_txt)
            self.assertIn("Noise / Unclustered (-1)", summary_txt)
            self.assertIn("RMSD to Reference (Å)", summary_txt)
            self.assertIn("OpenMM Energy (kcal/mol)", summary_txt)

            # Dashboard accessibility & 3D scene / filter checks
            dash = generate_dashboard(tmpdir, df=df)
            with open(dash, "r", encoding="utf-8") as f:
                html = f.read()
            self.assertIn('<meta name="viewport"', html)
            self.assertIn("<aside", html)
            self.assertIn("<main", html)
            self.assertIn('tabindex="0"', html)
            self.assertIn('role="button"', html)
            self.assertIn('role="dialog"', html)
            self.assertIn('aria-modal="true"', html)
            self.assertIn('aria-live="polite"', html)
            self.assertNotIn("alert(", html)
            self.assertNotIn("#888", html)
            self.assertIn("layout.scene", html)
            self.assertIn("parseVal(d['OpenMM_Energy_kcal_mol']) !== null", html)


if __name__ == "__main__":
    unittest.main()
