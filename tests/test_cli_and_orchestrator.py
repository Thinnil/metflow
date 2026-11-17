"""Tests for CLI parsing, path-traversal validation, dashboard fast path, and orchestrator."""

import os
import tempfile
import unittest

import pandas as pd

from metflow.cli import parse_args_to_config, parse_list_arg, parse_res_string, save_params, validate_out_dir
from metflow.models import PipelineConfig, StageResult, TrajectoryContext
from scripts.run_pipeline import main as orchestrator_main, run_stages


class TestCLIAndOrchestrator(unittest.TestCase):
    def test_default_27_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = parse_args_to_config([], base_dir=tmpdir)
            self.assertEqual(cfg.skim, "false")
            self.assertEqual(cfg.equator, "")
            self.assertEqual(cfg.top, "")
            self.assertEqual(cfg.bottom, "")
            self.assertEqual(cfg.min_vol, 300.0)
            self.assertEqual(cfg.cluster, "pca_kmeans")
            self.assertEqual(cfg.k, 8)
            self.assertEqual(cfg.rmsd, 2.0)
            self.assertEqual(cfg.cnn_eps, 3.0)
            self.assertEqual(cfg.allostery, "false")
            self.assertEqual(cfg.cavity_res, "")
            self.assertEqual(cfg.loop_res, "")
            self.assertEqual(cfg.pca_res, "")
            self.assertEqual(cfg.advanced_plots, "false")
            self.assertEqual(cfg.gen_params, "")
            self.assertEqual(cfg.dataset_configs, "")
            self.assertEqual(cfg.pdb, "")
            self.assertEqual(cfg.xtc, "")
            self.assertEqual(cfg.multi_datasets, "false")
            self.assertEqual(cfg.dataset_labels, "")
            self.assertEqual(cfg.out_dir, os.path.abspath(os.path.join(tmpdir, "clustering_results")))
            self.assertEqual(cfg.calc_energy, "false")
            self.assertEqual(cfg.only_energy, "false")
            self.assertEqual(cfg.only_dashboard, "false")
            self.assertEqual(cfg.max_frames, 0)
            self.assertEqual(cfg.min_steps, 100)
            self.assertEqual(cfg.pca_ref, "combined")

    def test_only_energy_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = parse_args_to_config(["--only_energy", "true", "--skim", "true", "--cluster", "all"], base_dir=tmpdir)
            self.assertTrue(cfg.is_only_energy)
            self.assertEqual(cfg.skim, "false")
            self.assertEqual(cfg.allostery, "false")
            self.assertEqual(cfg.cluster, "none")
            self.assertEqual(cfg.calc_energy, "true")

    def test_parse_helpers(self):
        self.assertEqual(parse_res_string("10-12, 15 17"), [10, 11, 12, 15, 17])
        self.assertEqual(parse_list_arg("a.pdb, b.pdb"), ["a.pdb", "b.pdb"])
        self.assertEqual(parse_list_arg("a.pdb b.pdb"), ["a.pdb", "b.pdb"])
        self.assertEqual(parse_list_arg(""), [])

    def test_path_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outside = os.path.abspath(os.path.join(tmpdir, "..", "outside_dir"))
            with self.assertRaises(ValueError):
                validate_out_dir(outside, base_dir=tmpdir)

    def test_save_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "results")
            cfg = parse_args_to_config(["--out_dir", out_dir, "--k", "5"], base_dir=tmpdir)
            p = save_params(cfg, out_dir, base_dir=tmpdir)
            self.assertTrue(os.path.exists(p))
            with open(p, "r", encoding="utf-8") as f:
                txt = f.read()
            self.assertIn("k=5", txt)
            self.assertIn("pca_ref=combined", txt)

    def test_only_dashboard_fast_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "dash_out")
            os.makedirs(out_dir, exist_ok=True)
            pd.DataFrame({"Frame_Index": [0, 1], "Dataset": ["WT", "Mut"], "PCA_1": [0.1, -0.1], "PCA_2": [0.2, -0.2]}).to_csv(
                os.path.join(out_dir, "global_cluster_metrics.csv"), index=False
            )
            rc = orchestrator_main(["--only_dashboard", "true", "--out_dir", out_dir], base_dir=tmpdir, check_deps=False)
            self.assertEqual(rc, 0)
            dash_html = os.path.join(out_dir, "metflow_dashboard.html")
            self.assertTrue(os.path.exists(dash_html))
            with open(dash_html, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn('"Dataset":"WT"', content.replace(" ", ""))

    def test_orchestrator_stage_verification_and_errors(self):
        ctx = TrajectoryContext(config=PipelineConfig())

        def ok_stage(c):
            return StageResult(stage_name="ok", success=True, exit_code=0)

        def fail_stage(c):
            raise RuntimeError("boom")

        def bad_result_stage(c):
            return StageResult(stage_name="bad", success=False, exit_code=2, error="failed check")

        def interrupt_stage(c):
            raise KeyboardInterrupt()

        self.assertEqual(run_stages(ctx, [("S1", ok_stage), ("S2", ok_stage)]), 0)
        self.assertEqual(run_stages(ctx, [("S1", ok_stage), ("S2", fail_stage)]), 1)
        self.assertEqual(run_stages(ctx, [("S1", bad_result_stage)]), 2)
        self.assertEqual(run_stages(ctx, [("S1", interrupt_stage)]), 130)
        self.assertEqual(run_stages(ctx, [("S1", lambda c: "not_a_stage_result")]), 1)


if __name__ == "__main__":
    unittest.main()
