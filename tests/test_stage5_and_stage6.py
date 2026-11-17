"""Tests for Stage 5 (CSV fusion, energy validation) and Stage 6 (dashboard, plots, reports)."""

import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from metflow.stage5_energy.csv_fusion import fuse_energy_csv, is_valid_energy
from metflow.stage6_visualization.dashboard import generate_dashboard
from metflow.stage6_visualization.plots import (
    generate_advanced_plots,
    generate_cluster_plots,
    generate_comparison_plots,
)
from metflow.stage6_visualization.reports import write_cluster_summary, write_methodology


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


if __name__ == "__main__":
    unittest.main()
