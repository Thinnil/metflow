"""Tests for Stage 3 (Kabsch, PCA, DCCM, Sub-PCA) and Stage 4 (KMeans, Daura, CNN)."""

import unittest
import numpy as np

from metflow.stage3_reduction.allostery import compute_cavity_subpca, compute_dccm
from metflow.stage3_reduction.kabsch import batched_kabsch
from metflow.stage3_reduction.pca import compute_pca
from metflow.stage4_clustering.cnn import run_cnn
from metflow.stage4_clustering.daura import run_daura
from metflow.stage4_clustering.kmeans import run_kmeans
from metflow.stage4_clustering.pairwise import compute_pairwise_rmsd


class TestStage3AndStage4(unittest.TestCase):
    def test_batched_kabsch_alignment(self):
        rng = np.random.default_rng(42)
        Q_ref = rng.normal(size=(10, 3)).astype(np.float32)
        Q_ref -= Q_ref.mean(axis=0, keepdims=True)

        # Apply arbitrary 3D rotation + translation
        theta = 0.7
        R_true = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        P_raw = np.stack([Q_ref @ R_true + np.array([5.0, -3.0, 2.0], dtype=np.float32) for _ in range(5)])
        aligned = batched_kabsch(P_raw, Q_ref, chunk_size=2)
        np.testing.assert_allclose(aligned[0], Q_ref, atol=1e-4)

        # Empty input
        empty = np.empty((0, 10, 3), dtype=np.float32)
        self.assertEqual(len(batched_kabsch(empty, Q_ref)), 0)

    def test_pca_combined_ref_and_small_n(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 15))
        tags = ["A"] * 10 + ["B"] * 10
        X_pca, model, var, _ = compute_pca(X, tags, ["A", "B"], pca_ref="combined")
        self.assertEqual(X_pca.shape, (20, 3))
        self.assertEqual(len(var), 3)

        X_pca_ref, _, _, note = compute_pca(X, tags, ["A", "B"], pca_ref="first")
        self.assertEqual(X_pca_ref.shape, (20, 3))
        self.assertIn("reference dataset 'A'", note)

        # Edge case: n_frames == 2
        X_two = rng.normal(size=(2, 15))
        X_pca_two, _, var_two, _ = compute_pca(X_two, ["A", "B"], ["A", "B"])
        self.assertEqual(X_pca_two.shape, (2, 3))
        self.assertEqual(len(var_two), 3)

    def test_dccm_and_cavity_subpca(self):
        # 10 frames, 4 atoms
        rng = np.random.default_rng(1)
        ca = rng.normal(size=(10, 4, 3))
        dccm = compute_dccm(ca)
        self.assertEqual(dccm.shape, (4, 4))
        np.testing.assert_allclose(np.diag(dccm), np.ones(4), atol=1e-6)

        cav = rng.normal(size=(10, 3, 3))
        loop = rng.normal(size=(10, 5, 3))
        cav_pca, sub_var, loop_rmsd, comps = compute_cavity_subpca(cav, loop)
        self.assertEqual(cav_pca.shape, (10, 2))
        self.assertEqual(len(sub_var), 2)
        self.assertEqual(len(loop_rmsd), 10)
        self.assertEqual(comps.shape[0], 2)

    def test_clustering_algorithms_and_edge_cases(self):
        rng = np.random.default_rng(7)
        c1 = rng.normal(loc=0.0, scale=0.1, size=(6, 9))
        c2 = rng.normal(loc=5.0, scale=0.1, size=(6, 9))
        X = np.vstack([c1, c2])
        rmsd_mat = compute_pairwise_rmsd(X, n_atoms=3)
        self.assertEqual(rmsd_mat.shape, (12, 12))

        # KMeans explicit, auto-k, clamped k > n_samples, and n_samples == 2
        labels, centers, best_k, auto_det = run_kmeans(X[:, :3], k=2)
        self.assertEqual(best_k, 2)
        self.assertFalse(auto_det)
        self.assertEqual(len(set(labels)), 2)

        labels_auto, _, best_k_auto, auto_det_true = run_kmeans(X[:, :3], k=0)
        self.assertTrue(auto_det_true)
        self.assertGreaterEqual(best_k_auto, 2)

        _, _, clamped_k, _ = run_kmeans(X[:3, :3], k=10)
        self.assertEqual(clamped_k, 3)

        _, _, two_k, _ = run_kmeans(X[:2, :3], k=0)
        self.assertEqual(two_k, 2)

        # Daura with k=2 and k<=0 (unlimited)
        d_labels, d_cents = run_daura(rmsd_mat, rmsd_cutoff=1.0, k=2)
        self.assertEqual(len(d_cents), 2)
        self.assertEqual(len(d_labels), 12)

        d_labels_unlim, d_cents_unlim = run_daura(rmsd_mat, rmsd_cutoff=1.0, k=0)
        self.assertEqual(len(d_cents_unlim), 2)
        self.assertEqual(len(d_labels_unlim), 12)

        # CNN / DBSCAN
        cnn_labels = run_cnn(rmsd_mat, eps=1.0, min_samples=3)
        self.assertEqual(len(set(cnn_labels) - {-1}), 2)

    def test_detect_sub_runs(self):
        from metflow.stage3_reduction.allostery import detect_sub_runs

        # Simulate 3 concatenated runs of 15 frames each inside dataset 't14_p1'
        tags = ["t14_p1"] * 45
        rmsd = np.concatenate([
            np.linspace(0.5, 4.0, 15),
            np.linspace(0.5, 3.8, 15),
            np.linspace(0.5, 4.2, 15),
        ])
        pca = np.zeros((45, 3))
        for start in (0, 15, 30):
            pca[start:start + 15, 0] = np.linspace(1.0, 10.0, 15)
            pca[start:start + 15, 1] = np.linspace(-2.0, 5.0, 15)

        run_ids, run_frames, run_pct = detect_sub_runs(tags, rmsd, pca)
        self.assertEqual(list(np.unique(run_ids)), [1, 2, 3])
        self.assertEqual(run_frames[0], 0)
        self.assertEqual(run_frames[15], 0)
        self.assertEqual(run_frames[30], 0)
        self.assertAlmostEqual(run_pct[14], 100.0)
        self.assertAlmostEqual(run_pct[29], 100.0)
        self.assertAlmostEqual(run_pct[44], 100.0)


if __name__ == "__main__":
    unittest.main()
