"""Tests for Stage 1 (indexing, alignment, config resolution) and Stage 2 (skimming, sampling)."""

import unittest
import numpy as np

from metflow.models import PipelineConfig
from metflow.stage1_ingestion.alignment import (
    compute_pairwise_identity,
    find_consensus_bounds,
    verify_sequence_homology,
)
from metflow.stage1_ingestion.indexing import (
    canonical_atom_sort_key,
    is_heavy_atom,
    normalize_resname,
    precompute_atom_indexer,
    res_to_1letter,
)
from metflow.stage1_ingestion.loader import resolve_dataset_configs
from metflow.stage2_skimming.sampling import downsample_per_dataset
from metflow.stage2_skimming.skimmer import _calc_vol, compute_hull_volumes


class _DummyAtom:
    def __init__(self, name, atom_type="C", element="C"):
        self.name = name
        self.type = atom_type
        self.element = element


class TestStage1AndStage2(unittest.TestCase):
    def test_residue_and_atom_helpers(self):
        self.assertEqual(normalize_resname("HSD"), "HIS")
        self.assertEqual(normalize_resname("MSE"), "MET")
        self.assertEqual(res_to_1letter("HIE"), "H")
        self.assertEqual(res_to_1letter("UNK"), "X")
        self.assertLess(canonical_atom_sort_key("N"), canonical_atom_sort_key("CA"))
        self.assertLess(canonical_atom_sort_key("CA"), canonical_atom_sort_key("CB"))
        self.assertTrue(is_heavy_atom(_DummyAtom("CA")))
        self.assertFalse(is_heavy_atom(_DummyAtom("1HB", atom_type="H", element="H")))
        self.assertFalse(is_heavy_atom(_DummyAtom("HN", atom_type="H", element="H")))

    def test_precompute_atom_indexer(self):
        self.assertEqual(precompute_atom_indexer(np.array([2, 3, 4, 5])), slice(2, 6))
        idx = precompute_atom_indexer(np.array([2, 4, 7]))
        self.assertTrue(isinstance(idx, np.ndarray))
        self.assertEqual(precompute_atom_indexer(np.array([])), slice(0, 0))

    def test_sequence_homology_and_consensus(self):
        s_ref = "ACDEFGHIKLMNPQRSTVWY"
        s_mut = "ACDEFGHIKLMNPQRSTVWA"  # 19/20 = 95%
        self.assertAlmostEqual(compute_pairwise_identity(s_ref, s_mut), 0.95)
        cache = verify_sequence_homology([s_ref, s_mut], ["WT", "Mut"], threshold=0.90)
        self.assertIn((s_ref, s_mut), cache)

        with self.assertRaises(ValueError):
            verify_sequence_homology([s_ref, "GGGGGGGGGGGGGGGGGGGG"], ["WT", "Bad"], threshold=0.90)

        res_ref = tuple(f"R{i}" for i in range(15))
        res_off = ("EXTRA",) + res_ref
        matches, c_start, c_end, c_len, summary = find_consensus_bounds(
            [res_ref, res_off], ["WT", "Offset"], [s_ref[:15], "A" + s_ref[:15]], cache, min_consensus_len=10
        )
        self.assertEqual(c_start, 0)
        self.assertEqual(c_end, 15)
        self.assertEqual(c_len, 15)
        self.assertIn("N-terminal offset: -1", summary)

    def test_resolve_dataset_configs(self):
        cfg = PipelineConfig(dataset_configs="a.pdb,a.xtc,WT;b.pdb,b.xtc")
        res = resolve_dataset_configs(cfg)
        self.assertEqual(res, [("a.pdb", "a.xtc", "WT"), ("b.pdb", "b.xtc", "b")])

        with self.assertRaises(ValueError):
            resolve_dataset_configs(PipelineConfig(dataset_configs=";;;"))

        with self.assertRaises(ValueError):
            resolve_dataset_configs(PipelineConfig(multi_datasets="true", pdb="a.pdb,b.pdb", xtc="a.xtc"))

    def test_convex_hull_volume(self):
        # Unit cube has volume 1.0
        cube = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=float)
        self.assertAlmostEqual(_calc_vol(cube), 1.0, places=5)
        # Degenerate planar points -> 0.0
        planar = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0.5, 0.5, 0], [0.2, 0.8, 0], [0.8, 0.2, 0], [0.3, 0.3, 0],
        ], dtype=float)
        self.assertEqual(_calc_vol(planar), 0.0)

        vols = compute_hull_volumes(np.stack([cube, planar]), use_multiprocessing=False)
        self.assertAlmostEqual(vols[0], 1.0, places=5)
        self.assertEqual(vols[1], 0.0)

    def test_downsample_per_dataset(self):
        valid = np.arange(10)
        tags = ["A"] * 6 + ["B"] * 4
        sampled, out_tags, _ = downsample_per_dataset(valid, tags, max_frames=2)
        self.assertEqual(len(sampled), 4)
        self.assertEqual(out_tags.count("A"), 2)
        self.assertEqual(out_tags.count("B"), 2)

        all_sampled, all_tags, _ = downsample_per_dataset(valid, tags, max_frames=0)
        self.assertEqual(len(all_sampled), 10)
        self.assertEqual(len(all_tags), 10)

    def test_heavy_atom_exclusion_expr_handles_missing_elements(self):
        from metflow.stage1_ingestion.indexing import heavy_atom_exclusion_expr

        class TopoNoElements:
            types = ["C", "N"]

        class TopoBare:
            pass

        class TopoFull:
            types = ["C"]
            elements = ["C"]

        class DummyUniverse:
            def __init__(self, topo):
                self._topology = topo

        expr_no_elem = heavy_atom_exclusion_expr(DummyUniverse(TopoNoElements()))
        self.assertNotIn("element H", expr_no_elem)
        self.assertIn("type H", expr_no_elem)

        expr_bare = heavy_atom_exclusion_expr(DummyUniverse(TopoBare()))
        self.assertNotIn("element H", expr_bare)
        self.assertNotIn("type H", expr_bare)
        self.assertIn("name H*", expr_bare)

        expr_full = heavy_atom_exclusion_expr(DummyUniverse(TopoFull()))
        self.assertIn("element H", expr_full)
        self.assertIn("type H", expr_full)


if __name__ == "__main__":
    unittest.main()
