"""Sequence homology verification (>=90%) and consensus atom mapping."""

from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

from metflow.stage1_ingestion.indexing import (
    canonical_atom_sort_key,
    is_heavy_atom,
)


def compute_pairwise_identity(
    s1: str, s2: str, matcher: Optional[SequenceMatcher] = None
) -> float:
    """Compute sequence identity as sum(matching_block.size) / max(len(s1), len(s2))."""
    if s1 == s2:
        return 1.0 if len(s1) > 0 else 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    m = matcher if matcher is not None else SequenceMatcher(None, "", "")
    m.set_seqs(s1, s2)
    identical_count = sum(b.size for b in m.get_matching_blocks())
    return identical_count / max_len


def verify_sequence_homology(
    dataset_seqs: Sequence[str],
    labels: Sequence[str],
    threshold: float = 0.90,
) -> Dict[Tuple[str, str], float]:
    """Verify all dataset pairs meet the sequence identity threshold."""
    pair_ident_cache: Dict[Tuple[str, str], float] = {}
    matcher = SequenceMatcher(None, "", "")
    for i in range(len(dataset_seqs)):
        for j in range(i + 1, len(dataset_seqs)):
            s1, s2 = dataset_seqs[i], dataset_seqs[j]
            l1, l2 = labels[i], labels[j]
            if (s1, s2) in pair_ident_cache:
                identity = pair_ident_cache[(s1, s2)]
            elif (s2, s1) in pair_ident_cache:
                identity = pair_ident_cache[(s2, s1)]
            else:
                identity = compute_pairwise_identity(s1, s2, matcher=matcher)
                pair_ident_cache[(s1, s2)] = identity
                pair_ident_cache[(s2, s1)] = identity
            if identity < threshold:
                raise ValueError(
                    f"Sequence similarity across datasets '{l1}' and '{l2}' is "
                    f"{identity * 100:.1f}%, which is below the {threshold * 100:.1f}% threshold."
                )
    return pair_ident_cache


def find_consensus_bounds(
    resname_seqs: Sequence[Tuple[str, ...]],
    labels: Sequence[str],
    dataset_seqs: Sequence[str],
    pair_ident_cache: Dict[Tuple[str, str], float],
    min_consensus_len: int = 10,
) -> Tuple[List[Tuple[int, int, int, int]], int, int, int, str]:
    """Find longest contiguous homologous region across all datasets."""
    ref_seq = resname_seqs[0]
    ref_seq_len = len(ref_seq)
    seq_match_cache: Dict[Tuple[str, ...], Tuple[int, int, int, int]] = {}
    dataset_matches: List[Tuple[int, int, int, int]] = []
    reusable_ref_matcher = SequenceMatcher(None, ref_seq, None)

    for idx, ds_seq in enumerate(resname_seqs):
        label = labels[idx]
        if ds_seq == ref_seq:
            match_tuple = (0, ref_seq_len, 0, ref_seq_len)
        elif ds_seq in seq_match_cache:
            match_tuple = seq_match_cache[ds_seq]
        else:
            reusable_ref_matcher.set_seq2(ds_seq)
            match = reusable_ref_matcher.find_longest_match(0, ref_seq_len, 0, len(ds_seq))
            if match.size == 0:
                raise ValueError(
                    f"No matching residue sequence found between reference and dataset '{label}'!"
                )
            match_tuple = (match.a, match.a + match.size, match.b, match.b + match.size)
            seq_match_cache[ds_seq] = match_tuple
        dataset_matches.append(match_tuple)

    seq_summary_lines = ["\n--- Multi-Source Sequence Verification Summary ---"]
    for idx, label in enumerate(labels):
        ds_s = dataset_seqs[idx]
        ds_preview = (ds_s[:4] + "..." + ds_s[-4:]) if len(ds_s) > 10 else ds_s
        if idx == 0:
            seq_summary_lines.append(f"Dataset '{label}'       : {len(ds_s)} residues | Sequence {ds_preview}")
        else:
            ident_ref = pair_ident_cache.get(
                (dataset_seqs[0], ds_s),
                pair_ident_cache.get((ds_s, dataset_seqs[0]), 1.0),
            )
            ref_start, _, ds_start, _ = dataset_matches[idx]
            offset = ref_start - ds_start
            offset_str = f", N-terminal offset: {offset:+d}" if offset != 0 else ""
            seq_summary_lines.append(
                f"Dataset '{label}': {len(ds_s)} residues | Sequence {ds_preview} ({ident_ref * 100:.1f}% match{offset_str})"
            )

    consensus_ref_start = max(m[0] for m in dataset_matches)
    consensus_ref_end = min(m[1] for m in dataset_matches)
    consensus_len = consensus_ref_end - consensus_ref_start

    if consensus_len < min_consensus_len:
        raise ValueError(
            f"Common homologous protein sequence length across datasets is {consensus_len} residues, "
            f"which is below the minimum required {min_consensus_len} residues."
        )

    return dataset_matches, consensus_ref_start, consensus_ref_end, consensus_len, "\n".join(seq_summary_lines)


def build_consensus_atom_groups(
    temp_universes: List[Any],
    dataset_matches: List[Tuple[int, int, int, int]],
    consensus_ref_start: int,
    consensus_len: int,
) -> Tuple[List[Any], List[Any]]:
    """Build 1-to-1 heavy-atom AtomGroups in canonical order across all datasets."""
    import MDAnalysis as mda

    ds_residues_homologous = []
    for idx, (_, _, label, _, _, prot_res) in enumerate(temp_universes):
        ref_start, _, ds_start, _ = dataset_matches[idx]
        ds_offset_start = ds_start + (consensus_ref_start - ref_start)
        ds_slice = prot_res[ds_offset_start:ds_offset_start + consensus_len]
        if len(ds_slice) != consensus_len:
            raise ValueError(
                f"Dataset '{label}' homologous residue slice length ({len(ds_slice)}) "
                f"does not match expected consensus length ({consensus_len}). Boundary offset: {ds_offset_start}."
            )
        ds_residues_homologous.append(ds_slice)

    ds_consensus_atom_lists: List[List[Any]] = [[] for _ in range(len(temp_universes))]
    for r in range(consensus_len):
        res_atoms_per_ds = []
        for idx in range(len(temp_universes)):
            res_r = ds_residues_homologous[idx][r]
            heavy_atoms = [a for a in res_r.atoms if is_heavy_atom(a)]
            name_to_atom = {}
            for a in heavy_atoms:
                aname = a.name.strip().upper()
                if aname not in name_to_atom:
                    name_to_atom[aname] = a
            res_atoms_per_ds.append(name_to_atom)

        common_names = set(res_atoms_per_ds[0].keys())
        for idx in range(1, len(temp_universes)):
            common_names &= set(res_atoms_per_ds[idx].keys())

        sorted_names = sorted(common_names, key=canonical_atom_sort_key)
        for idx in range(len(temp_universes)):
            for aname in sorted_names:
                ds_consensus_atom_lists[idx].append(res_atoms_per_ds[idx][aname])

    ds_consensus_groups = [mda.AtomGroup(alist) for alist in ds_consensus_atom_lists]
    return ds_residues_homologous, ds_consensus_groups
