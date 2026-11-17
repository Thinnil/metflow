"""Residue/atom constants, canonical ordering, and coordinate indexing helpers."""

from typing import Any
import numpy as np

PROTEIN_RESNAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HIE", "HID", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "MSE",
}

RESNAME_MAP = {
    "HIE": "HIS", "HID": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    "CYX": "CYS", "CYM": "CYS", "MSE": "MET",
}

AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

CANONICAL_ATOM_ORDER = [
    "N", "CA", "C", "O",
    "CB", "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",
    "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "SE",
    "CZ", "CZ2", "CZ3", "NZ", "CH2", "NH1", "NH2", "OH", "OXT",
]
ATOM_ORDER_INDEX = {name: idx for idx, name in enumerate(CANONICAL_ATOM_ORDER)}


def normalize_resname(name: str) -> str:
    """Normalize protonation/selenomethionine residue variants to canonical 3-letter code."""
    n = name.upper()
    return RESNAME_MAP.get(n, n)


def res_to_1letter(resname: str) -> str:
    """Convert 3-letter residue name to 1-letter code ('X' if unknown)."""
    return AA_3TO1.get(normalize_resname(resname), "X")


def canonical_atom_sort_key(atom_name: str):
    """Sort key placing backbone atoms (N, CA, C, O, CB...) in canonical PDB order."""
    clean = atom_name.strip().upper()
    return (ATOM_ORDER_INDEX.get(clean, 999), clean)


def is_heavy_atom(atom: Any) -> bool:
    """Return True if atom is a canonical non-hydrogen heavy protein atom."""
    aname = atom.name.strip().upper()
    aname_no_digits = aname.lstrip("0123456789")
    if aname_no_digits.startswith("H"):
        return False
    if hasattr(atom, "type") and str(atom.type).strip().upper() == "H":
        return False
    if hasattr(atom, "element") and str(atom.element).strip().upper() == "H":
        return False
    return aname in ATOM_ORDER_INDEX


def extract_ca_atoms(u_obj: Any) -> Any:
    """Extract one C-alpha (or fallback) atom per protein residue."""
    import MDAnalysis as mda

    if hasattr(u_obj, "select_atoms"):
        res_q = " ".join(PROTEIN_RESNAMES)
        protein_atoms = u_obj.select_atoms(
            f"resname {res_q} and not (name H* or name [0-9]H* or type H or element H)"
        )
        target_residues = (
            protein_atoms.residues
            if len(protein_atoms) > 0
            else (u_obj.residues if hasattr(u_obj, "residues") else [])
        )
    else:
        target_residues = u_obj.residues if hasattr(u_obj, "residues") else []

    ca_list = []
    for res in target_residues:
        ca_in_res = res.atoms.select_atoms("name CA Ca ca CA* C-alpha Calpha")
        if len(ca_in_res) > 0:
            ca_list.append(ca_in_res[0])
        elif len(res.atoms) >= 2:
            ca_list.append(res.atoms[1])
        elif len(res.atoms) > 0:
            ca_list.append(res.atoms[0])

    if len(ca_list) > 0:
        return mda.AtomGroup(ca_list)
    return u_obj.atoms if hasattr(u_obj, "atoms") else u_obj


def precompute_atom_indexer(atom_group: Any):
    """Pre-compute direct numpy slice or fancy indexing array from an AtomGroup."""
    if hasattr(atom_group, "indices"):
        indices = atom_group.indices
    elif isinstance(atom_group, slice):
        return atom_group
    else:
        indices = np.asarray(atom_group)

    if len(indices) == 0:
        return slice(0, 0)
    if (indices[-1] - indices[0] + 1 == len(indices)) and np.all(np.diff(indices) == 1):
        return slice(int(indices[0]), int(indices[-1] + 1))
    return indices


def extract_group_coords(u_ds: Any, atom_group: Any, out: np.ndarray = None) -> np.ndarray:
    """Extract (n_frames, n_atoms, 3) float32 coordinate array for an atom group."""
    n_frames = len(u_ds.trajectory)
    if isinstance(atom_group, slice):
        indexer = atom_group
        n_atoms = max(0, indexer.stop - indexer.start)
    elif isinstance(atom_group, np.ndarray):
        indexer = atom_group
        n_atoms = len(indexer)
    else:
        indexer = precompute_atom_indexer(atom_group)
        n_atoms = (indexer.stop - indexer.start) if isinstance(indexer, slice) else len(indexer)

    if n_frames == 0 or n_atoms == 0:
        return np.empty((n_frames, n_atoms, 3), dtype=np.float32) if out is None else out
    coords = np.empty((n_frames, n_atoms, 3), dtype=np.float32) if out is None else out

    for idx_f, ts in enumerate(u_ds.trajectory):
        coords[idx_f] = ts.positions[indexer]
    return coords
