#!/bin/bash
set -e

echo "======================================================="
echo "   MET Workflow - SMART UNIVERSAL INSTALLER (v2.7 MERGED)"
echo "======================================================="
echo "This installer will automatically adapt to your hardware and username!"
echo "Requires ~15-20 GB of free storage."
read -r -p "Proceed? [Y/n]: " proceed
if [[ "$proceed" =~ ^[Nn] ]]; then exit 0; fi

# 1. Locate or Install Micromamba
MAMBA_BIN=$(command -v micromamba || echo "$HOME/.local/bin/micromamba")
if [ ! -f "$MAMBA_BIN" ]; then
    MAMBA_BIN=$(grep "export MAMBA_EXE=" ~/.bashrc 2>/dev/null | tail -n 1 | cut -d"'" -f2 || echo "")
fi

if [ -z "$MAMBA_BIN" ] || [ ! -f "$MAMBA_BIN" ]; then
    echo "📦 Installing Micromamba..."
    export INIT_YES=no
    export BIN_FOLDER="$HOME/.local/bin"
    mkdir -p "$HOME/.local/bin"
    if command -v curl &> /dev/null; then
        curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba && mv bin/micromamba "$HOME/.local/bin/micromamba"
    elif command -v wget &> /dev/null; then
        wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba && mv bin/micromamba "$HOME/.local/bin/micromamba"
    else
        echo "❌ Error: Neither 'curl' nor 'wget' is installed. Please install one of them and run the installer again."
        exit 1
    fi
    MAMBA_BIN="$HOME/.local/bin/micromamba"
fi

# 2. Setup Environment
if [ -n "$MAMBA_BIN" ] && [ -f "$MAMBA_BIN" ]; then
    eval "$("$MAMBA_BIN" shell hook --shell=bash)"
else
    echo "❌ Error: Could not locate micromamba executable."
    exit 1
fi

if micromamba run -n conf_sel_gen python --version &> /dev/null; then
    echo "✅ Environment 'conf_sel_gen' already exists and is healthy. Skipping creation."
else
    echo "📦 Creating 'conf_sel_gen' environment..."
    micromamba create -y -n conf_sel_gen python=3.10.14
fi

if micromamba run -n conf_sel_analysis python --version &> /dev/null; then
    echo "✅ Environment 'conf_sel_analysis' already exists and is healthy. Skipping creation."
else
    echo "📦 Creating 'conf_sel_analysis' environment..."
    micromamba create -y -n conf_sel_analysis python=3.10.14
fi

# 3. Detect GPU and Install Smart Dependencies
echo "🔍 Analyzing system hardware and environment..."

if micromamba run -n conf_sel_gen python -c "import bioemu, torch" &> /dev/null; then
    echo "⚡ Required packages (PyTorch, BioEmu) already detected in 'conf_sel_gen'! Skipping heavy installations..."
else
    TORCH_INSTALL_CMD="pip install --no-cache-dir torch==2.6.0"

    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        echo "✅ Detected GPU: $GPU_NAME"
        if [[ "$GPU_NAME" == *"RTX 50"* || "$GPU_NAME" == *"RTX 60"* || "$GPU_NAME" == *"Blackwell"* ]]; then
            TORCH_INSTALL_CMD="pip install --no-cache-dir torch==2.6.0"
        fi
    fi

    echo "📦 Installing PyTorch..."
    micromamba run -n conf_sel_gen $TORCH_INSTALL_CMD

    echo "📦 Installing C-libraries via Conda to prevent Pip compilation errors for TensorFlow..."
    micromamba install -n conf_sel_gen -y -c conda-forge h5py==3.10.0

    echo "📦 Installing TensorFlow (ignoring upcoming Pip warnings)..."
    micromamba run -n conf_sel_gen pip install --no-cache-dir "tensorflow-cpu==2.14.0" "tensorboard==2.14.0" "protobuf==3.20.3"

    echo "📦 Installing BioEmu and JAX..."
    micromamba run -n conf_sel_gen pip install --no-cache-dir "bioemu==1.3.1" "dm-haiku==0.0.12" "numpy==1.26.4" "jax[cuda12]==0.4.35" || true

fi

echo "📦 Ensuring dm-haiku is updated..."
micromamba run -n conf_sel_gen pip install --no-cache-dir --upgrade --no-deps "dm-haiku==0.0.12"

echo "🩹 Applying JAX & Haiku hotfixes..."
micromamba run -n conf_sel_gen python - << 'EOF_PATCH'
import os
import pathlib
file_path = os.path.expanduser('~/micromamba/envs/conf_sel_gen/lib/python3.10/site-packages/jax/_src/lib/__init__.py')
if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        code = f.read()
    target = 'cuda_nvcc_path = pathlib.Path(cuda_nvcc.__file__).parent'
    replacement = 'cuda_nvcc_path = pathlib.Path(cuda_nvcc.__file__ or (list(cuda_nvcc.__path__)[0] + "/__init__.py")).parent'
    if target in code:
        with open(file_path, 'w') as f:
            f.write(code.replace(target, replacement))

file_path_haiku = os.path.expanduser('~/micromamba/envs/conf_sel_gen/lib/python3.10/site-packages/haiku/_src/dot.py')
if os.path.exists(file_path_haiku):
    with open(file_path_haiku, 'r') as f:
        code = f.read()
    target_haiku = 'take_current_trace = jax.core.take_current_trace'
    replacement_haiku = 'take_current_trace = getattr(jax.core, "take_current_trace", None)'
    if target_haiku in code:
        with open(file_path_haiku, 'w') as f:
            f.write(code.replace(target_haiku, replacement_haiku))
EOF_PATCH

if micromamba run -n conf_sel_analysis python -c "import Bio, MDAnalysis" &> /dev/null; then
    echo "⚡ Required packages already detected in 'conf_sel_analysis'! Skipping pip install..."
else
    echo "📦 Installing Analysis dependencies via Conda-Forge to prevent Pip C-compilation errors..."
    micromamba install -n conf_sel_analysis -y -c conda-forge biopython==1.83 mdanalysis==2.7.0 pandas==2.2.0 scikit-learn==1.4.0 scipy==1.12.0 tqdm==4.66.1 openmm==8.1.1 pdbfixer==1.9 setuptools==69.0.0
fi

# 4. Create Scripts
echo "======================================================="
read -r -p "Install directory [Default: $HOME/metflow_pipeline]: " user_install_dir
INSTALL_DIR="${user_install_dir:-$HOME/metflow_pipeline}"
INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"
# Force absolute path to prevent relative PATH bugs in ~/.bashrc
mkdir -p "$INSTALL_DIR"
INSTALL_DIR=$(realpath "$INSTALL_DIR")

echo "📝 Writing scripts to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR/bin"

# ----- PYTHON BRAIN: run_pipeline.py -----
cat << 'EOF' > "$INSTALL_DIR/run_pipeline.py"
import os, sys, glob

# MOCK: Bypass TensorFlow 2.14's strict requirement for Protobuf 4.x runtime_version
import google.protobuf
if not hasattr(google.protobuf, 'runtime_version'):
    class DummyRuntimeVersion:
        Domain = type('Domain', (), {'PUBLIC': 1})
        def ValidateProtobufRuntimeVersion(*args, **kwargs):
            pass
    google.protobuf.runtime_version = DummyRuntimeVersion
    sys.modules['google.protobuf.runtime_version'] = DummyRuntimeVersion

try:
    import nvidia.cuda_nvcc
    if getattr(nvidia.cuda_nvcc, "__file__", None) is None:
        path_list = list(nvidia.cuda_nvcc.__path__)
        if path_list:
            nvidia.cuda_nvcc.__file__ = os.path.join(path_list[0], "__init__.py")
except Exception:
    pass

from Bio import PDB
from Bio.SeqUtils import seq1
try:
    from bioemu.sample import main as bioemu_main
except ImportError as e:
    print(e)
    sys.exit(1)

pdb_files = glob.glob("*.pdb")
if not pdb_files:
    sys.exit(1)
pdb_file = pdb_files[0]

parser = PDB.PDBParser(QUIET=True)
structure = parser.get_structure("protein", pdb_file)
sequence = "".join([seq1(res.resname) for res in list(list(structure)[0])[0] if PDB.is_aa(res, standard=True)])

kwargs = {"sequence": sequence, "num_samples": 2000, "output_dir": "./bioemu_ensemble"}

for i in range(1, len(sys.argv)):
    arg = sys.argv[i]
    if arg.startswith("--") and "=" in arg:
        key, val = arg[2:].split("=", 1)
        kwargs[key] = val
    elif arg.startswith("--") and i + 1 < len(sys.argv) and not sys.argv[i+1].startswith("--"):
        key, val = arg[2:], sys.argv[i+1]
        kwargs[key] = int(val) if val.isdigit() else float(val) if "." in val else val

num_samples = int(kwargs.pop("num_samples", 2000))
denoising_steps = int(kwargs.pop("denoiser_steps", 50))
denoiser_type = str(kwargs.pop("denoiser_type", "dpm")).strip()

solver_target_map = {"dpm": "bioemu.shortcuts.dpm_solver", "euler": "bioemu.shortcuts.euler_maruyama_solver"}
target = solver_target_map.get(denoiser_type, "bioemu.shortcuts.dpm_solver")
kwargs["denoiser_config"] = {"_target_": target, "_partial_": True, "eps_t": 0.001, "max_t": 0.99, "N": denoising_steps, "noise": 0.0}
kwargs["denoiser_type"] = denoiser_type
kwargs["num_samples"] = num_samples

status_file = ".metflow_status"
try:
    with open(status_file, "w") as f:
        f.write("PID={}\nTARGET_SAMPLES={}\n".format(os.getpid(), num_samples))
except Exception:
    pass

try:
    bioemu_main(**kwargs)
finally:
    if os.path.exists(status_file):
        os.remove(status_file)
EOF

cat << 'EOF' > "$INSTALL_DIR/analyze_trajectory.py"
import sys
import os

# --- Fast Path for Option [D] (Dashboard ONLY) ---
# Generates dashboard instantly without importing heavy packages or running dependency checks
if "--only_dashboard" in sys.argv:
    try:
        idx = sys.argv.index("--only_dashboard")
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1].lower() == "true":
            out_dir = "clustering_results"
            if "--out_dir" in sys.argv:
                out_idx = sys.argv.index("--out_dir")
                if out_idx + 1 < len(sys.argv):
                    out_dir = sys.argv[out_idx + 1]
            out_dir_abs = os.path.abspath(out_dir)
            base_dir_abs = os.path.abspath('.')
            if not (out_dir_abs == base_dir_abs or out_dir_abs.startswith(base_dir_abs + os.sep)):
                print("Error: Directory traversal blocked")
                sys.exit(1)
            os.makedirs(out_dir_abs, exist_ok=True)
            with open(__file__, "r", encoding="utf-8") as f:
                content = f.read()
            try:
                start_marker = "<!DOC" + "TYPE html>"
                end_marker = "</ht" + "ml>"
                html = start_marker + content.split(start_marker)[1].split(end_marker)[0] + end_marker
            except IndexError:
                print("❌ Error: Could not find HTML block in script source.")
                sys.exit(1)
            dash_file = os.path.join(out_dir_abs, "metflow_dashboard.html")
            with open(dash_file, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"\n✅ Dashboard generated instantly at {dash_file}")
            sys.exit(0)
    except Exception as e:
        pass

import subprocess

# Auto-install analysis dependencies if missing (Fast in-process check)
def install_deps():
    mod_map = {
        "scikit-learn": "sklearn",
        "scipy": "scipy",
        "MDAnalysis": "MDAnalysis",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "setuptools": "pkg_resources"
    }
    pins = {
        "scikit-learn": "scikit-learn>=1.0.0",
        "scipy": "scipy>=1.8.0",
        "MDAnalysis": "MDAnalysis>=2.0.0",
        "pandas": "pandas>=1.5.0",
        "matplotlib": "matplotlib>=3.5.0",
        "seaborn": "seaborn>=0.12.0",
        "setuptools": "setuptools>=60.0.0"
    }
    missing = []
    for pkg_name, mod_name in mod_map.items():
        try:
            __import__(mod_name)
        except ImportError:
            missing.append(pins[pkg_name])
    if missing:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + missing, check=True)
        except Exception:
            fallback = [m.split(">=")[0] for m in missing]
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + fallback, check=True)
            except Exception as e:
                print(f"⚠️ Dependency notice: {e}")
                print(f"👉 Please make sure the MetFlow environment is active: micromamba activate conf_sel_gen")

install_deps()

import argparse
import numpy as np
import pandas as pd
import MDAnalysis as mda
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import SequenceMatcher
from MDAnalysis.analysis import align
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, pairwise_distances
from scipy.spatial import ConvexHull
import datetime
import warnings
from multiprocessing import Pool

warnings.filterwarnings('ignore')

def _calc_vol(coords):
    try: return ConvexHull(coords).volume
    except Exception: return 0.0

def batched_kabsch(P_raw, Q_ref, out=None, chunk_size=1024):
    n_samples = len(P_raw)
    if n_samples == 0:
        return P_raw
    if out is None:
        out = np.empty_like(P_raw)
    
    # Process in chunks to avoid duplicate full-array memory allocations
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = P_raw[start:end]
        c_mean = chunk.mean(axis=1, keepdims=True)
        c_centered = chunk - c_mean
        H = np.matmul(c_centered.swapaxes(1, 2), Q_ref)
        U, _, Vt = np.linalg.svd(H)
        R = np.matmul(U, Vt)
        dets = (R[:, 0, 0] * (R[:, 1, 1] * R[:, 2, 2] - R[:, 1, 2] * R[:, 2, 1])
              - R[:, 0, 1] * (R[:, 1, 0] * R[:, 2, 2] - R[:, 1, 2] * R[:, 2, 0])
              + R[:, 0, 2] * (R[:, 1, 0] * R[:, 2, 1] - R[:, 1, 1] * R[:, 2, 0]))
        neg_mask = dets < 0
        if np.any(neg_mask):
            U_corr = U.copy()
            U_corr[neg_mask, :, 2] *= -1
            R = np.matmul(U_corr, Vt)
        np.matmul(c_centered, R, out=out[start:end])
    return out

def parse_res_string(res_str):
    res_list = []
    for part in res_str.replace(',', ' ').split():
        if '-' in part:
            start, end = map(int, part.split('-'))
            res_list.extend(range(start, end + 1))
        else:
            res_list.append(int(part))
    return res_list

def parse_list_arg(arg_str):
    if not arg_str:
        return []
    return [x.strip() for x in arg_str.replace(',', ' ').split() if x.strip()]

parser = argparse.ArgumentParser()
parser.add_argument("--skim", type=str, default="false")
parser.add_argument("--equator", type=str, default="")
parser.add_argument("--top", type=str, default="")
parser.add_argument("--bottom", type=str, default="")
parser.add_argument("--min_vol", type=float, default=300.0)
parser.add_argument("--cluster", type=str, default="pca_kmeans")
parser.add_argument("--k", type=int, default=8)
parser.add_argument("--rmsd", type=float, default=2.0)
parser.add_argument("--cnn_eps", type=float, default=3.0)
parser.add_argument("--allostery", type=str, default="false")
parser.add_argument("--cavity_res", type=str, default="")
parser.add_argument("--loop_res", type=str, default="")
parser.add_argument("--pca_res", type=str, default="")
parser.add_argument("--advanced_plots", type=str, default="false")
parser.add_argument("--gen_params", type=str, default="")
parser.add_argument("--dataset_configs", type=str, default="")
parser.add_argument("--pdb", type=str, default="")
parser.add_argument("--xtc", type=str, default="")
parser.add_argument("--multi_datasets", type=str, default="false")
parser.add_argument("--dataset_labels", type=str, default="")
parser.add_argument("--out_dir", type=str, default="clustering_results")
parser.add_argument("--calc_energy", type=str, default="false")
parser.add_argument("--only_energy", type=str, default="false")
parser.add_argument("--only_dashboard", type=str, default="false")
parser.add_argument("--max_frames", type=int, default=0)
parser.add_argument("--min_steps", type=int, default=100)
parser.add_argument("--pca_ref", type=str, default="combined")
args = parser.parse_args()

out_dir_abs = os.path.abspath(args.out_dir)
base_dir_abs = os.path.abspath('.')
if not (out_dir_abs == base_dir_abs or out_dir_abs.startswith(base_dir_abs + os.sep)):
    print("Error: Directory traversal blocked")
    sys.exit(1)
args.out_dir = out_dir_abs

if args.only_dashboard.lower() == "true":
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(__file__, "r", encoding="utf-8") as f:
        content = f.read()
    try:
        start_marker = "<!DOC" + "TYPE html>"
        end_marker = "</ht" + "ml>"
        html = start_marker + content.split(start_marker)[1].split(end_marker)[0] + end_marker
    except IndexError:
        print("❌ Error: Could not find HTML block in script source.")
        sys.exit(1)
    dash_file = os.path.join(out_dir, "metflow_dashboard.html")
    with open(dash_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✅ Dashboard generated instantly at {dash_file}")
    sys.exit(0)

if args.only_energy.lower() == "true":
    args.skim = "false"
    args.allostery = "false"
    args.cluster = "none"
    args.calc_energy = "true"

print("\n==================================================")
print("   Initializing Post-Generation Analysis (v2.7 MERGED)")
print("==================================================")
print("--------------------------------------------------")

# Trajectory loading resolution logic
if args.dataset_configs:
    dataset_configs = []
    for entry in args.dataset_configs.split(';'):
        entry = entry.strip()
        if not entry:
            continue
        parts = [p.strip() for p in entry.split(',')]
        if len(parts) >= 2:
            p_pdb = parts[0]
            p_xtc = parts[1]
            p_lbl = parts[2] if len(parts) >= 3 else f"Dataset_{len(dataset_configs)+1}"
            dataset_configs.append((p_pdb, p_xtc, p_lbl))
elif args.multi_datasets.lower() == "true" or (len(parse_list_arg(args.pdb)) > 1 and len(parse_list_arg(args.xtc)) > 1):
    pdb_list = parse_list_arg(args.pdb)
    xtc_list = parse_list_arg(args.xtc)
    label_list = parse_list_arg(args.dataset_labels)
    
    if len(pdb_list) != len(xtc_list):
        print(f"Error: Number of PDB files ({len(pdb_list)}) does not match XTC files ({len(xtc_list)}).")
        sys.exit(1)
        
    while len(label_list) < len(pdb_list):
        label_list.append(f"Dataset_{len(label_list) + 1}")
        
    dataset_configs = list(zip(pdb_list, xtc_list, label_list))
else:
    if args.pdb or args.xtc:
        pdb_path = parse_list_arg(args.pdb)[0] if parse_list_arg(args.pdb) else "./bioemu_ensemble/topology.pdb"
        xtc_path = parse_list_arg(args.xtc)[0] if parse_list_arg(args.xtc) else "./bioemu_ensemble/samples.xtc"
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
    ds_label = parse_list_arg(args.dataset_labels)[0] if parse_list_arg(args.dataset_labels) else "BioEmu_Ensemble"
    dataset_configs = [(pdb_path, xtc_path, ds_label)]

master_ref_pdb = dataset_configs[0][0]
if not os.path.exists(master_ref_pdb):
    print(f"Error: Could not find master reference topology PDB: {master_ref_pdb}")
    sys.exit(1)

u_master = mda.Universe(master_ref_pdb)
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

# Save run parameters to out_dir/metflow_params.txt and working directory metflow_params.txt
param_str = "\n".join(f"{arg_k}={arg_v}" for arg_k, arg_v in vars(args).items()) + "\n"
with open(os.path.join(out_dir, "metflow_params.txt"), "w", encoding="utf-8") as f_param:
    f_param.write(param_str)
if out_dir != base_dir_abs:
    try:
        with open("metflow_params.txt", "w", encoding="utf-8") as f_param2:
            f_param2.write(param_str)
    except Exception:
        pass

methodology = [f"=== MET Pipeline Methodology ===\nDate: {datetime.datetime.now()}"]
if args.gen_params:
    methodology.append(f"Generation Parameters:\n{args.gen_params}")

PROTEIN_RESNAMES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'HIE', 'HID', 'HIP', 'HSD', 'HSE', 'HSP', 'CYX', 'CYM', 'MSE'
}

def extract_ca_atoms(u_obj):
    if hasattr(u_obj, 'select_atoms'):
        res_q = " ".join(PROTEIN_RESNAMES)
        protein_atoms = u_obj.select_atoms(f"resname {res_q} and not (name H* or type H)")
        target_residues = protein_atoms.residues if len(protein_atoms) > 0 else (u_obj.residues if hasattr(u_obj, 'residues') else [])
    else:
        target_residues = u_obj.residues if hasattr(u_obj, 'residues') else []
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
    return u_obj.atoms if hasattr(u_obj, 'atoms') else u_obj

def precompute_atom_indexer(atom_group):
    """
    Pre-compute direct numpy slice or fancy indexing array from an AtomGroup.
    Returns slice(start, end) if contiguous, or numpy index array if non-contiguous.
    """
    if hasattr(atom_group, 'indices'):
        indices = atom_group.indices
    elif isinstance(atom_group, slice):
        return atom_group
    elif isinstance(atom_group, (np.ndarray, list, tuple)):
        indices = np.asarray(atom_group)
    else:
        indices = np.asarray(atom_group)

    if len(indices) == 0:
        return slice(0, 0)
    if (indices[-1] - indices[0] + 1 == len(indices)) and np.all(np.diff(indices) == 1):
        return slice(int(indices[0]), int(indices[-1] + 1))
    return indices

def extract_group_coords(u_ds, atom_group, out=None):
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
    if out is None:
        coords = np.empty((n_frames, n_atoms, 3), dtype=np.float32)
    else:
        coords = out

    for idx_f, ts in enumerate(u_ds.trajectory):
        coords[idx_f] = ts.positions[indexer]
    return coords

# Pre-validate PDB/XTC files, initialize universes, and filter non-protein atoms across datasets
temp_universes = []
prot_res_str = " ".join(PROTEIN_RESNAMES)
for pdb_p, xtc_p, label in dataset_configs:
    if not os.path.exists(pdb_p):
        print(f"Error: PDB file not found: {pdb_p}")
        sys.exit(1)
    if not os.path.exists(xtc_p):
        print(f"Error: XTC file not found: {xtc_p}")
        sys.exit(1)
    u_temp = mda.Universe(pdb_p, xtc_p)
    # Whitelist standard and common modified protein residues; strip waters, ions, ligands, explicit hydrogens
    protein_atoms = u_temp.select_atoms(f"resname {prot_res_str} and not (name H* or type H)")
    if len(protein_atoms) == 0:
        raise ValueError(f"No protein residues matching PROTEIN_RESNAMES found in {pdb_p}.")
    ca_group = extract_ca_atoms(protein_atoms)
    prot_res = protein_atoms.residues
    temp_universes.append((pdb_p, xtc_p, label, u_temp, ca_group, prot_res))

is_only_energy = (args.only_energy.lower() == "true")
if is_only_energy:
    print("\n--- Option [C]: Fast Energy ONLY Calculation Mode (Consensus-Aligned) ---")

# Determine common reference matching bounds across all datasets using fast sequence matching & caching
RESNAME_MAP = {
    'HIE': 'HIS', 'HID': 'HIS', 'HIP': 'HIS', 'HSD': 'HIS', 'HSE': 'HIS', 'HSP': 'HIS',
    'CYX': 'CYS', 'CYM': 'CYS', 'MSE': 'MET'
}
def normalize_resname(name):
    n = name.upper()
    return RESNAME_MAP.get(n, n)

AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}
def res_to_1letter(resname):
    norm = normalize_resname(resname)
    return AA_3TO1.get(norm, 'X')

dataset_seqs = []
for _, _, label, _, _, prot_res in temp_universes:
    s_d = "".join(res_to_1letter(r.resname) for r in prot_res)
    dataset_seqs.append(s_d)

pair_ident_cache = {}
reusable_pair_matcher = SequenceMatcher(None, "", "")
for i in range(len(dataset_seqs)):
    for j in range(i + 1, len(dataset_seqs)):
        s1, s2 = dataset_seqs[i], dataset_seqs[j]
        l1, l2 = temp_universes[i][2], temp_universes[j][2]
        if s1 == s2:
            identity = 1.0
        elif (s1, s2) in pair_ident_cache:
            identity = pair_ident_cache[(s1, s2)]
        elif (s2, s1) in pair_ident_cache:
            identity = pair_ident_cache[(s2, s1)]
        else:
            max_len = max(len(s1), len(s2))
            min_len = min(len(s1), len(s2))
            if max_len == 0:
                identity = 0.0
            elif min_len / max_len < 0.95:
                identity = min_len / max_len
            elif len(s1) == len(s2) and (sum(1 for a, b in zip(s1, s2) if a == b) / max_len) >= 0.95:
                identity = sum(1 for a, b in zip(s1, s2) if a == b) / max_len
            else:
                reusable_pair_matcher.set_seqs(s1, s2)
                matching_blocks = reusable_pair_matcher.get_matching_blocks()
                identical_count = sum(b.size for b in matching_blocks)
                identity = identical_count / max_len if max_len > 0 else 0.0
            pair_ident_cache[(s1, s2)] = identity
            pair_ident_cache[(s2, s1)] = identity
        if identity < 0.95:
            raise ValueError(f"Sequence similarity across datasets '{l1}' and '{l2}' is {identity*100:.1f}%, which is below the 95% threshold.")

ref_prot_res = temp_universes[0][5]
ref_seq = tuple(normalize_resname(r.resname) for r in ref_prot_res)
ref_seq_len = len(ref_seq)

seq_match_cache = {}
dataset_matches = []
reusable_ref_matcher = SequenceMatcher(None, ref_seq, None)
for _, _, label, _, _, prot_res in temp_universes:
    ds_seq = tuple(normalize_resname(r.resname) for r in prot_res)
    if ds_seq == ref_seq:
        match_tuple = (0, ref_seq_len, 0, ref_seq_len)
    elif ds_seq in seq_match_cache:
        match_tuple = seq_match_cache[ds_seq]
    else:
        reusable_ref_matcher.set_seq2(ds_seq)
        match = reusable_ref_matcher.find_longest_match(0, ref_seq_len, 0, len(ds_seq))
        if match.size == 0:
            raise ValueError(f"No matching residue sequence found between reference and dataset '{label}'!")
        match_tuple = (match.a, match.a + match.size, match.b, match.b + match.size)
        seq_match_cache[ds_seq] = match_tuple
    dataset_matches.append(match_tuple)

consensus_ref_start = max(m[0] for m in dataset_matches)
consensus_ref_end = min(m[1] for m in dataset_matches)
consensus_len = consensus_ref_end - consensus_ref_start

if consensus_len < 10:
    raise ValueError(f"Common homologous protein sequence length across datasets is {consensus_len} residues, which is below the minimum required 10 residues.")

CANONICAL_ATOM_ORDER = [
    'N', 'CA', 'C', 'O',
    'CB', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG',
    'CD', 'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2',
    'CE', 'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2',
    'CZ', 'CZ2', 'CZ3', 'NZ', 'CH2', 'NH1', 'NH2', 'OH', 'OXT'
]
ATOM_ORDER_INDEX = {name: idx for idx, name in enumerate(CANONICAL_ATOM_ORDER)}

def canonical_atom_sort_key(atom_name):
    clean = atom_name.strip().upper()
    return (ATOM_ORDER_INDEX.get(clean, 999), clean)

ds_residues_homologous = []
for idx, (_, _, _, _, _, prot_res) in enumerate(temp_universes):
    ref_start, _, ds_start, _ = dataset_matches[idx]
    ds_offset_start = ds_start + (consensus_ref_start - ref_start)
    ds_residues_homologous.append(prot_res[ds_offset_start : ds_offset_start + consensus_len])

ds_consensus_atom_lists = [[] for _ in range(len(temp_universes))]

for r in range(consensus_len):
    res_atoms_per_ds = []
    for idx in range(len(temp_universes)):
        res_r = ds_residues_homologous[idx][r]
        heavy_atoms = [a for a in res_r.atoms if not (a.name.strip().upper().startswith('H') or (hasattr(a, 'type') and a.type.strip().upper() == 'H'))]
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

def make_alias(src, dst):
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if os.path.exists(dst) or os.path.islink(dst):
        try:
            os.remove(dst)
        except Exception:
            pass
    try:
        os.symlink(os.path.basename(src), dst)
    except Exception:
        import shutil
        shutil.copyfile(src, dst)

ref_atoms = ds_consensus_groups[0]
u_consensus = mda.Merge(ref_atoms)

for i, res in enumerate(u_consensus.residues, start=1):
    res.resid = i
for i, atom in enumerate(u_consensus.atoms, start=1):
    atom.id = i

ref_consensus_pos = ref_atoms.positions.copy()
ref_consensus_mean = ref_consensus_pos - ref_consensus_pos.mean(axis=0)
u_consensus.atoms.positions = ref_consensus_mean

merged_ref_path = os.path.join(out_dir, "merged_reference.pdb")
unified_ref_path = os.path.join(out_dir, "unified_reference.pdb")
u_consensus.atoms.write(merged_ref_path)
print(f"  -> Exported consensus reference PDB to {merged_ref_path}")
make_alias(merged_ref_path, unified_ref_path)
print(f"  -> Created unified reference alias at {unified_ref_path}")

ca_mask = np.array([a.name.strip().upper() == 'CA' for a in u_consensus.atoms])
ca_indices = np.where(ca_mask)[0]

if args.pca_res:
    pca_res_list = set(parse_res_string(args.pca_res))
    rel_indices = [
        i for i in range(len(ca_indices))
        if (i + 1) in pca_res_list or u_consensus.residues[i].resid in pca_res_list
    ]
    if not rel_indices:
        rel_indices = list(range(min(len(pca_res_list), len(ca_indices))))
    pca_atom_indices = ca_indices[rel_indices]
    methodology.append(f"Alignment and Global PCA restricted to {len(rel_indices)} C-alpha residues from: {args.pca_res}")
else:
    rel_indices = list(range(len(ca_indices)))
    pca_atom_indices = ca_indices
    methodology.append(f"Alignment and Global PCA used {len(ca_indices)} common C-alpha backbone atoms.")

ref_bb_atoms = u_consensus.atoms[pca_atom_indices]
ref_bb = ref_consensus_mean[pca_atom_indices]

offset_info = []
for idx, (pdb_p, xtc_p, label, _, _, _) in enumerate(temp_universes):
    ref_start, _, ds_start, _ = dataset_matches[idx]
    offset = ref_start - ds_start
    if offset != 0:
        offset_info.append(f"{label} offset {offset:+} detected and aligned")

offset_str = f" ({', '.join(offset_info)})" if offset_info else ""
print(f"Aligning frames across {len(dataset_configs)} dataset(s) in memory...")
print(f"  -> Dynamic sequence alignment matched {consensus_len} residues ({len(u_consensus.atoms)} heavy atoms) across datasets{offset_str}.")

total_frames = sum(len(u_ds.trajectory) for _, _, _, u_ds, _, _ in temp_universes)
n_consensus_atoms = len(u_consensus.atoms)
all_consensus_coords = np.empty((total_frames, n_consensus_atoms, 3), dtype=np.float32)
dataset_tags = []
frame_universe_map = []
universes = []

print(f"\n--- Loading and Aligning {len(dataset_configs)} Trajectory Dataset(s) ---")
curr_frame_offset = 0
ref_rg = np.mean(np.linalg.norm(ref_consensus_mean, axis=1))

ds_consensus_indexers = [precompute_atom_indexer(ag) for ag in ds_consensus_groups]

for idx, (pdb_p, xtc_p, label, u_ds, ca_g, prot_res) in enumerate(temp_universes):
    print(f"Loading dataset '{label}' (PDB: {pdb_p}, XTC: {xtc_p})...")
    universes.append(u_ds)
    n_ds_frames = len(u_ds.trajectory)
    raw_slice = all_consensus_coords[curr_frame_offset : curr_frame_offset + n_ds_frames]
    extract_group_coords(u_ds, ds_consensus_indexers[idx], out=raw_slice)
    
    if n_ds_frames > 0:
        ca_centered = raw_slice[0] - raw_slice[0].mean(axis=0)
        ds_rg = np.mean(np.linalg.norm(ca_centered, axis=1))
        
        if ref_rg > 5.0 and ds_rg > 0 and ds_rg < 5.0:
            print(f"  -> Auto-detected nanometer (nm) coordinate scale for dataset '{label}' (Rg={ds_rg:.2f} nm vs ref {ref_rg:.2f} A). Auto-scaling to Angstroms (x10)...")
            raw_slice *= 10.0
        elif ref_rg < 5.0 and ds_rg > 5.0:
            print(f"  -> Auto-detected nanometer (nm) coordinate scale for reference (Rg={ref_rg:.2f} nm). Auto-scaling dataset '{label}' (/10)...")
            raw_slice /= 10.0
            
    dataset_tags.extend([label] * n_ds_frames)
    for local_idx in range(n_ds_frames):
        frame_universe_map.append((u_ds, local_idx))
    curr_frame_offset += n_ds_frames

if total_frames > 0:
    batched_kabsch(all_consensus_coords, ref_consensus_mean, out=all_consensus_coords)

n_frames_initial = total_frames
print(f"Loaded {n_frames_initial} total conformations across {len(dataset_configs)} dataset(s).")
methodology.append(f"Loaded {len(dataset_configs)} dataset(s). Total conformations: {n_frames_initial}")

valid_frames = np.arange(n_frames_initial)

# --- SKIMMING MODULE ---
if args.skim.lower() == "true" and args.equator and args.top and args.bottom:
    eq_res = [int(x) for x in args.equator.replace(',', ' ').split()]
    top_res, bot_res = int(args.top.replace(',', '')), int(args.bottom.replace(',', ''))
    all_skim_res = eq_res + [top_res, bot_res]
    res_str = " ".join([str(x) for x in all_skim_res])
    
    print(f"\n--- Starting Tandem Skimming ---")
    
    skim_indexers = []
    top_indexers = []
    bot_indexers = []
    for u_ds in universes:
        s_atoms = u_ds.select_atoms(f"resid {res_str} and name CA")
        t_atom = u_ds.select_atoms(f"resid {top_res} and name CA")
        b_atom = u_ds.select_atoms(f"resid {bot_res} and name CA")
        if len(t_atom) == 0 or len(b_atom) == 0:
            print("Error: Top or Bottom residue CA not found. Aborting skim.")
            sys.exit(1)
        skim_indexers.append(precompute_atom_indexer(s_atoms))
        top_indexers.append(precompute_atom_indexer(t_atom))
        bot_indexers.append(precompute_atom_indexer(b_atom))
        
    skim_coords_list = []
    top_coords_list = []
    bot_coords_list = []
    for idx, u_ds in enumerate(universes):
        s_coords = extract_group_coords(u_ds, skim_indexers[idx])
        t_coords = extract_group_coords(u_ds, top_indexers[idx])[:, 0, :]
        b_coords = extract_group_coords(u_ds, bot_indexers[idx])[:, 0, :]
        skim_coords_list.append(s_coords)
        top_coords_list.append(t_coords)
        bot_coords_list.append(b_coords)
        
    skim_coords_all = np.concatenate(skim_coords_list, axis=0)
    top_coords_all = np.concatenate(top_coords_list, axis=0)
    bot_coords_all = np.concatenate(bot_coords_list, axis=0)
    heights = np.linalg.norm(top_coords_all - bot_coords_all, axis=1)
    
    print("Computing volumes with multiprocessing...")
    with Pool() as p:
        vols = np.array(p.map(_calc_vol, skim_coords_all))
        
    passed_mask = (vols > args.min_vol) & (heights > 8.0)
    passed_frames = np.where(passed_mask)[0].tolist()
    valid_frames = np.array(passed_frames)
    print(f"Skimming Complete: {len(valid_frames)} / {n_frames_initial} passed.")
    methodology.append(f"Tandem skim (Volume > {args.min_vol}, Dist > 8.0). Surviving: {len(valid_frames)}")
    
    if len(valid_frames) > 0:
        u_first, l_first = frame_universe_map[valid_frames[0]]
        u_first.trajectory[l_first]
        skim_atoms_first = u_first.select_atoms(f"resid {res_str} and name CA")
        hull = ConvexHull(skim_atoms_first.positions)
        verts = skim_atoms_first.positions
        cgo = ["from pymol.cgo import *", "from pymol import cmd", "obj = []"]
        for s in hull.simplices:
            v1, v2, v3 = verts[s[0]], verts[s[1]], verts[s[2]]
            cgo.append(f"obj.extend([BEGIN, TRIANGLES, COLOR, 0.4, 0.8, 0.8, VERTEX, {v1[0]:.3f}, {v1[1]:.3f}, {v1[2]:.3f}, VERTEX, {v2[0]:.3f}, {v2[1]:.3f}, {v2[2]:.3f}, VERTEX, {v3[0]:.3f}, {v3[1]:.3f}, {v3[2]:.3f}, END])")
        cgo.extend(["cmd.load_cgo(obj, 'internal_volume_mesh')", "cmd.set('cgo_transparency', 0.4, 'internal_volume_mesh')"])
        cgo.append(f"cmd.distance('gate_distance', 'resid {top_res} and name CA', 'resid {bot_res} and name CA')")
        with open(f"{out_dir}/visualize_hull.py", "w") as f:
            f.write("\n".join(cgo) + "\n")
        print(f"Generated PyMOL visualization script: {out_dir}/visualize_hull.py")
else:
    print("Skipped skimming.")
    methodology.append("\n--- Conformational Skimming ---\nNo skimming was applied.")

if len(valid_frames) < 2:
    print("Error: Too few valid frames remaining to perform clustering.")
    sys.exit(1)

unique_tags = list(dict.fromkeys(dataset_tags))
if args.max_frames > 0:
    dataset_tags_arr = np.array(dataset_tags)[valid_frames]
    sampled_parts = []
    for t in unique_tags:
        t_idxs = valid_frames[dataset_tags_arr == t]
        if len(t_idxs) > args.max_frames:
            st = max(1, len(t_idxs) // args.max_frames)
            sampled_parts.append(t_idxs[::st][:args.max_frames])
        else:
            sampled_parts.append(t_idxs)
            
    valid_frames = np.sort(np.concatenate(sampled_parts))
    print(f"\n[Per-Dataset Sampling] Uniformly sampled up to {args.max_frames} frames per dataset ({len(valid_frames)} total frames across {len(unique_tags)} dataset(s)).")
    methodology.append(f"Per-dataset sampling: up to {args.max_frames} frames per dataset ({len(valid_frames)} total).")
else:
    print(f"\n[No Downsampling] Preserved 100% of frames ({len(valid_frames)} total frames across {len(unique_tags)} dataset(s)).")
    methodology.append(f"No downsampling applied: 100% of frames retained ({len(valid_frames)} total).")

dataset_tags = [dataset_tags[i] for i in valid_frames]

# --- Physical Unified Trajectory Export ---
merged_xtc_path = os.path.join(out_dir, "merged_trajectory.xtc")
unified_xtc_path = os.path.join(out_dir, "unified_trajectory.xtc")

print(f"\n--- Exporting Physical Unified Trajectory ---")
print(f"Writing {len(valid_frames)} aligned frames ({len(u_consensus.atoms)} atoms per frame) to {merged_xtc_path}...")
with mda.Writer(merged_xtc_path, n_atoms=u_consensus.atoms.n_atoms) as w:
    for f_idx in valid_frames:
        u_consensus.atoms.positions = all_consensus_coords[f_idx]
        w.write(u_consensus.atoms)
print(f"  -> Exported merged trajectory to {merged_xtc_path}")
make_alias(merged_xtc_path, unified_xtc_path)
print(f"  -> Created unified trajectory alias at {unified_xtc_path}")

# Extract PCA/RMSD coordinates strictly for valid frames, then free full consensus buffer
coords_array = all_consensus_coords[valid_frames][:, pca_atom_indices, :]
del all_consensus_coords

# Calculate RMSD to Reference
diff_ref = coords_array - ref_bb
rmsd_to_ref = np.sqrt(np.einsum('nai,nai->n', diff_ref, diff_ref) / coords_array.shape[1])

# Calculate PCA
X = coords_array.reshape(len(valid_frames), -1)
pca_model = PCA(n_components=3)

if args.pca_ref.lower() in ["first", "1", "ref", "reference"] or (args.pca_ref and args.pca_ref.lower() != "combined"):
    target_tag = unique_tags[0]
    if args.pca_ref.lower() not in ["first", "1", "ref", "reference"]:
        for t in unique_tags:
            if args.pca_ref.lower() in t.lower():
                target_tag = t
                break
    fit_mask = np.array([t == target_tag for t in dataset_tags])
    if np.sum(fit_mask) >= 2:
        pca_model.fit(X[fit_mask])
        X_pca = pca_model.transform(X)
        print(f"\n[Reference PCA Projection] Fitted PCA model strictly on reference dataset '{target_tag}' ({np.sum(fit_mask)} frames) and projected all datasets into '{target_tag}'s PCA coordinate space.")
        methodology.append(f"Fitted PCA on reference dataset '{target_tag}' ({np.sum(fit_mask)} frames) and projected all datasets into reference PCA space.")
    else:
        X_pca = pca_model.fit_transform(X)
else:
    X_pca = pca_model.fit_transform(X)

var_explained = pca_model.explained_variance_ratio_ * 100
methodology.append(f"PCA Variance Explained: PC1={var_explained[0]:.1f}%, PC2={var_explained[1]:.1f}%, PC3={var_explained[2]:.1f}%")

df_data = {
    "Frame_Index": np.arange(len(valid_frames)),
    "Dataset": dataset_tags,
    "RMSD_to_Reference": rmsd_to_ref,
    "PCA_1": X_pca[:, 0],
    "PCA_2": X_pca[:, 1],
    "PCA_3": X_pca[:, 2]
}

# --- ADVANCED ALLOSTERY (DCCM & SUB-PCA) ---
if args.allostery.lower() == "true":
    print("\n--- Running Advanced Allostery Metrics ---")
    methodology.append("\n--- Advanced Allostery Metrics ---")
    
    print("Calculating Dynamic Cross-Correlation Matrix (DCCM)...")
    ca_coords = coords_array
    
    n_ca = ca_coords.shape[1]
    mean_coords = np.mean(ca_coords, axis=0)
    fluct = ca_coords - mean_coords
    
    fluct_flat = fluct.transpose(1, 0, 2).reshape(n_ca, -1)
    cov = (fluct_flat @ fluct_flat.T) / len(valid_frames)
            
    var = np.diag(cov)
    std = np.sqrt(np.maximum(var, 1e-12))
    dccm = cov / np.outer(std, std)
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(dccm, cmap="coolwarm", center=0, xticklabels=20, yticklabels=20)
    plt.title("Dynamic Cross-Correlation Matrix (DCCM)")
    plt.xlabel("Residue Index")
    plt.ylabel("Residue Index")
    plt.savefig(f"{out_dir}/dccm_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved DCCM Heatmap to {out_dir}/dccm_heatmap.png")
    methodology.append("Generated Dynamic Cross-Correlation Matrix (DCCM).")

    if args.cavity_res and args.loop_res:
        cav_res = parse_res_string(args.cavity_res)
        loop_res = parse_res_string(args.loop_res)
        
        cav_str = " ".join([str(x) for x in cav_res])
        loop_str = " ".join([str(x) for x in loop_res])
        
        cav_indexers = [precompute_atom_indexer(u_ds.select_atoms(f"resid {cav_str} and name CA")) for u_ds in universes]
        loop_indexers = [precompute_atom_indexer(u_ds.select_atoms(f"resid {loop_str} and name CA")) for u_ds in universes]
        
        cav_coords_list = [extract_group_coords(u_ds, cav_indexers[idx]) for idx, u_ds in enumerate(universes)]
        loop_coords_list = [extract_group_coords(u_ds, loop_indexers[idx]) for idx, u_ds in enumerate(universes)]
            
        cav_coords = np.concatenate(cav_coords_list, axis=0)[valid_frames]
        loop_coords = np.concatenate(loop_coords_list, axis=0)[valid_frames]
        
        if cav_coords.shape[1] > 0 and loop_coords.shape[1] > 0:
            print("Calculating Targeted Sub-PCA for Allosteric Linkage...")
            X_cav = cav_coords.reshape(len(valid_frames), -1)
            sub_pca_model = PCA(n_components=2)
            cav_pca = sub_pca_model.fit_transform(X_cav)
            sub_var_explained = sub_pca_model.explained_variance_ratio_ * 100
            
            mean_loop = np.mean(loop_coords, axis=0)
            loop_rmsd = np.sqrt(np.mean(np.sum((loop_coords - mean_loop)**2, axis=2), axis=1))
            
            plt.figure(figsize=(9, 7))
            scatter = plt.scatter(cav_pca[:, 0], cav_pca[:, 1], c=loop_rmsd, cmap="viridis", alpha=0.7, s=15)
            plt.colorbar(scatter, label="Loop RMSD to Mean (Å)")
            plt.title("Cavity PCA colored by Loop Fluctuation (Allostery Plot)")
            plt.xlabel(f"Cavity PC1 ({sub_var_explained[0]:.1f}%)")
            plt.ylabel(f"Cavity PC2 ({sub_var_explained[1]:.1f}%)")
            plt.savefig(f"{out_dir}/allostery_subpca_plot.png", dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved Allostery Plot to {out_dir}/allostery_subpca_plot.png")
            methodology.append(f"Targeted Sub-PCA applied to Cavity (res {cav_res}), colored by Loop RMSD (res {loop_res}). Variance: PC1={sub_var_explained[0]:.1f}%, PC2={sub_var_explained[1]:.1f}%")

            sub_n_atoms = cav_coords.shape[1]
            u_first, l_first = frame_universe_map[valid_frames[0]]
            u_first.trajectory[l_first]
            sub_resids = u_first.select_atoms(f"resid {cav_str} and name CA").resids
            sub_components = sub_pca_model.components_
            
            sub_atom_contrib_pc1 = np.sqrt(np.sum(sub_components[0].reshape(sub_n_atoms, 3)**2, axis=1))
            sub_atom_contrib_pc2 = np.sqrt(np.sum(sub_components[1].reshape(sub_n_atoms, 3)**2, axis=1))
            
            sub_unique_resids, sub_res_inv = np.unique(sub_resids, return_inverse=True)
            sub_res_contrib_pc1 = np.bincount(sub_res_inv, weights=sub_atom_contrib_pc1)
            sub_res_contrib_pc2 = np.bincount(sub_res_inv, weights=sub_atom_contrib_pc2)
            
            plt.figure(figsize=(8, 4))
            width = 0.35
            x = np.arange(len(sub_unique_resids))
            plt.bar(x - width/2, sub_res_contrib_pc1, width, label='PC1 Contribution', alpha=0.8, color='blue')
            plt.bar(x + width/2, sub_res_contrib_pc2, width, label='PC2 Contribution', alpha=0.8, color='orange')
            plt.xticks(x, sub_unique_resids)
            plt.xlabel('Cavity Residue Number')
            plt.ylabel('Loading Magnitude (RMS)')
            plt.title('Per-Residue Contribution to Cavity Sub-PCA')
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            plt.savefig(f'{out_dir}/allostery_subpca_loadings.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved Sub-PCA Loadings Plot to {out_dir}/allostery_subpca_loadings.png")

# --- CLUSTERING MODULE ---
if args.cluster != "none":
    print(f"\n--- Starting Clustering Modules ---")
    methodology.append("\n--- Clustering Analysis ---")

    methods_to_run = ["pca_kmeans", "daura", "cnn"] if args.cluster == "all" else [args.cluster]

    if args.cluster in ["daura", "cnn", "all"]:
        print("Calculating pairwise distance matrix...")
        N = len(valid_frames)
        rmsd_matrix = pairwise_distances(X, metric='euclidean', n_jobs=-1) / np.sqrt(coords_array.shape[1])
    else:
        N = len(valid_frames)

    def extract_c(pfx, c_idx, cid, out_dir):
        frame_idx = valid_frames[c_idx]
        u_curr, local_idx = frame_universe_map[frame_idx]
        u_curr.trajectory[local_idx]
        u_curr.atoms.write(f"{out_dir}/{pfx}_centroid_{cid}.pdb")

    for m in methods_to_run:
        
        if m == "pca_kmeans":
            best_k = args.k
            auto_det = False
            
            if args.k <= 0:
                print("\n[ PCA + KMeans ] Auto-detecting optimal 'k' using Silhouette Analysis...")
                auto_det = True
                best_score = -1.0
                best_k = 2
                max_test_k = min(15, len(X_pca) - 1)
                for k_test in range(2, max_test_k + 1):
                    labels_test = KMeans(n_clusters=k_test, random_state=42, n_init="auto").fit_predict(X_pca)
                    score = silhouette_score(X_pca, labels_test)
                    if score > best_score:
                        best_score = score
                        best_k = k_test
                print(f"  -> Optimal clusters detected: k = {best_k} (Silhouette Score: {best_score:.3f})")
            else:
                print(f"\n[ PCA + KMeans ] Running with explicit k={args.k}...")
                
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(X_pca)
            df_data["PCA_KMeans_Cluster"] = labels
            
            method_dir = f"{out_dir}/pca_kmeans"
            os.makedirs(method_dir, exist_ok=True)
            for k in range(best_k):
                pts = X_pca[labels == k]
                if len(pts) > 0:
                    c_idx = np.where(labels == k)[0][np.argmin(np.linalg.norm(pts - kmeans.cluster_centers_[k], axis=1))]
                    extract_c("pca_kmeans", c_idx, k, method_dir)
                    
            df_method = pd.DataFrame({"Frame_Index": np.arange(len(valid_frames)), "Dataset": dataset_tags, "RMSD_to_Ref": rmsd_to_ref, "PCA_1": X_pca[:,0], "PCA_2": X_pca[:,1], "PCA_3": X_pca[:,2], "Cluster_ID": labels})
            df_method.to_csv(f"{method_dir}/{m}_metrics.csv", index=False)
            methodology.append(f"PCA+KMeans (k={best_k}) used." + (" Optimal 'k' detected via Silhouette Score." if auto_det else ""))
            
        elif m == "daura":
            print(f"\n[ Daura ] Running Vectorized Daura Clustering with RMSD Cutoff {args.rmsd} Å...")
            unclustered = np.ones(N, dtype=bool)
            centroids = []
            
            neighborhoods = (rmsd_matrix < args.rmsd)
            while unclustered.any():
                counts = neighborhoods.dot(unclustered)
                counts[~unclustered] = -1
                best_c = np.argmax(counts)
                best_h = neighborhoods[best_c] & unclustered
                centroids.append(best_c)
                unclustered[best_h] = False
                if len(centroids) >= args.k:
                    break
                    
            labels = np.argmin(rmsd_matrix[:, centroids], axis=1)
            df_data["Daura_Cluster"] = labels
            
            method_dir = f"{out_dir}/daura"
            os.makedirs(method_dir, exist_ok=True)
            for i, c in enumerate(centroids):
                extract_c("daura", c, i, method_dir)
            
            df_method = pd.DataFrame({"Frame_Index": np.arange(len(valid_frames)), "Dataset": dataset_tags, "RMSD_to_Ref": rmsd_to_ref, "PCA_1": X_pca[:,0], "PCA_2": X_pca[:,1], "PCA_3": X_pca[:,2], "Cluster_ID": labels})
            df_method.to_csv(f"{method_dir}/{m}_metrics.csv", index=False)
            methodology.append(f"Daura greedy clustering (cutoff {args.rmsd} Å) extracted {len(centroids)} centroids.")
        
        elif m == "cnn":
            print(f"\n[ CNN ] Running Optimized DBSCAN with eps={args.cnn_eps}...")
            dbscan = DBSCAN(eps=args.cnn_eps, min_samples=3, metric="precomputed", n_jobs=-1)
            labels = dbscan.fit_predict(rmsd_matrix)
            
            df_data["CNN_Cluster"] = labels
            
            method_dir = f"{out_dir}/cnn"
            os.makedirs(method_dir, exist_ok=True)
            for k in set(labels):
                if k == -1:
                    continue
                idxs = np.where(labels == k)[0]
                sums = rmsd_matrix[np.ix_(idxs, idxs)].sum(axis=1)
                center = idxs[np.argmin(sums)]
                extract_c("cnn", center, k, method_dir)
                
            df_method = pd.DataFrame({"Frame_Index": np.arange(len(valid_frames)), "Dataset": dataset_tags, "RMSD_to_Ref": rmsd_to_ref, "PCA_1": X_pca[:,0], "PCA_2": X_pca[:,1], "PCA_3": X_pca[:,2], "Cluster_ID": labels})
            df_method.to_csv(f"{method_dir}/{m}_metrics.csv", index=False)
            methodology.append(f"CNN/DBSCAN: Density-based clustering (eps={args.cnn_eps}) extracted {len(set(labels)-{-1})} clusters.")

# --- ENERGY CALCULATION MODULE (OpenMM) ---
if args.calc_energy.lower() == "true":
    print("\n--- Calculating Thermodynamic Energy (OpenMM Physical Minimization) ---")
    methodology.append("\n--- OpenMM Energy Calculation ---")
    try:
        try:
            import pkg_resources
        except ImportError:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "setuptools==69.0.0"], check=True)
                import pkg_resources
            except Exception:
                pass
        from openmm.app import PDBFile, ForceField, Modeller, HBonds, NoCutoff
        import openmm as mm
        import openmm.unit as unit
        from tqdm import tqdm
        from pdbfixer import PDBFixer

        print("  -> Initializing OpenMM system from clean consensus reference...")
        merged_ref_path = os.path.join(out_dir, "merged_reference.pdb")
        if not os.path.exists(merged_ref_path):
            merged_ref_path = master_ref_pdb

        prot_res_str = " ".join(PROTEIN_RESNAMES)
        u_ref_clean = mda.Universe(merged_ref_path)
        u_heavy = u_ref_clean.select_atoms(f"resname {prot_res_str} and not (name H* or type H)")
        if len(u_heavy) == 0:
            u_heavy = u_ref_clean.select_atoms("not (name H* or type H or resname WAT or resname HOH or resname SOL or resname NA or resname CL)")
        
        resnames = list(u_heavy.residues.resnames)
        for i in range(len(resnames)):
            if resnames[i] in ['HIE', 'HID', 'HIP', 'HSD', 'HSE', 'HSP']:
                resnames[i] = 'HIS'
            elif resnames[i] in ['CYX', 'CYM']:
                resnames[i] = 'CYS'
            elif resnames[i] in ['MSE']:
                resnames[i] = 'MET'
        u_heavy.residues.resnames = resnames
        
        tmp_pdb = f"{out_dir}/tmp_clean_heavy.pdb"
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
                        if other.name.strip().upper() in ['CA', 'C', 'N']:
                            break
                parent_atom.append(anchor)

        fixed_to_orig_arr = np.array(fixed_to_orig, dtype=int)
        parent_atom_arr = np.array(parent_atom, dtype=int)
        
        fixed_mask = (fixed_to_orig_arr != -1)
        anchor_mask = (~fixed_mask) & (parent_atom_arr != -1)
        fallback_mask = (~fixed_mask) & (~anchor_mask)

        fixed_orig_idx = fixed_to_orig_arr[fixed_mask]
        anchor_parent_idx = parent_atom_arr[anchor_mask]

        num_fixer_atoms = fixer.topology.getNumAtoms()
        n_anchor = int(np.sum(anchor_mask))
        n_fallback = int(np.sum(fallback_mask))

        forcefield = ForceField('amber14-all.xml', 'implicit/obc2.xml')
        
        print("  -> Building reference system and compiling context...")
        ref_modeller = Modeller(fixer.topology, fixer.positions)
        ref_modeller.addHydrogens(forcefield)
        system = forcefield.createSystem(ref_modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
        
        integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
        
        try:
            platform = mm.Platform.getPlatformByName('CUDA')
            print("  -> 🚀 Using CUDA (GPU) for blazing fast OpenMM calculations!")
            simulation = mm.app.Simulation(ref_modeller.topology, system, integrator, platform)
        except Exception:
            platform = mm.Platform.getPlatformByName('CPU')
            print("  -> 🐢 Using CPU. (No CUDA detected). Calculations may be slow.")
            simulation = mm.app.Simulation(ref_modeller.topology, system, integrator, platform)
        
        ref_full_pos = np.array(ref_modeller.positions.value_in_unit(unit.nanometers), dtype=np.float32)
        ref_pos = np.array(fixer.positions.value_in_unit(unit.nanometers), dtype=np.float32)
        ref_pos_anchor = ref_pos[anchor_parent_idx] if n_anchor > 0 else None
        ref_pos_anchor_offset = (ref_pos[anchor_mask] - ref_pos_anchor) if n_anchor > 0 else None
        
        heavy_to_full = []
        full_atoms = list(ref_modeller.topology.atoms())
        full_atom_dict = {(a.residue.chain.index, a.residue.index, a.name.strip().upper()): i for i, a in enumerate(full_atoms)}
        for a in fixer.topology.atoms():
            key = (a.residue.chain.index, a.residue.index, a.name.strip().upper())
            heavy_to_full.append(full_atom_dict.get(key, -1))
            
        heavy_to_full_arr = np.array(heavy_to_full, dtype=int)
        valid_h2f_mask = (heavy_to_full_arr != -1)
        valid_h2f_src = np.where(valid_h2f_mask)[0]
        valid_h2f_dst = heavy_to_full_arr[valid_h2f_mask]
            
        # Map parent heavy atom for every hydrogen to prevent ghost clashes
        h_parent_map = {}
        for bond in ref_modeller.topology.bonds():
            a1, a2 = bond[0], bond[1]
            is_h1 = (a1.element.symbol == 'H')
            is_h2 = (a2.element.symbol == 'H')
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

        # Inspect if merged trajectory is available for direct coordinate streaming
        merged_xtc_path = os.path.join(out_dir, "merged_trajectory.xtc")
        if os.path.exists(merged_ref_path) and os.path.exists(merged_xtc_path):
            try:
                u_openmm = mda.Universe(merged_ref_path, merged_xtc_path)
                use_merged_traj = (len(u_openmm.trajectory) == len(valid_frames))
            except Exception:
                use_merged_traj = False
        else:
            use_merged_traj = False

        if not use_merged_traj:
            universe_heavy_info = {}
            for u_obj in set(u for u, _ in frame_universe_map):
                u_h = u_obj.select_atoms(f"resname {prot_res_str} and not (name H* or type H)")
                if len(u_h) == 0:
                    u_h = u_obj.select_atoms("not (name H* or type H or resname WAT or resname HOH or resname SOL or resname NA or resname CL)")
                h_indexer = precompute_atom_indexer(u_h)
                if len(u_h) == len(u_heavy):
                    map_idx = None
                else:
                    curr_map = {(a.resid, a.name): i for i, a in enumerate(u_h)}
                    map_idx = np.zeros(len(u_heavy), dtype=int)
                    for m_idx, m_atom in enumerate(u_heavy.atoms):
                        key = (m_atom.resid, m_atom.name)
                        if key in curr_map:
                            map_idx[m_idx] = curr_map[key]
                        else:
                            map_idx[m_idx] = min(m_idx, len(u_h) - 1)
                universe_heavy_info[id(u_obj)] = (h_indexer, map_idx)

        energy_csv = f"{out_dir}/energy_checkpoint.csv"
        energy_dict = {}

        def is_valid_energy(val):
            if pd.isna(val):
                return False
            try:
                f = float(val)
                return not (np.isnan(f) or np.isinf(f))
            except Exception:
                return False
        
        if os.path.exists(energy_csv):
            try:
                df_resume = pd.read_csv(energy_csv)
                ene_col = "Energy_kcal_mol" if "Energy_kcal_mol" in df_resume.columns else "Energy"
                raw_dict = dict(zip(df_resume["Frame"], df_resume[ene_col]))
                energy_dict = {f: v for f, v in raw_dict.items() if is_valid_energy(v)}
                print(f"  -> Found checkpoint! Resuming {len(energy_dict)} valid previously minimized frames...")
            except Exception:
                energy_dict = {}
                
        if args.min_steps > 0:
            print(f"  -> Minimizing and evaluating energy for each valid frame ({args.min_steps} steps per frame)...")
        else:
            print("  -> Evaluating direct potential energy for each valid frame (0 minimization steps)...")

        try:
            for local_idx, frame_idx in enumerate(tqdm(valid_frames)):
                if frame_idx in energy_dict and is_valid_energy(energy_dict[frame_idx]):
                    continue
                    
                if use_merged_traj:
                    ts = u_openmm.trajectory[local_idx]
                    frame_orig_pos = ts.positions * 0.1  # Angstroms to nm
                    heavy_pos_src = frame_orig_pos
                else:
                    u_curr, raw_local_idx = frame_universe_map[frame_idx]
                    ts = u_curr.trajectory[raw_local_idx]
                    h_indexer, map_idx = universe_heavy_info[id(u_curr)]
                    pos_angstroms = ts.positions[h_indexer]
                    pos_centered = pos_angstroms - pos_angstroms.mean(axis=0)
                    rms_dist = np.sqrt(np.mean(np.sum(pos_centered**2, axis=1)))
                    scale = 0.1 if rms_dist >= 5.0 else 1.0
                    frame_orig_pos = pos_angstroms * scale
                    if map_idx is None:
                        heavy_pos_src = frame_orig_pos
                    else:
                        heavy_pos_src = frame_orig_pos[map_idx]

                full_heavy_pos = np.zeros((num_fixer_atoms, 3), dtype=np.float32)
                full_heavy_pos[fixed_mask] = heavy_pos_src[fixed_orig_idx]
                if n_anchor > 0:
                    full_heavy_pos[anchor_mask] = heavy_pos_src[anchor_parent_idx] + ref_pos_anchor_offset + np.random.normal(0, 0.001, (n_anchor, 3))
                if n_fallback > 0:
                    full_heavy_pos[fallback_mask] = ref_pos[fallback_mask]

                current_full_pos = ref_full_pos.copy()
                current_full_pos[valid_h2f_dst] = full_heavy_pos[valid_h2f_src]
                if len(h_indices) > 0:
                    current_full_pos[h_indices] = current_full_pos[h_parents] + h_offsets
                
                try:
                    simulation.context.setPositions(current_full_pos * unit.nanometers)
                    if args.min_steps > 0:
                        simulation.minimizeEnergy(maxIterations=args.min_steps)
                        state = simulation.context.getState(getEnergy=True)
                        energy_val = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
                    else:
                        state = simulation.context.getState(getEnergy=True)
                        energy_val = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
                        # Adaptive clash-relief fallback: If static hydrogen placement caused an extreme clash (>1e5 kcal/mol or NaN),
                        # run a micro-minimization (25 steps) to relieve the clash and converge to physical energy
                        if np.isnan(energy_val) or np.isinf(energy_val) or energy_val > 1e5:
                            try:
                                simulation.minimizeEnergy(maxIterations=25)
                                state = simulation.context.getState(getEnergy=True)
                                energy_val = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
                            except Exception:
                                pass

                    if np.isnan(energy_val) or np.isinf(energy_val):
                        energy_val = np.nan
                except Exception:
                    energy_val = np.nan

                energy_dict[frame_idx] = energy_val
                
                if len(energy_dict) % 50 == 0:
                    pd.DataFrame(list(energy_dict.items()), columns=["Frame", "Energy_kcal_mol"]).to_csv(energy_csv, index=False)
                    
        except KeyboardInterrupt:
            df_partial = pd.DataFrame(list(energy_dict.items()), columns=["Frame", "Energy_kcal_mol"])
            df_partial.to_csv(energy_csv, index=False)
            print(f"\n⚠️  Energy calculation PAUSED by user! Saved checkpoint ({len(energy_dict)} frames) to {energy_csv}")
            
            if len(df_partial) > 1:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.scatter(df_partial["Frame"], df_partial["Energy_kcal_mol"], alpha=0.5, s=10, color='red')
                plt.xlabel("Frame Index")
                plt.ylabel("Potential Energy (kcal/mol)")
                plt.title("Partial Energy Timeline")
                
                plt.subplot(1, 2, 2)
                valid_partial = df_partial["Energy_kcal_mol"].dropna()
                if len(valid_partial) > 0:
                    plt.hist(valid_partial, bins=30, color='blue', alpha=0.7)
                plt.xlabel("Potential Energy (kcal/mol)")
                plt.ylabel("Frequency (Count)")
                plt.title("Partial Energy Distribution")
                
                plt.tight_layout()
                partial_plot = f"{out_dir}/partial_energy_progress.png"
                plt.savefig(partial_plot, dpi=300)
                plt.close()
                print(f"📊 Generated a progress visualization of your partial results: {partial_plot}")
                
            print("Run the script again to resume exactly where you left off.")
            sys.exit(0)
            
        pd.DataFrame(list(energy_dict.items()), columns=["Frame", "Energy_kcal_mol"]).to_csv(energy_csv, index=False)
        
        energies_kcal = [energy_dict.get(f, np.nan) for f in valid_frames]
        df_data["OpenMM_Energy_kcal_mol"] = energies_kcal
        valid_cnt = sum(1 for e in energies_kcal if pd.notna(e))
        methodology.append(f"Evaluated Thermodynamic Potential Energy (OpenMM, Amber14/OBC2, {args.min_steps}-step Minimization, {valid_cnt}/{len(valid_frames)} valid).")
        print(f"✅ OpenMM Energy Evaluation complete ({valid_cnt}/{len(valid_frames)} valid frames).")
        
    except Exception as e:
        print(f"❌ Error during OpenMM Energy calculation: {e}")
        methodology.append(f"OpenMM Energy Calculation failed: {e}")

# --- Smart CSV Fusion (Preserve Energy Data) ---
energy_csv = f"{out_dir}/energy_checkpoint.csv"
if "OpenMM_Energy_kcal_mol" not in df_data and os.path.exists(energy_csv):
    try:
        df_resume = pd.read_csv(energy_csv)
        ene_col = "Energy_kcal_mol" if "Energy_kcal_mol" in df_resume.columns else "Energy"
        energy_dict = dict(zip(df_resume["Frame"], df_resume[ene_col]))
        df_data["OpenMM_Energy_kcal_mol"] = [energy_dict.get(f, np.nan) for f in valid_frames]
        print(f"✅ Auto-fused {len(energy_dict)} existing energy evaluations into clustering results!")
    except Exception as e:
        print(f"⚠️ Could not fuse energy checkpoint: {e}")

print("\n--- Generating Final Outputs ---")
df = pd.DataFrame(df_data)

# ==========================================
# Generate Static HTML Art Nouveau Dashboard
# ==========================================
html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MET Flow - Art Nouveau Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.4.1/papaparse.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Lato:wght@300;400;700&display=swap" rel="stylesheet">
    <style>
        :root { --bg-color: #f4f1ea; --text-color: #1b3a30; --gold: #d4af37; --dark-green: #0f251e; --panel-bg: #ffffff; }
        body { font-family: 'Lato', sans-serif; background-color: var(--bg-color); color: var(--text-color); margin: 0; padding: 0; display: flex; flex-direction: column; height: 100vh; }
        header { background-color: var(--dark-green); color: var(--gold); padding: 20px; text-align: center; font-family: 'Playfair Display', serif; border-bottom: 4px solid var(--gold); box-shadow: 0 4px 10px rgba(0,0,0,0.2); }
        h1 { margin: 0; font-size: 2.5em; font-weight: 700; letter-spacing: 2px; }
        .container { display: flex; flex: 1; padding: 20px; gap: 20px; overflow: hidden; }
        .sidebar { width: 300px; background: var(--panel-bg); border: 2px solid var(--gold); border-radius: 15px 50px 15px 50px; padding: 20px; box-shadow: 5px 5px 15px rgba(0,0,0,0.1); display: flex; flex-direction: column; gap: 15px; overflow-y: auto; }
        .sidebar h2 { font-family: 'Playfair Display', serif; color: var(--dark-green); margin-top: 0; border-bottom: 1px solid var(--gold); padding-bottom: 10px;}
        .main-content { flex: 1; display: flex; flex-direction: column; gap: 20px; overflow-y: auto; padding-right: 10px;}
        .plot-container { background: var(--panel-bg); border: 2px solid var(--gold); border-radius: 5px; padding: 10px; box-shadow: 3px 3px 10px rgba(0,0,0,0.1); position: relative; }
        select, input[type="file"], .export-btn { width: 100%; padding: 8px; margin-top: 5px; border: 1px solid var(--gold); border-radius: 5px; background-color: var(--bg-color); font-family: 'Lato', sans-serif; }
        .export-btn { background: var(--gold); color: var(--dark-green); font-weight: bold; cursor: pointer; transition: 0.2s; border: none;}
        .export-btn:hover { filter: brightness(1.1); }
        .file-drop { border: 2px dashed var(--gold); padding: 15px; text-align: center; cursor: pointer; background: var(--bg-color); border-radius: 10px; transition: 0.3s; font-size: 0.9em; }
        .file-drop:hover { background: #e8e4d8; }
        #pymol-modal { display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: var(--dark-green); color: var(--gold); padding: 30px; border: 3px solid var(--gold); border-radius: 10px; z-index: 1000; box-shadow: 0 10px 30px rgba(0,0,0,0.5); width: 80%; max-width: 600px; }
        #pymol-modal h3 { font-family: 'Playfair Display', serif; margin-top:0; }
        #pymol-code { background: #000; color: #0f0; padding: 15px; border-radius: 5px; font-family: monospace; white-space: pre-wrap; word-wrap: break-word; }
        .close-btn { background: var(--gold); color: var(--dark-green); border: none; padding: 10px 20px; cursor: pointer; font-weight: bold; margin-top: 15px;}
        .overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 999; }
    </style>
</head>
<body>
    <header>
        <h1>MET Flow</h1>
        <div style="font-style: italic;">Thermodynamic & Conformational Analytics</div>
    </header>
    <div class="container">
        <div class="sidebar">
            <h2>Data Ingestion</h2>
            <div class="file-drop" id="drop-zone">
                Drop <b>global_cluster_metrics.csv</b>, <b>energy_checkpoint.csv</b>, or <b>pca_loadings.csv</b> here
                <input type="file" id="file-input" style="display: none;" accept=".csv" multiple>
            </div>
            <button class="export-btn" onclick="exportCSV()">💾 Export Merged CSV</button>
            <h2>Plot Controls</h2>
            <label>Filter Dataset: <select id="sel-filter-dataset" onchange="updatePlots()">
                <option value="all" selected>All Datasets (Combined)</option>
            </select></label>
            <label>X-Axis: <select id="sel-x"></select></label>
            <label>Y-Axis: <select id="sel-y"></select></label>
            <label>Z-Axis (3D): <select id="sel-z"><option value="none">None (2D Plot)</option></select></label>
            <label>Color By: <select id="sel-color"></select></label>
            <div id="color-mode-container" style="margin-top:8px;">
                <label>Coloring Mode: <select id="sel-color-mode" onchange="updatePlots()">
                    <option value="binned" selected>Interval Ranges (Box-Plot with Outliers)</option>
                    <option value="continuous">Continuous Gradient</option>
                </select></label>
            </div>
            <div id="colorscale-container" style="display:none; margin-top:8px;">
                <label>Colorscale: <select id="sel-colorscale" onchange="updatePlots()">
                    <option value="Viridis">Viridis</option>
                    <option value="Plasma">Plasma</option>
                    <option value="Inferno">Inferno</option>
                    <option value="Jet">Jet</option>
                    <option value="RdBu">Red-Blue (RdBu)</option>
                </select></label>
            </div>
            <div style="margin-top: 15px;">
                <label style="display:flex; align-items:center; gap: 8px;">
                    <input type="checkbox" id="filter-energy" onchange="updatePlots()"> Only show frames with Energy data
                </label>
            </div>
        </div>
        <div class="main-content">
            <div class="plot-container" id="main-plot" style="height: 85vh; min-height: 700px;">
                <div style="text-align:center; padding: 150px; color: #888;">Please load a CSV to visualize data.</div>
            </div>
            <div class="plot-container" id="loadings-plot" style="display: none; height: 400px;"></div>
            <div style="display: flex; gap: 20px;">
                <div class="plot-container" id="timeseries-plot" style="flex: 1; height: 400px; overflow:hidden;">
                    <div style="text-align:center; padding: 100px; color: #888;">No Energy Data Found.<br><br>Drag & drop <b>energy_checkpoint.csv</b> to dynamically merge and visualize energy!</div>
                </div>
                <div class="plot-container" id="hist-plot" style="flex: 1; height: 400px;"></div>
            </div>
        </div>
    </div>
    <div class="overlay" id="overlay"></div>
    <div id="pymol-modal">
        <h3>Frame Selected!</h3>
        <p>Frame ID: <span id="modal-frame"></span> | Cluster/Color: <span id="modal-cluster"></span> | Energy: <span id="modal-energy"></span></p>
        <p>To extract this frame in PyMOL on your server, copy and paste this command:</p>
        <div id="pymol-code"></div>
        <button class="close-btn" onclick="closeModal()">Close</button>
    </div>
    <script>
        let globalData = [];
        let globalLoadings = [];
        let cols = [];
        let pendingFiles = [];
        let mergeTimeout = null;
        
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.background = '#e8e4d8'; });
        dropZone.addEventListener('dragleave', () => dropZone.style.background = '');
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault(); dropZone.style.background = '';
            Array.from(e.dataTransfer.files).forEach(f => processFile(f));
        });
        fileInput.addEventListener('change', (e) => { 
            Array.from(e.target.files).forEach(f => processFile(f)); 
        });

        window.addEventListener('DOMContentLoaded', () => {
            fetch('global_cluster_metrics.csv')
                .then(r => { if (r.ok) return r.text(); throw new Error('Not found'); })
                .then(text => {
                    Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true, complete: function(res) {
                        if (res.meta && res.meta.fields) {
                            globalData = res.data;
                            cols = res.meta.fields;
                            populateDropdowns();
                            updatePlots();
                        }
                    }});
                }).catch(() => {});
        });
        
        function processFile(file) {
            Papa.parse(file, { header: true, dynamicTyping: true, skipEmptyLines: true,
                complete: function(results) {
                    if (results.meta.fields.includes('Residue') && results.meta.fields.includes('PC1_Contribution')) {
                        globalLoadings = results.data;
                        alert("✅ Loaded PCA Loadings Data!");
                        updatePlots();
                        return;
                    }
                    
                    pendingFiles.push(results);
                    
                    clearTimeout(mergeTimeout);
                    mergeTimeout = setTimeout(() => {
                        if (pendingFiles.length > 0) mergePendingFiles();
                    }, 100);
                }
            });
        }
        
        function mergePendingFiles() {
            pendingFiles.sort((a, b) => b.data.length - a.data.length);
            
            pendingFiles.forEach(results => {
                let rCols = results.meta.fields;
                
                if(rCols.includes('Energy_kcal_mol') && !rCols.includes('OpenMM_Energy_kcal_mol')) {
                    results.data.forEach(d => d['OpenMM_Energy_kcal_mol'] = d['Energy_kcal_mol']);
                    rCols.push('OpenMM_Energy_kcal_mol');
                }
                if(rCols.includes('Energy') && !rCols.includes('OpenMM_Energy_kcal_mol')) {
                    results.data.forEach(d => d['OpenMM_Energy_kcal_mol'] = d['Energy']);
                    rCols.push('OpenMM_Energy_kcal_mol');
                }
                
                if (globalData.length === 0) {
                    globalData = results.data; 
                    cols = rCols;
                } else {
                    let newCols = rCols.filter(c => !cols.includes(c));
                    if (newCols.length > 0) {
                        let dataMap = new Map();
                        results.data.forEach(d => { 
                            let df = (d['Frame'] !== undefined) ? d['Frame'] : d['Frame_Index'];
                            if (df !== undefined) dataMap.set(df, d); 
                        });
                        
                        globalData.forEach(g => {
                            let f = (g['Frame'] !== undefined) ? g['Frame'] : g['Frame_Index'];
                            if (dataMap.has(f)) {
                                let nd = dataMap.get(f);
                                newCols.forEach(c => g[c] = nd[c]);
                            }
                        });
                        cols = [...cols, ...newCols];
                        alert("✅ Successfully merged new dataset into existing plots!");
                    }
                }
            });
            
            pendingFiles = [];
            populateDropdowns(); 
            updatePlots();
        }
        
        function exportCSV() {
            if(globalData.length === 0) return;
            const csv = Papa.unparse(globalData);
            const blob = new Blob([csv], {type: 'text/csv;charset=utf-8;'});
            const link = document.createElement("a");
            const url = URL.createObjectURL(blob);
            link.setAttribute("href", url);
            link.setAttribute("download", "global_cluster_metrics_MERGED.csv");
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
        
        function populateDropdowns() {
            const sels = ['sel-x', 'sel-y', 'sel-z', 'sel-color'];
            const prevVals = {};
            sels.forEach(id => {
                prevVals[id] = document.getElementById(id).value;
            });

            sels.forEach(id => {
                const el = document.getElementById(id);
                if(id !== 'sel-z') el.innerHTML = '';
                else el.innerHTML = '<option value="none">None (2D Plot)</option>';
                cols.forEach(c => { el.innerHTML += `<option value="${c}">${c}</option>`; });
            });

            const pc1_col = cols.find(c => c.startsWith('PCA_1'));
            const pc2_col = cols.find(c => c.startsWith('PCA_2'));

            const selX = document.getElementById('sel-x');
            if (prevVals['sel-x'] && cols.includes(prevVals['sel-x'])) selX.value = prevVals['sel-x'];
            else if (pc1_col) selX.value = pc1_col;

            const selY = document.getElementById('sel-y');
            if (prevVals['sel-y'] && cols.includes(prevVals['sel-y'])) selY.value = prevVals['sel-y'];
            else if (pc2_col) selY.value = pc2_col;

            const selZ = document.getElementById('sel-z');
            if (prevVals['sel-z'] && (prevVals['sel-z'] === 'none' || cols.includes(prevVals['sel-z']))) selZ.value = prevVals['sel-z'];
            else if (cols.includes('OpenMM_Energy_kcal_mol')) selZ.value = 'OpenMM_Energy_kcal_mol';
            else selZ.value = 'none';

            const dsSelect = document.getElementById('sel-filter-dataset');
            if (dsSelect && cols.includes('Dataset')) {
                const prevDs = dsSelect.value;
                const uniqueDs = Array.from(new Set(globalData.map(d => d['Dataset']).filter(Boolean)));
                dsSelect.innerHTML = '<option value="all">All Datasets (Combined)</option>';
                uniqueDs.forEach(ds => {
                    const opt = document.createElement('option');
                    opt.value = ds;
                    opt.innerText = ds;
                    dsSelect.appendChild(opt);
                });
                if (uniqueDs.includes(prevDs)) dsSelect.value = prevDs;
            }

            const selColor = document.getElementById('sel-color');
            if (prevVals['sel-color'] && cols.includes(prevVals['sel-color'])) {
                selColor.value = prevVals['sel-color'];
            } else if (cols.includes('Dataset')) {
                selColor.value = 'Dataset';
            } else if (cols.includes('Daura_Cluster')) {
                selColor.value = 'Daura_Cluster';
            } else if (cols.includes('PCA_KMeans_Cluster')) {
                selColor.value = 'PCA_KMeans_Cluster';
            }
        }
        
        ['sel-x', 'sel-y', 'sel-z', 'sel-color', 'sel-filter-dataset'].forEach(id => { 
            const el = document.getElementById(id);
            if (el) el.addEventListener('change', updatePlots); 
        });
        
        function updatePlots() {
            if(globalLoadings.length > 0) {
                document.getElementById('loadings-plot').style.display = 'block';
                Plotly.react('loadings-plot', [
                    { x: globalLoadings.map(d=>d.Residue), y: globalLoadings.map(d=>d.PC1_Contribution), name: 'PC1', type: 'scatter', mode: 'lines+markers', line: {color: '#1b3a30'} },
                    { x: globalLoadings.map(d=>d.Residue), y: globalLoadings.map(d=>d.PC2_Contribution), name: 'PC2', type: 'scatter', mode: 'lines+markers', line: {color: '#d4af37'} }
                ], { title: {text: 'PCA Loadings (Residue Contribution)', font: {family: 'Playfair Display', color: '#1b3a30'}}, xaxis: {title: 'Residue Number'}, yaxis: {title: 'Loading Magnitude'}, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' }, {responsive: true});
            }
            
            if(!globalData.length) return;
            
            let plotData = globalData;
            const selDataset = document.getElementById('sel-filter-dataset') ? document.getElementById('sel-filter-dataset').value : 'all';
            if (selDataset && selDataset !== 'all' && cols.includes('Dataset')) {
                plotData = plotData.filter(d => d['Dataset'] === selDataset);
            }
            if(document.getElementById('filter-energy') && document.getElementById('filter-energy').checked && cols.includes('OpenMM_Energy_kcal_mol')) {
                plotData = plotData.filter(d => d['OpenMM_Energy_kcal_mol'] !== undefined && d['OpenMM_Energy_kcal_mol'] !== null && d['OpenMM_Energy_kcal_mol'] !== '');
            }
            if(!plotData.length) plotData = globalData;
            
            const cx = document.getElementById('sel-x').value;
            const cy = document.getElementById('sel-y').value;
            const cz = document.getElementById('sel-z').value;
            const cColor = document.getElementById('sel-color').value;
            const cScale = document.getElementById('sel-colorscale') ? document.getElementById('sel-colorscale').value : 'Viridis';
            
            const x = plotData.map(d => d[cx]);
            const y = plotData.map(d => d[cy]);
            const color = plotData.map(d => d[cColor]);
            const frames = plotData.map(d => (d['Frame'] !== undefined) ? d['Frame'] : d['Frame_Index']);
            const hovertext = plotData.map(d => `Frame: ${(d['Frame'] !== undefined) ? d['Frame'] : d['Frame_Index']}<br>${cColor}: ${d[cColor]}${d['Dataset'] ? '<br>Dataset: ' + d['Dataset'] : ''}`);
            
            const colorMode = document.getElementById('sel-color-mode') ? document.getElementById('sel-color-mode').value : 'binned';
            const csContainer = document.getElementById('colorscale-container');
            const cmContainer = document.getElementById('color-mode-container');

            function parseVal(v) {
                if (v === null || v === undefined) return null;
                if (typeof v === 'number') return isNaN(v) ? null : v;
                let s = String(v).trim();
                if (s === '' || s.toLowerCase() === 'null' || s.toLowerCase() === 'nan' || s.toLowerCase() === 'n/a' || s.toLowerCase() === 'undefined') return null;
                let n = parseFloat(s);
                return isNaN(n) ? null : n;
            }

            let numericVals = color.map(v => parseVal(v));
            let validNums = numericVals.filter(v => v !== null);
            let validCount = validNums.length;
            
            const isEnergy = cColor.toLowerCase().includes('energy');
            const isExplicitCategory = ['dataset', 'cluster', 'label', 'method'].some(k => cColor.toLowerCase().includes(k));
            const uniqueValCount = new Set(color.map(v => (v === null || v === undefined) ? '' : String(v).trim())).size;
            const isCategory = isExplicitCategory || (!isEnergy && !['pca', 'rmsd', 'tsne', 'loading'].some(k => cColor.toLowerCase().includes(k)) && uniqueValCount <= 40);
            
            if (isCategory) {
                if (cmContainer) cmContainer.style.display = 'none';
                if (csContainer) csContainer.style.display = 'none';
            } else if (isEnergy) {
                if (cmContainer) cmContainer.style.display = 'block';
                if (csContainer) csContainer.style.display = (colorMode === 'continuous') ? 'block' : 'none';
            } else {
                if (cmContainer) cmContainer.style.display = 'none';
                if (csContainer) csContainer.style.display = 'block';
            }

            let traces = [];
            
            if (isCategory && color.length > 0) {
                const catPalette = [
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                    '#3366cc', '#dc3912', '#ff9900', '#109618', '#990099',
                    '#0099c6', '#dd4477', '#66aa00', '#b82e2e', '#316395'
                ];
                let groupMap = new Map();
                for (let i = 0; i < color.length; i++) {
                    let val = (color[i] === null || color[i] === undefined || String(color[i]).trim() === '') ? 'Unknown' : String(color[i]);
                    if (!groupMap.has(val)) groupMap.set(val, []);
                    groupMap.get(val).push(i);
                }
                let colorIdx = 0;
                groupMap.forEach((indices, val) => {
                    const c = catPalette[colorIdx % catPalette.length];
                    colorIdx++;
                    let trace = {
                        name: `${cColor}: ${val} (n=${indices.length})`,
                        x: indices.map(i => x[i]),
                        y: indices.map(i => y[i]),
                        mode: 'markers',
                        marker: { size: 5, color: c },
                        text: indices.map(i => hovertext[i]),
                        hoverinfo: 'text+x+y+z',
                        customdata: indices.map(i => plotData[i])
                    };
                    if (cz !== 'none') { trace.type = 'scatter3d'; trace.z = indices.map(i => plotData[i][cz]); } 
                    else { trace.type = 'scatter'; }
                    traces.push(trace);
                });
            } else if (isEnergy && colorMode === 'binned') {
                if (validNums.length === 0) {
                    let trace = {
                        name: `${cColor}: No Valid Data / NaN (n=${plotData.length})`,
                        x: x,
                        y: y,
                        mode: 'markers',
                        marker: { size: 4, color: '#6b7280', opacity: 0.5 },
                        text: hovertext,
                        hoverinfo: 'text+x+y+z',
                        customdata: plotData
                    };
                    if (cz !== 'none') { trace.type = 'scatter3d'; trace.z = plotData.map(d => d[cz]); } 
                    else { trace.type = 'scatter'; }
                    traces.push(trace);
                } else {
                    const sorted = [...validNums].sort((a, b) => a - b);
                    const nValid = sorted.length;
                    
                    const q1 = sorted[Math.floor(nValid * 0.25)];
                    const med = sorted[Math.floor(nValid * 0.50)];
                    const q3 = sorted[Math.floor(nValid * 0.75)];
                    const iqr = q3 - q1;
                    
                    // For energy or skewed metrics, calculate robust cutoff
                    const p90 = sorted[Math.floor(nValid * 0.90)];
                    let highCutoff = (iqr > 0) ? (q3 + 1.5 * iqr) : p90;
                    // If extreme clashes pull highCutoff above 2500 kcal/mol, clamp it to p90 or 2000
                    if (cColor.toLowerCase().includes('energy')) {
                        if (p90 > q3 && p90 < highCutoff) highCutoff = p90;
                        if (highCutoff > 2000 && q3 < 2000) highCutoff = Math.max(q3, 1500);
                    }
                    let lowCutoff = (iqr > 0) ? (q1 - 1.5 * iqr) : sorted[0];
                    
                    const repVals = sorted.filter(v => v >= lowCutoff && v <= highCutoff);
                    const repMin = repVals.length > 0 ? repVals[0] : sorted[0];
                    const repMax = repVals.length > 0 ? repVals[repVals.length - 1] : highCutoff;
                    const nRep = repVals.length;
                    
                    const b1 = repVals[Math.floor(nRep * 0.20)] || repMin;
                    const b2 = repVals[Math.floor(nRep * 0.40)] || b1;
                    const b3 = repVals[Math.floor(nRep * 0.60)] || b2;
                    const b4 = repVals[Math.floor(nRep * 0.80)] || b3;
                    
                    const colors = ['#059669', '#0d9488', '#0284c7', '#d97706', '#dc2626'];
                    
                    let hasLowOutliers = (lowCutoff > sorted[0]);
                    let buckets = [];
                    if (hasLowOutliers) {
                        buckets.push({ name: `🟣 Low Outliers (< ${lowCutoff.toFixed(1)})`, color: '#4338ca', indices: [] });
                    }
                    buckets.push(
                        { name: `[${repMin.toFixed(1)}, ${b1.toFixed(1)})`, color: colors[0], indices: [] },
                        { name: `[${b1.toFixed(1)}, ${b2.toFixed(1)})`, color: colors[1], indices: [] },
                        { name: `[${b2.toFixed(1)}, ${b3.toFixed(1)})`, color: colors[2], indices: [] },
                        { name: `[${b3.toFixed(1)}, ${b4.toFixed(1)})`, color: colors[3], indices: [] },
                        { name: `[${b4.toFixed(1)}, ${repMax.toFixed(1)}]`, color: colors[4], indices: [] },
                        { name: `⚫ High Outliers / Clashes (> ${highCutoff.toFixed(1)})`, color: '#111111', indices: [] },
                        { name: `❌ Clashes / NaN`, color: '#4b5563', indices: [] }
                    );
                    
                    const offset = hasLowOutliers ? 1 : 0;
                    for (let i = 0; i < plotData.length; i++) {
                        const val = numericVals[i];
                        if (val === null || isNaN(val)) {
                            buckets[buckets.length - 1].indices.push(i);
                        } else if (val > highCutoff) {
                            buckets[buckets.length - 2].indices.push(i);
                        } else if (hasLowOutliers && val < lowCutoff) {
                            buckets[0].indices.push(i);
                        } else if (val < b1) {
                            buckets[offset].indices.push(i);
                        } else if (val < b2) {
                            buckets[offset + 1].indices.push(i);
                        } else if (val < b3) {
                            buckets[offset + 2].indices.push(i);
                        } else if (val < b4) {
                            buckets[offset + 3].indices.push(i);
                        } else {
                            buckets[offset + 4].indices.push(i);
                        }
                    }
                    
                    buckets.forEach(b => {
                        if (b.indices.length > 0) {
                            let isHighOutlier = b.name.includes('High Outliers');
                            let isNaNGroup = b.name.includes('NaN');
                            let trace = {
                                name: `${b.name} (n=${b.indices.length})`,
                                x: b.indices.map(i => x[i]),
                                y: b.indices.map(i => y[i]),
                                mode: 'markers',
                                marker: {
                                    size: isHighOutlier ? 4 : (isNaNGroup ? 3.5 : 5),
                                    color: b.color,
                                    opacity: isNaNGroup ? 0.35 : (isHighOutlier ? 0.85 : 0.85)
                                },
                                text: b.indices.map(i => hovertext[i]),
                                hoverinfo: 'text+x+y+z',
                                customdata: b.indices.map(i => plotData[i])
                            };
                            if (cz !== 'none') { trace.type = 'scatter3d'; trace.z = b.indices.map(i => plotData[i][cz]); } 
                            else { trace.type = 'scatter'; }
                            traces.push(trace);
                        }
                    });
                }
            } else {
                let cmin = undefined;
                let cmax = undefined;
                const sorted = [...validNums].sort((a, b) => a - b);
                if (sorted.length > 20) {
                    cmin = sorted[Math.floor(sorted.length * 0.05)];
                    cmax = sorted[Math.floor(sorted.length * 0.95)];
                }

                let validIndices = [];
                let nanIndices = [];
                for (let i = 0; i < plotData.length; i++) {
                    if (numericVals[i] !== null && !isNaN(numericVals[i])) validIndices.push(i);
                    else nanIndices.push(i);
                }

                if (validIndices.length > 0) {
                    let trace = {
                        name: `${cColor} (Continuous)`,
                        x: validIndices.map(i => x[i]),
                        y: validIndices.map(i => y[i]),
                        mode: 'markers',
                        marker: {
                            size: 5,
                            color: validIndices.map(i => numericVals[i]),
                            colorscale: cScale,
                            cmin: cmin,
                            cmax: cmax,
                            showscale: true,
                            colorbar: { title: cColor }
                        },
                        text: validIndices.map(i => hovertext[i]),
                        hoverinfo: 'text+x+y+z',
                        customdata: validIndices.map(i => plotData[i])
                    };
                    if (cz !== 'none') { trace.type = 'scatter3d'; trace.z = validIndices.map(i => plotData[i][cz]); } 
                    else { trace.type = 'scatter'; }
                    traces.push(trace);
                }

                if (nanIndices.length > 0) {
                    let nanTrace = {
                        name: `${cColor}: No Data / NaN (n=${nanIndices.length})`,
                        x: nanIndices.map(i => x[i]),
                        y: nanIndices.map(i => y[i]),
                        mode: 'markers',
                        marker: { size: 3.5, color: '#9ca3af', opacity: 0.4 },
                        text: nanIndices.map(i => hovertext[i]),
                        hoverinfo: 'text+x+y+z',
                        customdata: nanIndices.map(i => plotData[i])
                    };
                    if (cz !== 'none') { nanTrace.type = 'scatter3d'; nanTrace.z = nanIndices.map(i => plotData[i][cz]); } 
                    else { nanTrace.type = 'scatter'; }
                    traces.push(nanTrace);
                }
            }
            
            let layout = {
                title: { text: 'Structural Landscape', font: {family: 'Playfair Display', size: 24, color: '#1b3a30'} },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: {l: 50, r: 50, b: 50, t: 80},
                font: {family: 'Lato', color: '#1b3a30'},
                showlegend: true,
                legend: { title: { text: cColor }, orientation: 'v', x: 1.02, y: 1 }
            };
            
            Plotly.react('main-plot', traces, layout, {responsive: true});
            
            const mainPlotEl = document.getElementById('main-plot');
            if (!mainPlotEl._hasClickBound) {
                mainPlotEl._hasClickBound = true;
                mainPlotEl.on('plotly_click', function(data){
                    const pt = data.points[0]; const custom = pt.customdata;
                    const frame = (custom['Frame'] !== undefined) ? custom['Frame'] : custom['Frame_Index'];
                    const cluster = custom[cColor];
                    const ds_tag = custom['Dataset'] ? ` (${custom['Dataset']})` : '';
                    const energy = (custom['OpenMM_Energy_kcal_mol'] !== undefined && custom['OpenMM_Energy_kcal_mol'] !== null && !isNaN(custom['OpenMM_Energy_kcal_mol'])) ? custom['OpenMM_Energy_kcal_mol'] : 'N/A (Steric Clash / Uncalculated)';
                    
                    document.getElementById('modal-frame').innerText = frame;
                    document.getElementById('modal-cluster').innerText = (cluster !== undefined && cluster !== null) ? (cluster + ds_tag) : ('N/A' + ds_tag);
                    document.getElementById('modal-energy').innerText = energy;
                    
                    const code = `# Load unified trajectory in PyMOL:\nload merged_reference.pdb\nload_traj merged_trajectory.xtc, merged_reference\ncmd.frame(${frame + 1})\nsave frame_${frame}_extracted.pdb, merged_reference, state=${frame + 1}\nprint("Viewing Frame ${frame} (State ${frame + 1})")`;
                    document.getElementById('pymol-code').innerText = code;
                    document.getElementById('overlay').style.display = 'block';
                    document.getElementById('pymol-modal').style.display = 'block';
                });
            }
            
            if(cols.includes('OpenMM_Energy_kcal_mol')) {
                const e_vals = globalData.map(d => Number(d['OpenMM_Energy_kcal_mol'])).filter(v => v !== null && v !== undefined && !isNaN(v));
                if (e_vals.length > 0) {
                    document.getElementById('timeseries-plot').innerHTML = '';
                    document.getElementById('hist-plot').innerHTML = '';
                    const sorted_e = [...e_vals].sort((a,b) => a - b);
                    const q1 = sorted_e[Math.floor(sorted_e.length * 0.25)];
                    const q3 = sorted_e[Math.floor(sorted_e.length * 0.75)];
                    const iqr = q3 - q1;
                    const e_filt = (iqr > 0) ? e_vals.filter(v => v >= q1 - 2*iqr && v <= q3 + 2*iqr) : e_vals;
                    Plotly.react('timeseries-plot', [{ x: frames, y: globalData.map(d => d['OpenMM_Energy_kcal_mol']), type: 'scatter', mode: 'lines', line: {color: '#d4af37'} }], { title: {text: 'Energy Progression', font: {family: 'Playfair Display', color: '#1b3a30'}}, xaxis: {title: 'Frame Index'}, yaxis: {title: 'Energy (kcal/mol)'}, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' }, {responsive: true});
                    Plotly.react('hist-plot', [{ x: e_filt, type: 'histogram', marker: {color: '#0f251e'} }], { title: {text: 'Energy Distribution (Representative Subset)', font: {family: 'Playfair Display', color: '#1b3a30'}}, xaxis: {title: 'Energy (kcal/mol)'}, yaxis: {title: 'Frequency'}, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' }, {responsive: true});
                }
            }
        }
        function closeModal() { document.getElementById('overlay').style.display = 'none'; document.getElementById('pymol-modal').style.display = 'none'; }
    </script>
</body>
</html>"""

if is_only_energy:
    df.to_csv(f"{out_dir}/global_cluster_metrics.csv", index=False)
    print(f"Saved complete energy metrics to {out_dir}/global_cluster_metrics.csv")
    
    with open(f"{out_dir}/metflow_dashboard.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"🌟 Generated interactive Dashboard: {out_dir}/metflow_dashboard.html")

    with open(f"{out_dir}/methodology.txt", "w") as f:
        f.write("\n".join(methodology) + "\n")
    print("Saved methodology.txt")
    print("==================================================")
    print(" ✅ Option [C] Energy ONLY Evaluation Complete!")
    print("==================================================")
    sys.exit(0)

try:
    df.rename(columns={"PCA_1": f"PCA_1 ({var_explained[0]:.1f}%)", "PCA_2": f"PCA_2 ({var_explained[1]:.1f}%)", "PCA_3": f"PCA_3 ({var_explained[2]:.1f}%)"}, inplace=True)
except Exception:
    pass
df.to_csv(f"{out_dir}/global_cluster_metrics.csv", index=False)
print(f"Saved complete metrics to {out_dir}/global_cluster_metrics.csv")

with open(f"{out_dir}/metflow_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"🌟 Generated interactive Dashboard (Copy to your laptop!): {out_dir}/metflow_dashboard.html")

# Generate PCA Dataset Origin Comparison Plot
plt.figure(figsize=(10, 8))
unique_datasets = list(pd.unique(df["Dataset"]))
palette = sns.color_palette("Set1", max(1, len(unique_datasets)))

pc1_col_name = [c for c in df.columns if c.startswith("PCA_1")][0]
pc2_col_name = [c for c in df.columns if c.startswith("PCA_2")][0]

for i, ds in enumerate(unique_datasets):
    sub = df[df["Dataset"] == ds]
    plt.scatter(sub[pc1_col_name], sub[pc2_col_name], label=f"{ds} (n={len(sub)})", color=palette[i % len(palette)], alpha=0.6, s=15)

plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Multi-Trajectory PCA Projections by Origin Dataset")
plt.xlabel(pc1_col_name)
plt.ylabel(pc2_col_name)
plt.savefig(f"{out_dir}/pca_dataset_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved PCA Dataset Comparison Plot to {out_dir}/pca_dataset_comparison.png")

# Generate dataset cluster summary text file
summary_lines = ["=== Multi-Trajectory Dataset Cluster Distribution Summary ===", f"Total Datasets: {len(unique_datasets)}", f"Total Frames: {len(df)}\n"]

for ds in unique_datasets:
    ds_count = len(df[df["Dataset"] == ds])
    summary_lines.append(f"Dataset '{ds}': {ds_count} frames ({ds_count / len(df) * 100:.1f}%)")

for cluster_col in ["PCA_KMeans_Cluster", "Daura_Cluster", "CNN_Cluster"]:
    if cluster_col in df.columns:
        summary_lines.append(f"\n--- Cross-Tabulation: {cluster_col} ---")
        ct = pd.crosstab(df[cluster_col], df["Dataset"], margins=True)
        ct_pct = pd.crosstab(df[cluster_col], df["Dataset"], normalize='index') * 100
        summary_lines.append("Counts:")
        summary_lines.append(ct.to_string())
        summary_lines.append("\nRow Percentages (%):")
        summary_lines.append(ct_pct.round(1).to_string())

with open(f"{out_dir}/dataset_cluster_summary.txt", "w") as f:
    f.write("\n".join(summary_lines) + "\n")
print(f"Saved dataset cluster summary report to {out_dir}/dataset_cluster_summary.txt")

# Generate PCA Plots for each clustering method
for cluster_col, title, out_name in [
    ("PCA_KMeans_Cluster", "K-Means Clusters", "pca_kmeans_plot.png"),
    ("Daura_Cluster", "Daura Clusters", "daura_plot.png"),
    ("CNN_Cluster", "CNN/DBSCAN Clusters", "cnn_plot.png")
]:
    if cluster_col in df.columns:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df[f"PCA_1 ({var_explained[0]:.1f}%)"], df[f"PCA_2 ({var_explained[1]:.1f}%)"], c=df[cluster_col], cmap="tab20", alpha=0.6, s=10)
        plt.colorbar(scatter, label=title)
        plt.title(f"PCA Projection ({title})")
        plt.xlabel(f"Principal Component 1 ({var_explained[0]:.1f}%)")
        plt.ylabel(f"Principal Component 2 ({var_explained[1]:.1f}%)")
        plt.savefig(f"{out_dir}/{out_name}", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved 2D Cluster Graph to {out_dir}/{out_name}")

# Generate Advanced Plots (t-SNE and RMSD Heatmaps)
if args.advanced_plots.lower() == "true":
    print("\n--- Generating Advanced Non-linear Projections & Heatmaps ---")
    methodology.append("\n--- Advanced Cluster Visualizations ---")
    
    print("Calculating t-SNE projection (this may take a minute)...")
    tsne_model = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    X_tsne = tsne_model.fit_transform(X)
    df["tSNE_1"] = X_tsne[:, 0]
    df["tSNE_2"] = X_tsne[:, 1]
    
    plt.figure(figsize=(10, 8))
    for i, ds in enumerate(unique_datasets):
        sub = df[df["Dataset"] == ds]
        plt.scatter(sub["tSNE_1"], sub["tSNE_2"], label=f"{ds} (n={len(sub)})", color=palette[i % len(palette)], alpha=0.6, s=15)
    plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Multi-Trajectory t-SNE Projections by Origin Dataset")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(f"{out_dir}/tsne_dataset_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved t-SNE Dataset Comparison Plot to {out_dir}/tsne_dataset_comparison.png")
    
    for cluster_col, title, out_name in [
        ("Daura_Cluster", "Daura Clusters", "tsne_daura_plot.png"),
        ("CNN_Cluster", "CNN/DBSCAN Clusters", "tsne_cnn_plot.png")
    ]:
        if cluster_col in df.columns:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(df["tSNE_1"], df["tSNE_2"], c=df[cluster_col], cmap="tab20", alpha=0.6, s=10)
            plt.colorbar(scatter, label=title)
            plt.title(f"t-SNE Projection ({title})")
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.savefig(f"{out_dir}/{out_name}", dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved t-SNE Graph to {out_dir}/{out_name}")
            methodology.append(f"Generated t-SNE non-linear projection for {title}.")
            
    print("Generating Cluster RMSD Heatmaps...")
    for cluster_col, title, out_name in [
        ("Daura_Cluster", "Daura", "daura_rmsd_heatmap.png"),
        ("CNN_Cluster", "CNN/DBSCAN", "cnn_rmsd_heatmap.png")
    ]:
        if cluster_col in df.columns:
            sorted_idx = np.argsort(df[cluster_col].values)
            
            step = max(1, len(sorted_idx) // 1000)
            downsampled_idx = sorted_idx[::step]
            sorted_dist = rmsd_matrix[np.ix_(downsampled_idx, downsampled_idx)]
            
            plt.figure(figsize=(9, 8))
            plt.imshow(sorted_dist, cmap="viridis_r", aspect="auto", interpolation="none")
            plt.colorbar(label="RMSD (Å)")
            plt.title(f"RMSD Pairwise Distance Matrix (Sorted by {title} Clusters)")
            plt.xlabel("Frames (Sorted by Cluster)")
            plt.ylabel("Frames (Sorted by Cluster)")
            plt.savefig(f"{out_dir}/{out_name}", dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved RMSD Heatmap to {out_dir}/{out_name}")
            methodology.append(f"Generated RMSD pairwise distance block heatmap for {title} clusters.")

# Generate PCA Loadings Plot
n_atoms = len(ref_bb_atoms)
resids = ref_bb_atoms.resids
components = pca_model.components_

atom_contrib_pc1 = np.sqrt(np.sum(components[0].reshape(n_atoms, 3)**2, axis=1))
atom_contrib_pc2 = np.sqrt(np.sum(components[1].reshape(n_atoms, 3)**2, axis=1))

unique_resids, res_inv = np.unique(resids, return_inverse=True)
res_contrib_pc1 = np.bincount(res_inv, weights=atom_contrib_pc1)
res_contrib_pc2 = np.bincount(res_inv, weights=atom_contrib_pc2)

all_resids = np.arange(unique_resids.min(), unique_resids.max() + 1)
pc1_full = np.full(len(all_resids), np.nan)
pc2_full = np.full(len(all_resids), np.nan)

idx_arr = unique_resids - unique_resids.min()
pc1_full[idx_arr] = res_contrib_pc1
pc2_full[idx_arr] = res_contrib_pc2

pd.DataFrame({
    "Residue": all_resids,
    "PC1_Contribution": pc1_full,
    "PC2_Contribution": pc2_full
}).to_csv(f'{out_dir}/pca_loadings.csv', index=False)
print(f"Saved PCA Loadings data to {out_dir}/pca_loadings.csv")

plt.figure(figsize=(12, 5))
plt.plot(all_resids, pc1_full, label='PC1 Contribution', alpha=0.8, color='blue', marker='.', markersize=4)
plt.plot(all_resids, pc2_full, label='PC2 Contribution', alpha=0.8, color='orange', marker='.', markersize=4)
plt.xlabel('Residue Number')
plt.ylabel('Loading Magnitude (RMS)')
plt.title('Per-Residue Contribution to Principal Components')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{out_dir}/pca_loadings_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved PCA Loadings Plot to {out_dir}/pca_loadings_plot.png")

# Generate Energy Landscape Plot
if "OpenMM_Energy_kcal_mol" in df.columns:
    plt.figure(figsize=(10, 8))
    
    en = df["OpenMM_Energy_kcal_mol"]
    vmax = np.percentile(en, 95)
    vmin = np.min(en)
    
    scatter = plt.scatter(df[f"PCA_1 ({var_explained[0]:.1f}%)"], df[f"PCA_2 ({var_explained[1]:.1f}%)"], c=en, cmap="coolwarm", alpha=0.7, s=15, vmin=vmin, vmax=vmax)
    plt.colorbar(scatter, label="Energy (kcal/mol)")
    plt.title("Free Energy Landscape (PCA colored by OpenMM Energy)")
    plt.xlabel(f"Principal Component 1 ({var_explained[0]:.1f}%)")
    plt.ylabel(f"Principal Component 2 ({var_explained[1]:.1f}%)")
    plt.savefig(f"{out_dir}/energy_landscape_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Energy Landscape Plot to {out_dir}/energy_landscape_plot.png")

# Generate PCA 1D Distribution (Frequency vs PC Value)
plt.figure(figsize=(10, 6))
plt.hist(df[f"PCA_1 ({var_explained[0]:.1f}%)"], bins=50, density=True, alpha=0.5, color="blue", label="PC1 Score")
plt.hist(df[f"PCA_2 ({var_explained[1]:.1f}%)"], bins=50, density=True, alpha=0.5, color="orange", label="PC2 Score")
plt.xlabel("Principal Component Score")
plt.ylabel("Frequency (Density)")
plt.title("1D Distribution of Principal Component Scores")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{out_dir}/pca_1d_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved PCA 1D Distributions to {out_dir}/pca_1d_distributions.png")

with open(f"{out_dir}/methodology.txt", "w") as f:
    f.write("\n".join(methodology) + "\n")
print("Saved methodology.txt")

print(f"\n✅ Post-Generation Analysis Complete! Results saved to {out_dir}/ directory.")

print("==================================================")
print(" ✅ Post-Generation Analysis Complete!")
print("==================================================")
EOF

cp "$INSTALL_DIR/analyze_trajectory.py" "./analyze_trajectory.py" 2>/dev/null || true
if [ -d "/data1/mgs/programs/metflow" ]; then
    cp "$INSTALL_DIR/analyze_trajectory.py" "/data1/mgs/programs/metflow/analyze_trajectory.py" 2>/dev/null || true
fi
EXISTING_METFLOW=$(command -v metflow || echo "")
if [ -n "$EXISTING_METFLOW" ]; then
    EXISTING_DIR=$(dirname "$(dirname "$(readlink -f "$EXISTING_METFLOW")")")
    if [ -d "$EXISTING_DIR" ]; then
        cp "$INSTALL_DIR/analyze_trajectory.py" "$EXISTING_DIR/analyze_trajectory.py" 2>/dev/null || true
    fi
fi

# ----- BASH WRAPPER: metflow -----
cat << 'EOF' > "$INSTALL_DIR/bin/metflow"
#!/bin/bash
set -e

SCRIPT_PATH=$(readlink -f "$0" 2>/dev/null || readlink -f "$BASH_SOURCE")
BIN_DIR=$(dirname "$SCRIPT_PATH")
INSTALL_DIR=$(dirname "$BIN_DIR")
MAMBA_BIN=$(command -v micromamba || echo "$HOME/.local/bin/micromamba")

eval "$("$MAMBA_BIN" shell hook --shell=bash)"
micromamba activate conf_sel_gen

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_FORCE_GPU_ALLOW_GROWTH=true
export JAX_PLATFORMS=cpu

LOG_FILE="metflow_run.log"
OUTPUT_DIR="./bioemu_ensemble"
STATUS_FILE=".metflow_status"

run_analysis() {
    echo "================================================="
    echo "   MET Pipeline Analysis & Clustering Menu"
    echo "================================================="
    
    PARAM_FILE="metflow_params.txt"
    LOAD_PREV=false

    if [ "$2" == "true" ]; then
        OUT_DIR_VAL="${CUSTOM_OUT:-clustering_results}"
        CMD_ARGS=(--only_dashboard "true" --out_dir "$OUT_DIR_VAL")
        mkdir -p "$OUT_DIR_VAL"
        PARAM_CONTENT="max_frames=0
skim=false
cluster=none
calc_energy=false
min_steps=0
allostery=false
pca_ref=combined
pdb=${CUSTOM_PDB:-}
xtc=${CUSTOM_XTC:-}
labels=${CUSTOM_LABELS:-}
out_dir=${OUT_DIR_VAL}
multi_datasets=${IS_MULTI_DATASET:-false}
only_energy=false
only_dashboard=true"
        echo "$PARAM_CONTENT" > "$PARAM_FILE"
        echo "$PARAM_CONTENT" > "$OUT_DIR_VAL/$PARAM_FILE"

        TARGET_SCRIPT="$INSTALL_DIR/analyze_trajectory.py"
        if [ -f "./analyze_trajectory.py" ]; then
            TARGET_SCRIPT="./analyze_trajectory.py"
        elif [ -f "$BIN_DIR/analyze_trajectory.py" ]; then
            TARGET_SCRIPT="$BIN_DIR/analyze_trajectory.py"
        fi

        micromamba run -n conf_sel_analysis python "$TARGET_SCRIPT" "${CMD_ARGS[@]}"
        exit 0
    fi

    if [ "$1" != "true" ] && [ "$2" != "true" ] && [ -f "$PARAM_FILE" ]; then
        echo "⚠️ Found existing parameter file: $PARAM_FILE"
        echo "  [1] Start New Analysis (Interactive Prompt) [DEFAULT]"
        echo "  [2] Load Previous Parameters (from $PARAM_FILE)"
        read -r -p "Selection [1-2] (Default: 1): " param_choice
        if [[ "$param_choice" == "2" ]]; then
            LOAD_PREV=true
            echo "Loading previous parameters from $PARAM_FILE..."
        fi
    fi

    if [ "$LOAD_PREV" = true ]; then
        CMD_ARGS=()
        while IFS='=' read -r key val || [ -n "$key" ]; do
            # Fast zero-copy bash string manipulation
            key="${key%$'\r'}"
            val="${val%$'\r'}"
            key="${key#"${key%%[![:space:]]*}"}"
            key="${key%"${key##*[![:space:]]}"}"
            val="${val#"${val%%[![:space:]]*}"}"
            val="${val%"${val##*[![:space:]]}"}"
            case "$key" in
                ''|\#*) continue ;;
                max_frames) [ -n "$val" ] && [ "$val" != "0" ] && CMD_ARGS+=(--max_frames "$val") ;;
                skim) [ "$val" == "true" ] && CMD_ARGS+=(--skim "true" --equator "11, 41, 67, 102, 129, 155" --top "126" --bottom "166") ;;
                cluster) [ -n "$val" ] && CMD_ARGS+=(--cluster "$val") ;;
                calc_energy) [ "$val" == "true" ] && CMD_ARGS+=(--calc_energy "true") ;;
                min_steps) [ -n "$val" ] && CMD_ARGS+=(--min_steps "$val") ;;
                allostery) [ "$val" == "true" ] && CMD_ARGS+=(--allostery "true" --advanced_plots "true") ;;
                pca_ref) [ -n "$val" ] && CMD_ARGS+=(--pca_ref "$val") ;;
                pdb) [ -n "$val" ] && CMD_ARGS+=(--pdb "$val") ;;
                xtc) [ -n "$val" ] && CMD_ARGS+=(--xtc "$val") ;;
                labels|dataset_labels) [ -n "$val" ] && CMD_ARGS+=(--dataset_labels "$val") ;;
                out_dir) [ -n "$val" ] && CMD_ARGS+=(--out_dir "$val"); OUT_DIR_VAL="$val" ;;
                multi_datasets) [ "$val" == "true" ] && CMD_ARGS+=(--multi_datasets "$val") ;;
                only_energy) [ "$val" == "true" ] && CMD_ARGS+=(--only_energy "true") ;;
                only_dashboard) [ "$val" == "true" ] && CMD_ARGS+=(--only_dashboard "true") ;;
            esac
        done < "$PARAM_FILE"
    elif [ "$1" == "true" ]; then
        echo ""
        echo "[Downsampling] Processing thousands of frames takes a long time."
        read -r -p "Max frames per dataset (e.g. 1000) or leave blank for ALL: " max_frames_input

        echo ""
        read -r -p "OpenMM minimization steps per frame (Default: 0): " min_steps_input
        MIN_STEPS="${min_steps_input:-0}"

        CLUSTER_OPT="none"
        SKIM_VAL="false"
        ALLOSTERY_VAL="false"
        PCA_REF_VAL="combined"
        CALC_ENERGY_VAL="true"
        ONLY_ENERGY_VAL="true"
        ONLY_DASHBOARD_VAL="false"

        CMD_ARGS=(--calc_energy "true" --only_energy "true" --cluster "none" --min_steps "$MIN_STEPS")
        if [ -n "$max_frames_input" ]; then
            CMD_ARGS+=(--max_frames "$max_frames_input")
        fi
        if [ -n "$CUSTOM_PDB" ]; then
            CMD_ARGS+=(--pdb "$CUSTOM_PDB")
        fi
        if [ -n "$CUSTOM_XTC" ]; then
            CMD_ARGS+=(--xtc "$CUSTOM_XTC")
        fi
        if [ -n "$CUSTOM_LABELS" ]; then
            CMD_ARGS+=(--dataset_labels "$CUSTOM_LABELS")
        fi
        OUT_DIR_VAL="${CUSTOM_OUT:-energy_results}"
        CMD_ARGS+=(--out_dir "$OUT_DIR_VAL")
        if [ -n "$IS_MULTI_DATASET" ]; then
            CMD_ARGS+=(--multi_datasets "$IS_MULTI_DATASET")
        fi

        mkdir -p "$OUT_DIR_VAL"
        PARAM_CONTENT="max_frames=${max_frames_input:-0}
skim=${SKIM_VAL}
cluster=${CLUSTER_OPT}
calc_energy=${CALC_ENERGY_VAL}
min_steps=${MIN_STEPS}
allostery=${ALLOSTERY_VAL}
pca_ref=${PCA_REF_VAL}
pdb=${CUSTOM_PDB:-}
xtc=${CUSTOM_XTC:-}
labels=${CUSTOM_LABELS:-}
out_dir=${OUT_DIR_VAL}
multi_datasets=${IS_MULTI_DATASET:-false}
only_energy=${ONLY_ENERGY_VAL}
only_dashboard=${ONLY_DASHBOARD_VAL}"
        echo "$PARAM_CONTENT" > "$PARAM_FILE"
        echo "$PARAM_CONTENT" > "$OUT_DIR_VAL/$PARAM_FILE"
    else
        echo ""
        echo "[Downsampling] Processing thousands of frames takes a long time."
        read -r -p "Max frames per dataset (e.g. 1000) or leave blank for ALL: " max_frames_input
        
        echo ""
        echo "[1] Skim based on 6 inner-barrel residues"
        echo "[2] No skimming, cluster everything"
        read -r -p "Selection [1-2] (Default: 2): " skim_choice
        
        echo ""
        echo "Select Clustering Algorithm:"
        echo "  [1] PCA + K-Means (Fast, geometric clustering in PC space) [DEFAULT]"
        echo "  [2] Daura / GROMOS RMSD Clustering (Biophysical, RMSD cutoff based)"
        echo "  [3] Common Nearest Neighbor (CNN) Density Clustering"
        echo "  [4] Compare All 3 Clustering Algorithms"
        read -r -p "Clustering Algorithm [1-4] (Default: 1): " cluster_choice

        echo ""
        read -r -p "Enable Advanced Options (Allostery/OpenMM Energy/t-SNE/Heatmaps)? [y/N]: " adv_choice

        MIN_STEPS="100"
        CALC_ENERGY_VAL="false"
        if [[ "$adv_choice" =~ ^[Yy]$ ]]; then
            CALC_ENERGY_VAL="true"
            read -r -p "OpenMM minimization steps per frame (Default: 100): " min_steps_input
            MIN_STEPS="${min_steps_input:-100}"
        fi

        echo ""
        echo "PCA Projection Space:"
        echo "  [1] Combined PCA Space (Global variance across datasets) [DEFAULT]"
        echo "  [2] Reference PCA Space (Fit on Dataset 1 'bioemu', project all into it)"
        read -r -p "Selection [1-2] (Default: 1): " pca_ref_choice

        case $cluster_choice in
            2) CLUSTER_OPT="daura" ;;
            3) CLUSTER_OPT="cnn" ;;
            4) CLUSTER_OPT="all" ;;
            *) CLUSTER_OPT="pca_kmeans" ;;
        esac

        CMD_ARGS=(--cluster "$CLUSTER_OPT")
        if [ -n "$max_frames_input" ]; then
            CMD_ARGS+=(--max_frames "$max_frames_input")
        fi
        SKIM_VAL="false"
        if [[ "$skim_choice" == "1" ]]; then
            SKIM_VAL="true"
            CMD_ARGS+=(--skim "true" --equator "11, 41, 67, 102, 129, 155" --top "126" --bottom "166")
        fi
        ALLOSTERY_VAL="false"
        if [[ "$adv_choice" =~ ^[Yy]$ ]]; then
            ALLOSTERY_VAL="true"
            CMD_ARGS+=(--calc_energy "true" --allostery "true" --advanced_plots "true" --min_steps "$MIN_STEPS")
        fi
        PCA_REF_VAL="combined"
        if [[ "$pca_ref_choice" == "2" ]]; then
            PCA_REF_VAL="first"
            CMD_ARGS+=(--pca_ref "first")
        fi

        if [ -n "$CUSTOM_PDB" ]; then
            CMD_ARGS+=(--pdb "$CUSTOM_PDB")
        fi
        if [ -n "$CUSTOM_XTC" ]; then
            CMD_ARGS+=(--xtc "$CUSTOM_XTC")
        fi
        if [ -n "$CUSTOM_LABELS" ]; then
            CMD_ARGS+=(--dataset_labels "$CUSTOM_LABELS")
        fi
        OUT_DIR_VAL="${CUSTOM_OUT:-clustering_results}"
        CMD_ARGS+=(--out_dir "$OUT_DIR_VAL")
        if [ -n "$IS_MULTI_DATASET" ]; then
            CMD_ARGS+=(--multi_datasets "$IS_MULTI_DATASET")
        fi

        ONLY_ENERGY_VAL="false"
        ONLY_DASHBOARD_VAL="false"

        mkdir -p "$OUT_DIR_VAL"
        PARAM_CONTENT="max_frames=${max_frames_input:-0}
skim=${SKIM_VAL}
cluster=${CLUSTER_OPT}
calc_energy=${CALC_ENERGY_VAL}
min_steps=${MIN_STEPS}
allostery=${ALLOSTERY_VAL}
pca_ref=${PCA_REF_VAL}
pdb=${CUSTOM_PDB:-}
xtc=${CUSTOM_XTC:-}
labels=${CUSTOM_LABELS:-}
out_dir=${OUT_DIR_VAL}
multi_datasets=${IS_MULTI_DATASET:-false}
only_energy=${ONLY_ENERGY_VAL}
only_dashboard=${ONLY_DASHBOARD_VAL}"
        echo "$PARAM_CONTENT" > "$PARAM_FILE"
        echo "$PARAM_CONTENT" > "$OUT_DIR_VAL/$PARAM_FILE"
    fi

    TARGET_SCRIPT="$INSTALL_DIR/analyze_trajectory.py"
    if [ -f "./analyze_trajectory.py" ]; then
        TARGET_SCRIPT="./analyze_trajectory.py"
    elif [ -f "$BIN_DIR/analyze_trajectory.py" ]; then
        TARGET_SCRIPT="$BIN_DIR/analyze_trajectory.py"
    fi

    echo ""
    read -r -p "Run analysis in the BACKGROUND (nohup ... &)? [y/N]: " bg_run_choice
    if [[ "$bg_run_choice" =~ ^[Yy]$ ]]; then
        LOG_OUT="metflow_analysis.log"
        nohup micromamba run -n conf_sel_analysis python "$TARGET_SCRIPT" "${CMD_ARGS[@]}" < /dev/null > "$LOG_OUT" 2>&1 &
        BG_PID=$!
        disown $BG_PID 2>/dev/null || true
        echo "🚀 Analysis running in background (PID: $BG_PID)."
        echo "📋 Log file: $LOG_OUT (Monitor with: tail -f $LOG_OUT)"
        exit 0
    else
        micromamba run -n conf_sel_analysis python "$TARGET_SCRIPT" "${CMD_ARGS[@]}"
        exit 0
    fi
}

RUNNING_PID=$(pgrep -f "run_pipeline.py|bioemu.sample" | head -n 1)
DO_ANALYSIS=false

if [ -f "$STATUS_FILE" ]; then
    RUNNING_PID=$(grep "PID=" "$STATUS_FILE" | cut -d'=' -f2)
    TARGET_SAMPLES=$(grep "TARGET_SAMPLES=" "$STATUS_FILE" | cut -d'=' -f2)
    
    if [ -n "$RUNNING_PID" ] && kill -0 "$RUNNING_PID" 2>/dev/null; then
        COMPLETED_SAMPLES=0
        if [ -d "$OUTPUT_DIR" ]; then
            COMPLETED_SAMPLES=$(ls -1q "$OUTPUT_DIR"/*.pdb 2>/dev/null | wc -l)
        fi
        
        PERCENTAGE=0
        if [ "$TARGET_SAMPLES" -gt 0 ]; then
            PERCENTAGE=$(( 100 * COMPLETED_SAMPLES / TARGET_SAMPLES ))
        fi
        
        echo "================================================="
        echo "⚠️  MET Pipeline is currently active in the background! (PID: $RUNNING_PID)"
        echo "📊 Progress: $COMPLETED_SAMPLES / $TARGET_SAMPLES samples generated ($PERCENTAGE% complete)"
        echo "================================================="
        echo "[1] Leave it running"
        echo "[2] Pause (Freeze) the run"
        echo "[3] Resume (Unfreeze) the run"
        echo "[4] Kill it (Stop permanently)"
        echo "[5] Analyze results (Skip to analysis menu)"
        read -p "Select an option [1-5]: " bg_choice
        case $bg_choice in
            2) kill -STOP $RUNNING_PID; echo "❄️  Paused!"; exit 0 ;;
            3) kill -CONT $RUNNING_PID; echo "🔥 Resumed!"; exit 0 ;;
            4) kill -9 $RUNNING_PID; rm -f "$STATUS_FILE"; echo "💀 Killed." ;;
            5) DO_ANALYSIS=true ;;
            *) echo "Check progress with: tail -f $LOG_FILE"; exit 0 ;;
        esac
    else
        rm -f "$STATUS_FILE"
    fi
elif [ -n "$RUNNING_PID" ]; then
    echo "================================================="
    echo "⚠️  MET Pipeline is currently active in the background! (PID: $RUNNING_PID)"
    echo "================================================="
    echo "[1] Leave it running"
    echo "[2] Pause (Freeze) the run"
    echo "[3] Resume (Unfreeze) the run"
    echo "[4] Kill it (Stop permanently)"
    echo "[5] Analyze results (Skip to analysis menu)"
    read -p "Select an option [1-5]: " bg_choice
    case $bg_choice in
        2) kill -STOP $RUNNING_PID; echo "❄️  Paused!"; exit 0 ;;
        3) kill -CONT $RUNNING_PID; echo "🔥 Resumed!"; exit 0 ;;
        4) kill -9 $RUNNING_PID; rm -f "$STATUS_FILE"; echo "💀 Killed." ;;
        5) DO_ANALYSIS=true ;;
        *) echo "Check progress with: tail -f $LOG_FILE"; exit 0 ;;
    esac
fi

if [ "$DO_ANALYSIS" = true ]; then
    run_analysis
fi

if [ -f "$OUTPUT_DIR/samples.xtc" ] || [ -f "samples.xtc" ]; then
    echo "✅ Detected completed trajectory (samples.xtc)!"
    read -p "Proceed to analysis and clustering? [Y/n]: " proceed_ans
    if [[ "$proceed_ans" =~ ^[Yy]$ || -z "$proceed_ans" ]]; then
        run_analysis
    fi
fi

NUM_SAMPLES=2000; DENOISING_STEPS=50; DENOISER_SOLVER="dpm"
while true; do
    echo "================================================="
    echo "   MET Pipeline Launch Configuration (v2.7 MERGED)"
    echo "================================================="
    echo "-------------------------------------------------"
    echo "  [P] Change parameters (Current: $NUM_SAMPLES samples, $DENOISING_STEPS steps, $DENOISER_SOLVER solver)"
    echo "  [S] Start run with Resource Configurator"
    echo "  [A] Analyze completed BioEmu results"
    echo "  [E] Analyze external trajectory (XTC/PDB)"
    echo "  [C] Calculate Thermodynamic Energy ONLY (OpenMM)"
    echo "  [D] Generate Dashboard HTML ONLY"
    echo "  [Q] Quit"
    read -p "Option: " main_choice
    
    case $main_choice in
        [Pp])
            echo ""
            echo "Enter the number [1-3] of the parameter you want to change: "
            echo "  [1] Number of samples (Current: $NUM_SAMPLES)"
            echo "  [2] Denoising steps (Current: $DENOISING_STEPS)"
            echo "  [3] Denoiser solver (Current: $DENOISER_SOLVER)"
            read -p "Parameter index: " param_idx
            case $param_idx in
                1)
                    read -p "Value (Default: 2000): " input_val
                    if [ -n "$input_val" ]; then NUM_SAMPLES=$input_val; fi
                    ;;
                2)
                    read -p "Value (Default: 50): " input_val
                    if [ -n "$input_val" ]; then DENOISING_STEPS=$input_val; fi
                    ;;
                3)
                    read -p "Value (dpm/euler): " input_val
                    if [ "$input_val" == "dpm" ] || [ "$input_val" == "euler" ]; then DENOISER_SOLVER=$input_val; fi
                    ;;
            esac
            ;;
        [Aa])
            unset CUSTOM_PDB CUSTOM_XTC CUSTOM_LABELS CUSTOM_OUT IS_MULTI_DATASET
            echo -e "\n--- Option [A]: Analyze BioEmu Results ---"
            echo "Select Master Reference Topology PDB for BioEmu analysis:"
            echo "  [1] ./bioemu_ensemble/topology.pdb [DEFAULT]"
            echo "  [2] Custom PDB file in working directory..."
            read -p "Selection [1-2] (Default: 1): " bioemu_pdb_choice
            if [[ "$bioemu_pdb_choice" == "2" ]]; then
                read -p "Enter path to Custom Reference PDB file: " CUSTOM_PDB
            else
                CUSTOM_PDB="./bioemu_ensemble/topology.pdb"
            fi
            run_analysis
            ;;
        [Cc])
            unset CUSTOM_PDB CUSTOM_XTC CUSTOM_LABELS CUSTOM_OUT IS_MULTI_DATASET
            echo -e "\n--- Option [C]: Calculate Energy ONLY ---"
            echo "  [1] Single Trajectory Dataset (Default: topology.pdb & samples.xtc)"
            echo "  [2] Multi-Trajectory Comparative Analysis (2+ Datasets)"
            echo "  [3] Auto-Detect Matching Datasets in Directory (PDB & XTC with identical stems)"
            read -p "Selection [1-3] (Default: 1): " c_mode

            if [[ "$c_mode" == "3" ]]; then
                PDB_ARRAY=(); XTC_ARRAY=(); LABEL_ARRAY=()
                orig_nullglob=$(shopt -p nullglob || true)
                shopt -s nullglob
                for pdb_file in *.pdb; do
                    stem="${pdb_file%.pdb}"
                    xtc_file="${stem}.xtc"
                    if [[ -f "$xtc_file" ]]; then
                        PDB_ARRAY+=("$pdb_file")
                        XTC_ARRAY+=("$xtc_file")
                        LABEL_ARRAY+=("$stem")
                    fi
                done
                eval "$orig_nullglob"
                if [ ${#PDB_ARRAY[@]} -eq 0 ]; then
                    echo "❌ No matching PDB and XTC pairs found with identical stems."
                    exit 1
                fi
                echo "✅ Auto-detected ${#PDB_ARRAY[@]} matching dataset pair(s)."
                read -p "Output folder name [Default: auto_detected_results]: " out_dir_in
                CUSTOM_OUT="${out_dir_in:-auto_detected_results}"
                CUSTOM_PDB=$(IFS=,; echo "${PDB_ARRAY[*]}")
                CUSTOM_XTC=$(IFS=,; echo "${XTC_ARRAY[*]}")
                CUSTOM_LABELS=$(IFS=,; echo "${LABEL_ARRAY[*]}")
                IS_MULTI_DATASET="true"
            elif [[ "$c_mode" == "2" ]]; then
                read -p "Number of datasets to calculate energy for: " num_ds
                num_ds="${num_ds:-2}"
                PDB_ARRAY=(); XTC_ARRAY=(); LABEL_ARRAY=()
                for ((i=1; i<=num_ds; i++)); do
                    echo -e "\n--- Dataset $i ---"
                    read -p "  PDB File Path: " pdb_i
                    read -p "  XTC File Path: " xtc_i
                    read -p "  Dataset Label: " label_i
                    PDB_ARRAY+=("$pdb_i")
                    XTC_ARRAY+=("$xtc_i")
                    LABEL_ARRAY+=("${label_i:-Dataset_$i}")
                done
                read -p "Output folder name [Default: energy_results]: " out_dir_in
                CUSTOM_OUT="${out_dir_in:-energy_results}"
                CUSTOM_PDB=$(IFS=,; echo "${PDB_ARRAY[*]}")
                CUSTOM_XTC=$(IFS=,; echo "${XTC_ARRAY[*]}")
                CUSTOM_LABELS=$(IFS=,; echo "${LABEL_ARRAY[*]}")
                IS_MULTI_DATASET="true"
            fi
            run_analysis "true"
            ;;
        [Dd]) unset CUSTOM_PDB CUSTOM_XTC CUSTOM_LABELS CUSTOM_OUT IS_MULTI_DATASET; run_analysis "false" "true" ;;
        [Ee])
            echo -e "\n--- Analyze External Trajectory ---"
            echo "[1] Single Trajectory Dataset"
            echo "[2] Multi-Trajectory Comparative Analysis (2+ Datasets)"
            echo "[3] Auto-Detect Matching Datasets in Directory (PDB & XTC sharing same prefix/stem)"
            echo "(Note: PDB and XTC pairs must share the same filename stem to be auto-detected)"
            read -p "Selection [1-3] (Default: 1): " ext_mode
            
            if [[ "$ext_mode" == "3" ]]; then
                PDB_ARRAY=()
                XTC_ARRAY=()
                LABEL_ARRAY=()
                
                orig_nullglob=$(shopt -p nullglob || true)
                shopt -s nullglob

                for pdb_file in *.pdb; do
                    stem="${pdb_file%.pdb}"
                    xtc_file="${stem}.xtc"
                    if [[ -f "$xtc_file" ]]; then
                        PDB_ARRAY+=("$pdb_file")
                        XTC_ARRAY+=("$xtc_file")
                        LABEL_ARRAY+=("$stem")
                    fi
                done
                eval "$orig_nullglob"

                if [ ${#PDB_ARRAY[@]} -eq 0 ]; then
                    echo "❌ No matching PDB and XTC pairs found with identical stems."
                    exit 1
                fi

                echo ""
                echo "✅ Auto-detected ${#PDB_ARRAY[@]} matching dataset pair(s):"
                for i in "${!PDB_ARRAY[@]}"; do
                    echo "  - Pair $((i+1)): PDB=${PDB_ARRAY[i]} | XTC=${XTC_ARRAY[i]} | Label=${LABEL_ARRAY[i]}"
                done

                read -p "Output folder name [Default: auto_detected_results]: " out_dir_in
                CUSTOM_OUT="${out_dir_in:-auto_detected_results}"

                CUSTOM_PDB=$(IFS=,; echo "${PDB_ARRAY[*]}")
                CUSTOM_XTC=$(IFS=,; echo "${XTC_ARRAY[*]}")
                CUSTOM_LABELS=$(IFS=,; echo "${LABEL_ARRAY[*]}")
                IS_MULTI_DATASET="true"
            elif [[ "$ext_mode" == "2" ]]; then
                read -p "Number of datasets to compare (e.g. 2 or 3): " num_ds
                num_ds="${num_ds:-2}"
                PDB_ARRAY=()
                XTC_ARRAY=()
                LABEL_ARRAY=()
                for ((i=1; i<=num_ds; i++)); do
                    echo -e "\n--- Dataset $i ---"
                    read -p "  PDB File Path: " pdb_i
                    read -p "  XTC File Path: " xtc_i
                    read -p "  Dataset Label (e.g. Fenoxycarb / No_ligand): " label_i
                    PDB_ARRAY+=("$pdb_i")
                    XTC_ARRAY+=("$xtc_i")
                    LABEL_ARRAY+=("${label_i:-Dataset_$i}")
                done
                read -p "Output folder name [Default: comparative_results]: " out_dir_in
                CUSTOM_OUT="${out_dir_in:-comparative_results}"
                CUSTOM_PDB=$(IFS=,; echo "${PDB_ARRAY[*]}")
                CUSTOM_XTC=$(IFS=,; echo "${XTC_ARRAY[*]}")
                CUSTOM_LABELS=$(IFS=,; echo "${LABEL_ARRAY[*]}")
                IS_MULTI_DATASET="true"
            else
                read -p "External PDB file: " CUSTOM_PDB
                read -p "External XTC trajectory file: " CUSTOM_XTC
                read -p "Dataset Label [Default: External]: " CUSTOM_LABELS
                read -p "Output folder name [Default: external_results]: " CUSTOM_OUT
            fi
            run_analysis
            ;;
        [Ss])
            break
            ;;
        [Qq])
            echo "Exiting."
            exit 0
            ;;
    esac
done

echo "================================================="
echo "   MET Pipeline Resource Configurator"
echo "================================================="
echo "How much heavy load do you want to use this session?"
echo "  [1] High Load (100% - Fastest)"
echo "  [2] Heavy Load (85%)"
echo "  [3] Moderate Load (70%)"
echo "  [4] Balanced Load (55%)"
echo "  [5] Light Load (40%)"
echo "  [6] Low Load (25% - Background tasks)"
echo "  [7] Minimal Load (10% - Minimum impact)"
read -p "Select load level [1-7] (Default: 1): " load_choice

EXTRA_ARGS="--num_samples $NUM_SAMPLES --denoising_steps $DENOISING_STEPS --denoiser_type $DENOISER_SOLVER"
CPU_CORES=$(nproc)

case $load_choice in
    2) THREADS=$(( CPU_CORES * 85 / 100 )); EXTRA_ARGS="$EXTRA_ARGS --batch_size_100 40" ;;
    3) THREADS=$(( CPU_CORES * 70 / 100 )); EXTRA_ARGS="$EXTRA_ARGS --batch_size_100 35" ;;
    4) THREADS=$(( CPU_CORES * 55 / 100 )); EXTRA_ARGS="$EXTRA_ARGS --batch_size_100 25" ;;
    5) THREADS=$(( CPU_CORES * 40 / 100 )); EXTRA_ARGS="$EXTRA_ARGS --batch_size_100 20" ;;
    6) THREADS=$(( CPU_CORES * 25 / 100 )); EXTRA_ARGS="$EXTRA_ARGS --batch_size_100 10"; [ "$THREADS" -lt 2 ] && THREADS=2 ;;
    7) THREADS=1; EXTRA_ARGS="$EXTRA_ARGS --batch_size_100 5" ;;
    *) THREADS=$CPU_CORES; EXTRA_ARGS="$EXTRA_ARGS --batch_size_100 50" ;;
esac

export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
echo "⚙️ Set Load to ${THREADS} CPU threads."

echo ""
read -p "Do you want to start the pipeline in the BACKGROUND for the PDB in this directory? [Y/n]: " answer
if [[ "$answer" =~ ^[Yy]$ || -z "$answer" ]]; then
    nohup python "$INSTALL_DIR/run_pipeline.py" $EXTRA_ARGS < /dev/null > "$LOG_FILE" 2>&1 &
    BG_PID=$!
    disown $BG_PID 2>/dev/null || true
    echo "✅ Pipeline is running in the background (PID: $BG_PID)."
else
    echo "Aborted."
fi
EOF

chmod +x "$INSTALL_DIR/bin/metflow"
cp -f "$INSTALL_DIR/bin/metflow" "./bin/metflow" 2>/dev/null || true
cp -f "$INSTALL_DIR/analyze_trajectory.py" "./analyze_trajectory.py" 2>/dev/null || true

# Universal user binary installation (~/.local/bin)
mkdir -p "$HOME/.local/bin" 2>/dev/null || true
cp -f "$INSTALL_DIR/bin/metflow" "$HOME/.local/bin/metflow" 2>/dev/null || true
chmod +x "$HOME/.local/bin/metflow" 2>/dev/null || true

# Universal PATH discovery: automatically update any active metflow executable found in PATH on ANY system
ACTIVE_METFLOW=$(command -v metflow || echo "")
if [ -n "$ACTIVE_METFLOW" ] && [ -f "$ACTIVE_METFLOW" ]; then
    ACTIVE_REAL=$(readlink -f "$ACTIVE_METFLOW" 2>/dev/null || echo "$ACTIVE_METFLOW")
    if [ -n "$ACTIVE_REAL" ] && [ -f "$ACTIVE_REAL" ]; then
        cp -f "$INSTALL_DIR/bin/metflow" "$ACTIVE_REAL" 2>/dev/null || true
        chmod +x "$ACTIVE_REAL" 2>/dev/null || true
        ACTIVE_DIR=$(dirname "$(dirname "$ACTIVE_REAL")")
        if [ -n "$ACTIVE_DIR" ] && [ "$ACTIVE_DIR" != "." ] && [ -d "$ACTIVE_DIR" ]; then
            cp -f "$INSTALL_DIR/analyze_trajectory.py" "$ACTIVE_DIR/analyze_trajectory.py" 2>/dev/null || true
            cp -f "$INSTALL_DIR/analyze_trajectory.py" "$ACTIVE_DIR/bin/analyze_trajectory.py" 2>/dev/null || true
        fi
        echo "✅ Successfully updated active metflow binary at: $ACTIVE_REAL"
    fi
fi

if [ -f "/data1/mgs/programs/bin/metflow" ]; then
    cp -f "$INSTALL_DIR/bin/metflow" "/data1/mgs/programs/bin/metflow" 2>/dev/null || true
    chmod +x "/data1/mgs/programs/bin/metflow" 2>/dev/null || true
    cp -f "$INSTALL_DIR/analyze_trajectory.py" "/data1/mgs/programs/analyze_trajectory.py" 2>/dev/null || true
    cp -f "$INSTALL_DIR/analyze_trajectory.py" "/data1/mgs/programs/bin/analyze_trajectory.py" 2>/dev/null || true
    echo "✅ Successfully updated cluster binary at /data1/mgs/programs/bin/metflow"
fi
