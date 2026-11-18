#!/bin/bash
set -e

echo "======================================================="
echo "   MET Workflow - SMART UNIVERSAL INSTALLER (v2.8 MULTI-SOURCE)"
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
    micromamba run -n conf_sel_gen pip install --no-cache-dir "torch==2.6.0"

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
"""Backward-compatible CLI entrypoint for MET Flow (v2.8 MULTI-SOURCE).

Delegates execution to scripts/run_pipeline.py.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.run_pipeline import main


if __name__ == "__main__":
    sys.exit(main())
EOF

cp -f "$INSTALL_DIR/analyze_trajectory.py" "./analyze_trajectory.py" 2>/dev/null || true

# METFLOW_MODULES_START
mkdir -p "$INSTALL_DIR/metflow"
mkdir -p "$INSTALL_DIR/metflow/stage1_ingestion"
mkdir -p "$INSTALL_DIR/metflow/stage2_skimming"
mkdir -p "$INSTALL_DIR/metflow/stage3_reduction"
mkdir -p "$INSTALL_DIR/metflow/stage4_clustering"
mkdir -p "$INSTALL_DIR/metflow/stage5_energy"
mkdir -p "$INSTALL_DIR/metflow/stage6_visualization"
mkdir -p "$INSTALL_DIR/scripts"
cat << 'EOF' > "$INSTALL_DIR/scripts/run_pipeline.py"
"""Central Orchestrator for the MET Flow (v2.8 MULTI-SOURCE) modular pipeline."""

import logging
import os
import sys
from typing import Callable, List, Optional, Tuple

# Ensure project root is on sys.path when invoked as `python scripts/run_pipeline.py`
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from metflow.cli import parse_args_to_config, validate_out_dir
from metflow.deps import install_deps
from metflow.models import PipelineConfig, StageResult, TrajectoryContext
from metflow.stage1_ingestion.loader import run_stage1
from metflow.stage2_skimming.skimmer import run_stage2
from metflow.stage3_reduction.allostery import run_stage3
from metflow.stage4_clustering.cnn import run_stage4
from metflow.stage5_energy.openmm_engine import run_stage5
from metflow.stage6_visualization.dashboard import generate_dashboard
from metflow.stage6_visualization.reports import init_methodology, log_methodology, run_stage6

logger = logging.getLogger("metflow.orchestrator")


def _run_only_dashboard(argv: Optional[List[str]] = None, base_dir: str = ".") -> int:
    """Fast path for --only_dashboard true without importing heavy scientific packages."""
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    out_dir = "clustering_results"
    if "--out_dir" in raw_args:
        out_idx = raw_args.index("--out_dir")
        if out_idx + 1 < len(raw_args):
            out_dir = raw_args[out_idx + 1]
    try:
        out_dir_abs = validate_out_dir(out_dir, base_dir=base_dir)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    dash_file = generate_dashboard(out_dir_abs, df=None)
    ctx_dash = TrajectoryContext(config=PipelineConfig(out_dir=out_dir_abs, only_dashboard="true"), out_dir=out_dir_abs)
    init_methodology(ctx_dash, "Dashboard Only")
    log_methodology(ctx_dash, f"Regenerated interactive dashboard: {dash_file}")
    print(f"\n✅ Dashboard generated instantly at {dash_file}")
    return 0


def _close_universe_readers(ctx: TrajectoryContext) -> None:
    """Close any open MDAnalysis trajectory readers stored in ctx."""
    seen = set()
    for u_obj in ctx.universes:
        if id(u_obj) in seen:
            continue
        seen.add(id(u_obj))
        try:
            u_obj.trajectory.close()
        except Exception:
            pass
    for entry in ctx.temp_universes:
        if isinstance(entry, tuple) and len(entry) >= 4:
            u_obj = entry[3]
            if id(u_obj) in seen:
                continue
            seen.add(id(u_obj))
            try:
                u_obj.trajectory.close()
            except Exception:
                pass


def run_stages(
    ctx: TrajectoryContext,
    stages: Optional[List[Tuple[str, Callable[[TrajectoryContext], StageResult]]]] = None,
) -> int:
    """Execute pipeline stages sequentially, verifying StageResult and handling errors."""
    if stages is None:
        stages = [
            ("Stage 1 (Ingestion & Alignment)", run_stage1),
            ("Stage 2 (Skimming & Sampling)", run_stage2),
            ("Stage 3 (Reduction & Allostery)", run_stage3),
            ("Stage 4 (Clustering)", run_stage4),
            ("Stage 5 (OpenMM Energy & Fusion)", run_stage5),
            ("Stage 6 (Visualization & Reports)", run_stage6),
        ]

    try:
        for stage_label, stage_fn in stages:
            try:
                result = stage_fn(ctx)
            except KeyboardInterrupt:
                logger.warning("Pipeline interrupted by user during %s.", stage_label)
                print(f"\n⚠️ Pipeline interrupted by user during {stage_label}. Exiting cleanly.")
                if ctx.out_dir:
                    log_methodology(ctx, f"Pipeline interrupted by user (KeyboardInterrupt) during {stage_label}.")
                return 130
            except Exception as exc:
                logger.exception("Pipeline failed in %s: %s", stage_label, exc)
                print(f"\n❌ Error in {stage_label}: {exc}")
                if ctx.out_dir:
                    log_methodology(ctx, f"Pipeline error during {stage_label}: {exc}")
                return 1

            if not isinstance(result, StageResult):
                logger.error("%s did not return a StageResult instance.", stage_label)
                print(f"\n❌ Internal Error in {stage_label}: invalid stage result.")
                if ctx.out_dir:
                    log_methodology(ctx, f"Internal error in {stage_label}: invalid stage result.")
                return 1

            if not result.success or result.exit_code != 0:
                err_msg = result.error or f"exit_code={result.exit_code}"
                logger.error("%s failed: %s", stage_label, err_msg)
                print(f"\n❌ {stage_label} failed: {err_msg}")
                if ctx.out_dir:
                    log_methodology(ctx, f"{stage_label} failed: {err_msg}")
                return result.exit_code if result.exit_code != 0 else 1

            if ctx.energy_interrupted:
                return 0

        return 0
    finally:
        _close_universe_readers(ctx)


def main(
    argv: Optional[List[str]] = None,
    base_dir: str = ".",
    check_deps: bool = True,
) -> int:
    """Main orchestrator entrypoint."""
    raw_args = list(argv) if argv is not None else sys.argv[1:]

    # Fast path for --only_dashboard true
    if "--only_dashboard" in raw_args:
        idx = raw_args.index("--only_dashboard")
        if idx + 1 < len(raw_args) and raw_args[idx + 1].lower() == "true":
            return _run_only_dashboard(raw_args, base_dir=base_dir)

    try:
        config: PipelineConfig = parse_args_to_config(raw_args, base_dir=base_dir)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1

    if config.is_only_dashboard:
        return _run_only_dashboard(raw_args, base_dir=base_dir)

    if check_deps:
        install_deps()

    print("\n==================================================")
    print("   Initializing Post-Generation Analysis (v2.8 MULTI-SOURCE)")
    print("==================================================")
    print("--------------------------------------------------")

    ctx = TrajectoryContext(config=config, out_dir=config.out_dir)
    return run_stages(ctx)


if __name__ == "__main__":
    sys.exit(main())
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/__init__.py"
"""MET Flow (v2.8 MULTI-SOURCE) modular trajectory analysis package."""

from metflow.models import PipelineConfig, StageResult, TrajectoryContext

__version__ = "2.8.0"
__all__ = ["PipelineConfig", "StageResult", "TrajectoryContext", "__version__"]
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/cli.py"
"""CLI argument parsing, path-traversal validation, and parameter persistence."""

import argparse
import os
from typing import List, Optional

from metflow.models import PipelineConfig


def parse_res_string(res_str: str) -> List[int]:
    """Parse residue range/list strings like '10-20, 25 30' into integer residue IDs."""
    res_list: List[int] = []
    for part in res_str.replace(",", " ").split():
        if "-" in part:
            start, end = map(int, part.split("-"))
            res_list.extend(range(start, end + 1))
        else:
            res_list.append(int(part))
    return res_list


def parse_list_arg(arg_str: str) -> List[str]:
    """Parse comma- or whitespace-separated CLI list arguments."""
    if not arg_str:
        return []
    if "," in arg_str:
        return [x.strip() for x in arg_str.split(",") if x.strip()]
    return [x.strip() for x in arg_str.split() if x.strip()]


def validate_out_dir(out_dir: str, base_dir: str = ".") -> str:
    """Resolve out_dir against base_dir and block directory traversal outside base_dir."""
    base_dir_abs = os.path.realpath(base_dir)
    if os.path.isabs(out_dir):
        out_dir_abs = os.path.realpath(out_dir)
    else:
        out_dir_abs = os.path.realpath(os.path.join(base_dir_abs, out_dir))
    try:
        if os.path.commonpath([base_dir_abs, out_dir_abs]) != base_dir_abs:
            raise ValueError(f"Directory traversal blocked: {out_dir}")
    except ValueError as exc:
        raise ValueError(f"Directory traversal blocked: {out_dir}") from exc
    return out_dir_abs


def build_parser() -> argparse.ArgumentParser:
    """Build the ArgumentParser with all 27 MET Flow flags and exact defaults."""
    parser = argparse.ArgumentParser(description="MET Flow Trajectory Analysis & Clustering")
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
    return parser


def parse_args_to_config(
    argv: Optional[List[str]] = None,
    base_dir: str = ".",
) -> PipelineConfig:
    """Parse CLI arguments, validate output directory, and return PipelineConfig."""
    parser = build_parser()
    args = parser.parse_args(argv)
    validated_out_dir = validate_out_dir(args.out_dir, base_dir=base_dir)

    config = PipelineConfig(
        skim=args.skim,
        equator=args.equator,
        top=args.top,
        bottom=args.bottom,
        min_vol=args.min_vol,
        cluster=args.cluster,
        k=args.k,
        rmsd=args.rmsd,
        cnn_eps=args.cnn_eps,
        allostery=args.allostery,
        cavity_res=args.cavity_res,
        loop_res=args.loop_res,
        pca_res=args.pca_res,
        advanced_plots=args.advanced_plots,
        gen_params=args.gen_params,
        dataset_configs=args.dataset_configs,
        pdb=args.pdb,
        xtc=args.xtc,
        multi_datasets=args.multi_datasets,
        dataset_labels=args.dataset_labels,
        out_dir=validated_out_dir,
        calc_energy=args.calc_energy,
        only_energy=args.only_energy,
        only_dashboard=args.only_dashboard,
        max_frames=args.max_frames,
        min_steps=args.min_steps,
        pca_ref=args.pca_ref,
        base_dir=os.path.abspath(base_dir),
    )

    if config.is_only_energy:
        config.skim = "false"
        config.allostery = "false"
        config.cluster = "none"
        config.calc_energy = "true"

    return config


def save_params(config: PipelineConfig, out_dir: str, base_dir: str = ".") -> str:
    """Save run parameters to out_dir/registers/metflow_params.txt without polluting cwd."""
    reg_dir = os.path.join(out_dir, "registers")
    os.makedirs(reg_dir, exist_ok=True)
    param_keys = [
        "skim", "equator", "top", "bottom", "min_vol", "cluster", "k",
        "rmsd", "cnn_eps", "allostery", "cavity_res", "loop_res", "pca_res",
        "advanced_plots", "gen_params", "dataset_configs", "pdb", "xtc",
        "multi_datasets", "dataset_labels", "out_dir", "calc_energy",
        "only_energy", "only_dashboard", "max_frames", "min_steps", "pca_ref",
    ]
    param_str = "\n".join(f"{k}={getattr(config, k)}" for k in param_keys) + "\n"
    out_param_path = os.path.join(reg_dir, "metflow_params.txt")
    with open(out_param_path, "w", encoding="utf-8") as f_param:
        f_param.write(param_str)
    return out_param_path
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/deps.py"
"""Dependency verification and installation helper."""

import subprocess
import sys


def install_deps() -> None:
    """Install missing analysis dependencies if needed.

    Only invoked from the CLI/orchestrator entrypoint when running full analysis,
    never at module import time.
    """
    mod_map = {
        "scikit-learn": "sklearn",
        "scipy": "scipy",
        "MDAnalysis": "MDAnalysis",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "setuptools": "pkg_resources",
    }
    pins = {
        "scikit-learn": "scikit-learn==1.4.0",
        "scipy": "scipy==1.12.0",
        "MDAnalysis": "MDAnalysis==2.7.0",
        "pandas": "pandas==2.2.0",
        "matplotlib": "matplotlib==3.8.3",
        "seaborn": "seaborn==0.13.2",
        "setuptools": "setuptools==69.0.0",
    }
    missing = []
    for pkg_name, mod_name in mod_map.items():
        try:
            __import__(mod_name)
        except ImportError:
            missing.append(pins[pkg_name])
    if missing:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir"] + missing,
                check=True,
            )
        except Exception as exc:
            print(f"⚠️ Dependency notice: {exc}")
            print("👉 Please make sure the MetFlow environment is active: micromamba activate conf_sel_gen")
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/models.py"
"""Core dataclasses for the MET Flow pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PipelineConfig:
    """Configuration for all 27 CLI flags and runtime options."""

    skim: str = "false"
    equator: str = ""
    top: str = ""
    bottom: str = ""
    min_vol: float = 300.0
    cluster: str = "pca_kmeans"
    k: int = 8
    rmsd: float = 2.0
    cnn_eps: float = 3.0
    allostery: str = "false"
    cavity_res: str = ""
    loop_res: str = ""
    pca_res: str = ""
    advanced_plots: str = "false"
    gen_params: str = ""
    dataset_configs: str = ""
    pdb: str = ""
    xtc: str = ""
    multi_datasets: str = "false"
    dataset_labels: str = ""
    out_dir: str = "clustering_results"
    calc_energy: str = "false"
    only_energy: str = "false"
    only_dashboard: str = "false"
    max_frames: int = 0
    min_steps: int = 100
    pca_ref: str = "combined"
    base_dir: str = "."

    @property
    def is_only_energy(self) -> bool:
        return self.only_energy.lower() == "true"

    @property
    def is_only_dashboard(self) -> bool:
        return self.only_dashboard.lower() == "true"

    @property
    def is_skim(self) -> bool:
        return self.skim.lower() == "true"

    @property
    def is_allostery(self) -> bool:
        return self.allostery.lower() == "true"

    @property
    def is_advanced_plots(self) -> bool:
        return self.advanced_plots.lower() == "true"

    @property
    def is_calc_energy(self) -> bool:
        return self.calc_energy.lower() == "true"


@dataclass
class StageResult:
    """Standardized return object for every pipeline stage."""

    stage_name: str
    success: bool
    exit_code: int = 0
    artifacts: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class TrajectoryContext:
    """Shared mutable state passed sequentially across pipeline stages."""

    config: PipelineConfig
    out_dir: str = ""
    dataset_configs: List[Tuple[str, str, str]] = field(default_factory=list)
    master_ref_pdb: str = ""
    methodology: List[str] = field(default_factory=list)

    # Stage 1 state
    temp_universes: List[Any] = field(default_factory=list)
    dataset_seqs: List[str] = field(default_factory=list)
    dataset_matches: List[Tuple[int, int, int, int]] = field(default_factory=list)
    ds_residues_homologous: List[Any] = field(default_factory=list)
    ds_consensus_groups: List[Any] = field(default_factory=list)
    ds_consensus_indexers: List[Any] = field(default_factory=list)
    u_consensus: Any = None
    ref_consensus_mean: Any = None
    merged_ref_path: str = ""
    unified_ref_path: str = ""
    pca_atom_indices: Any = None
    ref_bb_atoms: Any = None
    ref_bb: Any = None
    universes: List[Any] = field(default_factory=list)
    dataset_scales: List[float] = field(default_factory=list)
    dataset_offsets: List[int] = field(default_factory=list)
    dataset_tags: List[str] = field(default_factory=list)
    frame_universe_map: List[Tuple[Any, int]] = field(default_factory=list)
    n_frames_initial: int = 0
    valid_frames: Any = None

    # Stage 2 state
    unique_tags: List[str] = field(default_factory=list)

    # Stage 3 state
    merged_xtc_path: str = ""
    unified_xtc_path: str = ""
    coords_array: Any = None
    rmsd_to_ref: Any = None
    X: Any = None
    X_pca: Any = None
    pca_model: Any = None
    var_explained: Any = None
    df_data: Dict[str, Any] = field(default_factory=dict)

    # Stage 4 state
    rmsd_matrix: Any = None

    # Stage 5 state
    energy_interrupted: bool = False

    # Stage 6 state
    df: Any = None
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage1_ingestion/__init__.py"
"""Stage 1: Dataset ingestion, sequence alignment, and consensus indexing."""

from metflow.stage1_ingestion.loader import run_stage1

__all__ = ["run_stage1"]
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage1_ingestion/alignment.py"
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
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage1_ingestion/indexing.py"
"""Residue/atom constants, canonical ordering, and coordinate indexing helpers."""

from typing import Any, Optional, Tuple, Union
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


def canonical_atom_sort_key(atom_name: str) -> Tuple[int, str]:
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


def heavy_atom_exclusion_expr(u_obj: Any) -> str:
    """Build an MDAnalysis hydrogen-exclusion expression safe for topologies lacking 'elements' or 'types'."""
    parts = ["name H*", "name [0-9]H*"]
    topo = getattr(getattr(u_obj, "universe", u_obj), "_topology", None)
    if topo is not None:
        if hasattr(topo, "types"):
            parts.append("type H")
        if hasattr(topo, "elements"):
            parts.append("element H")
    else:
        parts.append("type H")
    return "not (" + " or ".join(parts) + ")"


def extract_ca_atoms(u_obj: Any) -> Any:
    """Extract one C-alpha (or fallback) atom per protein residue."""
    import MDAnalysis as mda

    if hasattr(u_obj, "select_atoms"):
        res_q = " ".join(PROTEIN_RESNAMES)
        excl = heavy_atom_exclusion_expr(u_obj)
        protein_atoms = u_obj.select_atoms(f"resname {res_q} and {excl}")
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


def precompute_atom_indexer(atom_group: Any) -> Union[slice, np.ndarray]:
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


def extract_group_coords(u_ds: Any, atom_group: Any, out: Optional[np.ndarray] = None) -> np.ndarray:
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
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage1_ingestion/loader.py"
"""Stage 1 loader: dataset resolution, universe initialization, and consensus PDB export."""

import os
import shutil
from typing import List, Tuple

import numpy as np

from metflow.cli import parse_list_arg, parse_res_string, save_params
from metflow.models import PipelineConfig, StageResult, TrajectoryContext
from metflow.stage6_visualization.reports import init_methodology, log_methodology
from metflow.stage1_ingestion.alignment import (
    build_consensus_atom_groups,
    find_consensus_bounds,
    verify_sequence_homology,
)
from metflow.stage1_ingestion.indexing import (
    PROTEIN_RESNAMES,
    extract_ca_atoms,
    heavy_atom_exclusion_expr,
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

    if config.is_only_energy:
        mode_label = "Energy Only (OpenMM)"
    elif config.multi_datasets.lower() == "true" or len(dataset_configs) > 1:
        mode_label = "Multi-Dataset Comparative Analysis"
    else:
        mode_label = "Single-Dataset Analysis"
    init_methodology(ctx, mode_label)
    if config.gen_params:
        log_methodology(ctx, f"Generation Parameters:\n{config.gen_params}")

    temp_universes = []
    prot_res_str = " ".join(PROTEIN_RESNAMES)
    for pdb_p, xtc_p, label in dataset_configs:
        if not os.path.exists(pdb_p):
            raise FileNotFoundError(f"PDB file not found: {pdb_p}")
        if not os.path.exists(xtc_p):
            raise FileNotFoundError(f"XTC file not found: {xtc_p}")
        u_temp = mda.Universe(pdb_p, xtc_p)
        excl = heavy_atom_exclusion_expr(u_temp)
        protein_atoms = u_temp.select_atoms(f"resname {prot_res_str} and {excl}")
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
    log_methodology(ctx, seq_summary_text)

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
        log_methodology(
            ctx,
            f"Alignment and Global PCA restricted to {len(rel_indices)} C-alpha residues from: {config.pca_res}",
        )
    else:
        pca_atom_indices = ca_indices
        log_methodology(ctx, f"Alignment and Global PCA used {len(ca_indices)} common C-alpha backbone atoms.")

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
    log_methodology(
        ctx,
        f"Loaded {len(dataset_configs)} dataset(s) ({', '.join(labels)}). Total conformations: {total_frames}. "
        f"Consensus length: {consensus_len} residues ({len(u_consensus.atoms)} heavy atoms).",
    )

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
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage2_skimming/__init__.py"
"""Stage 2: Conformational skimming and per-dataset downsampling."""

from metflow.stage2_skimming.skimmer import run_stage2

__all__ = ["run_stage2"]
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage2_skimming/sampling.py"
"""Per-dataset uniform strided downsampling."""

from typing import List, Tuple
import numpy as np


def downsample_per_dataset(
    valid_frames: np.ndarray,
    dataset_tags: List[str],
    max_frames: int,
) -> Tuple[np.ndarray, List[str], str]:
    """Uniformly sample up to max_frames per dataset, preserving order."""
    unique_tags = list(dict.fromkeys(dataset_tags))
    if max_frames > 0:
        dataset_tags_arr = np.array(dataset_tags)[valid_frames]
        sampled_parts = []
        for t in unique_tags:
            t_idxs = valid_frames[dataset_tags_arr == t]
            if len(t_idxs) > max_frames:
                st = max(1, len(t_idxs) // max_frames)
                sampled_parts.append(t_idxs[::st][:max_frames])
            else:
                sampled_parts.append(t_idxs)
        sampled_frames = np.sort(np.concatenate(sampled_parts)) if sampled_parts else np.array([], dtype=int)
        msg = (
            f"Per-dataset sampling: up to {max_frames} frames per dataset "
            f"({len(sampled_frames)} total)."
        )
        print(
            f"\n[Per-Dataset Sampling] Uniformly sampled up to {max_frames} frames per dataset "
            f"({len(sampled_frames)} total frames across {len(unique_tags)} dataset(s))."
        )
    else:
        sampled_frames = valid_frames
        msg = f"No downsampling applied: 100% of frames retained ({len(sampled_frames)} total)."
        print(
            f"\n[No Downsampling] Preserved 100% of frames ({len(sampled_frames)} total frames "
            f"across {len(unique_tags)} dataset(s))."
        )

    filtered_tags = [dataset_tags[i] for i in sampled_frames]
    return sampled_frames, filtered_tags, msg
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage2_skimming/skimmer.py"
"""ConvexHull volume skimming, gate distance filtering, and PyMOL hull visualization."""

from multiprocessing import Pool
import os
from typing import List, Optional, Tuple
import numpy as np
from scipy.spatial import ConvexHull

from metflow.models import StageResult, TrajectoryContext
from metflow.stage1_ingestion.indexing import extract_group_coords, precompute_atom_indexer
from metflow.stage2_skimming.sampling import downsample_per_dataset
from metflow.stage6_visualization.reports import log_methodology


def _calc_vol(coords: np.ndarray) -> float:
    """Compute ConvexHull volume for a single frame's coordinates (0.0 on degenerate hull)."""
    try:
        return float(ConvexHull(coords).volume)
    except Exception:
        return 0.0


def compute_hull_volumes(skim_coords_all: np.ndarray, use_multiprocessing: bool = True) -> np.ndarray:
    """Compute ConvexHull volumes across frames with Windows-safe multiprocessing fallback."""
    if len(skim_coords_all) == 0:
        return np.array([], dtype=np.float64)
    if use_multiprocessing and len(skim_coords_all) >= 64:
        try:
            with Pool() as p:
                return np.array(p.map(_calc_vol, skim_coords_all), dtype=np.float64)
        except Exception:
            pass
    return np.array([_calc_vol(c) for c in skim_coords_all], dtype=np.float64)


def write_visualize_hull_py(
    out_dir: str,
    verts: np.ndarray,
    top_res: int,
    bot_res: int,
) -> str:
    """Write a PyMOL CGO visualization script for the convex hull and gate distance."""
    hull = ConvexHull(verts)
    cgo = ["from pymol.cgo import *", "from pymol import cmd", "obj = []"]
    for s in hull.simplices:
        v1, v2, v3 = verts[s[0]], verts[s[1]], verts[s[2]]
        cgo.append(
            f"obj.extend([BEGIN, TRIANGLES, COLOR, 0.4, 0.8, 0.8, "
            f"VERTEX, {v1[0]:.3f}, {v1[1]:.3f}, {v1[2]:.3f}, "
            f"VERTEX, {v2[0]:.3f}, {v2[1]:.3f}, {v2[2]:.3f}, "
            f"VERTEX, {v3[0]:.3f}, {v3[1]:.3f}, {v3[2]:.3f}, END])"
        )
    cgo.extend([
        "cmd.load_cgo(obj, 'internal_volume_mesh')",
        "cmd.set('cgo_transparency', 0.4, 'internal_volume_mesh')",
        f"cmd.distance('gate_distance', 'resid {top_res} and name CA', 'resid {bot_res} and name CA')",
    ])
    out_path = os.path.join(out_dir, "visualize_hull.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cgo) + "\n")
    return out_path


def skim_frames(ctx: TrajectoryContext) -> Tuple[np.ndarray, Optional[str]]:
    """Filter frames by ConvexHull volume > min_vol and gate distance > 8.0 A."""
    import MDAnalysis as mda

    config = ctx.config
    if not (config.is_skim and config.equator and config.top and config.bottom):
        print("Skipped skimming.")
        log_methodology(ctx, "\n--- Conformational Skimming ---\nNo skimming was applied.")
        return ctx.valid_frames, None

    eq_res = [int(x) for x in config.equator.replace(",", " ").split()]
    top_res = int(config.top.replace(",", ""))
    bot_res = int(config.bottom.replace(",", ""))
    all_skim_res = eq_res + [top_res, bot_res]
    res_str = " ".join(str(x) for x in all_skim_res)

    print("\n--- Starting Tandem Skimming ---")
    skim_coords_list = []
    top_idx_in_s = len(eq_res)
    bot_idx_in_s = len(eq_res) + 1

    for idx, u_ds in enumerate(ctx.universes):
        s_atom_list = []
        for r_num in all_skim_res:
            r_idx = r_num - 1
            if 0 <= r_idx < len(ctx.ds_residues_homologous[idx]):
                target_res = ctx.ds_residues_homologous[idx][r_idx]
                ca_matches = target_res.atoms.select_atoms("name CA Ca ca CA* C-alpha Calpha")
                s_atom_list.append(ca_matches[0] if len(ca_matches) > 0 else target_res.atoms[0])
            else:
                fallback_atoms = u_ds.select_atoms(f"resid {r_num} and name CA")
                if len(fallback_atoms) > 0:
                    s_atom_list.append(fallback_atoms[0])
                else:
                    raise ValueError(
                        f"Residue {r_num} CA not found in dataset '{ctx.temp_universes[idx][2]}'. Aborting skim."
                    )

        s_atoms = mda.AtomGroup(s_atom_list)
        s_coords = extract_group_coords(u_ds, precompute_atom_indexer(s_atoms))
        if ctx.dataset_scales[idx] != 1.0:
            s_coords *= ctx.dataset_scales[idx]
        skim_coords_list.append(s_coords)

    skim_coords_all = (
        skim_coords_list[0]
        if len(skim_coords_list) == 1
        else np.concatenate(skim_coords_list, axis=0)
    )
    heights = np.linalg.norm(
        skim_coords_all[:, top_idx_in_s, :] - skim_coords_all[:, bot_idx_in_s, :], axis=1
    )

    print("Computing volumes with multiprocessing...")
    vols = compute_hull_volumes(skim_coords_all, use_multiprocessing=True)

    passed_mask = (vols > config.min_vol) & (heights > 8.0)
    valid_frames = np.where(passed_mask)[0]
    print(f"Skimming Complete: {len(valid_frames)} / {ctx.n_frames_initial} passed.")
    log_methodology(
        ctx,
        f"\n--- Conformational Skimming ---\n"
        f"Tandem skim (ConvexHull Volume > {config.min_vol} Å³, Gate Dist > 8.0 Å; "
        f"equator={config.equator}, top={config.top}, bottom={config.bottom}). "
        f"Surviving: {len(valid_frames)} / {ctx.n_frames_initial} frames.",
    )

    hull_script: Optional[str] = None
    if len(valid_frames) > 0:
        u_first, l_first = ctx.frame_universe_map[valid_frames[0]]
        u_first.trajectory[l_first]
        skim_atoms_first = u_first.select_atoms(f"resid {res_str} and name CA")
        hull_script = write_visualize_hull_py(ctx.out_dir, skim_atoms_first.positions, top_res, bot_res)
        print(f"Generated PyMOL visualization script: {hull_script}")

    return valid_frames, hull_script


def run_stage2(ctx: TrajectoryContext) -> StageResult:
    """Execute Stage 2: conformational skimming and per-dataset downsampling."""
    artifacts: List[str] = []
    valid_frames, hull_script = skim_frames(ctx)
    if hull_script:
        artifacts.append(hull_script)

    if len(valid_frames) < 2:
        raise ValueError(
            f"Too few valid frames remaining ({len(valid_frames)}) to perform clustering (minimum 2 required)."
        )

    ctx.unique_tags = list(dict.fromkeys(ctx.dataset_tags))
    sampled_frames, filtered_tags, sampling_msg = downsample_per_dataset(
        valid_frames, ctx.dataset_tags, ctx.config.max_frames
    )
    ctx.valid_frames = sampled_frames
    ctx.dataset_tags = filtered_tags
    log_methodology(ctx, sampling_msg)

    return StageResult(
        stage_name="stage2_skimming",
        success=True,
        exit_code=0,
        artifacts=artifacts,
        metrics={"n_valid_frames": int(len(sampled_frames)), "n_datasets": len(ctx.unique_tags)},
    )
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage3_reduction/__init__.py"
"""Stage 3: Kabsch alignment, merged trajectory export, PCA, and allostery."""

from metflow.stage3_reduction.allostery import run_stage3

__all__ = ["run_stage3"]
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage3_reduction/allostery.py"
"""Advanced allostery (DCCM, Cavity Sub-PCA, Loop RMSD) and Stage 3 orchestration."""

import os
from typing import Any, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

from metflow.cli import parse_res_string
from metflow.models import StageResult, TrajectoryContext
from metflow.stage1_ingestion.indexing import precompute_atom_indexer
from metflow.stage3_reduction.kabsch import export_merged_trajectory
from metflow.stage3_reduction.pca import compute_pca
from metflow.stage6_visualization.reports import log_methodology


def compute_dccm(ca_coords: np.ndarray) -> np.ndarray:
    """Compute Dynamic Cross-Correlation Matrix (N_ca x N_ca) from aligned C-alpha coords."""
    n_frames, n_ca, _ = ca_coords.shape
    mean_coords = np.mean(ca_coords, axis=0)
    fluct = ca_coords - mean_coords
    fluct_flat = fluct.transpose(1, 0, 2).reshape(n_ca, -1)
    cov = (fluct_flat @ fluct_flat.T) / max(1, n_frames)
    var = np.diag(cov)
    std = np.sqrt(np.maximum(var, 1e-12))
    return cov / np.outer(std, std)


def compute_cavity_subpca(
    cav_coords: np.ndarray,
    loop_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2-comp Sub-PCA on cavity coordinates and Loop RMSD to mean."""
    n_frames = cav_coords.shape[0]
    X_cav = cav_coords.reshape(n_frames, -1)
    n_comp = max(1, min(2, X_cav.shape[0], X_cav.shape[1]))
    sub_pca_model = PCA(n_components=n_comp)
    cav_pca_raw = sub_pca_model.fit_transform(X_cav)
    sub_var_raw = sub_pca_model.explained_variance_ratio_ * 100.0

    if n_comp < 2:
        cav_pca = np.zeros((n_frames, 2), dtype=cav_pca_raw.dtype)
        cav_pca[:, :n_comp] = cav_pca_raw
        sub_var_explained = np.zeros(2, dtype=np.float64)
        sub_var_explained[:n_comp] = sub_var_raw
    else:
        cav_pca = cav_pca_raw
        sub_var_explained = sub_var_raw

    mean_loop = np.mean(loop_coords, axis=0)
    loop_rmsd = np.sqrt(np.mean(np.sum((loop_coords - mean_loop) ** 2, axis=2), axis=1))
    return cav_pca, sub_var_explained, loop_rmsd, sub_pca_model.components_


def run_allostery(ctx: TrajectoryContext) -> List[str]:
    """Run DCCM and optional Cavity Sub-PCA + Loop RMSD analysis."""
    import MDAnalysis as mda

    config = ctx.config
    if not config.is_allostery:
        return []

    out_dir = ctx.out_dir
    valid_frames = ctx.valid_frames
    artifacts: List[str] = []

    print("\n--- Running Advanced Allostery Metrics ---")
    log_methodology(ctx, "\n--- Advanced Allostery Metrics ---")

    print("Calculating Dynamic Cross-Correlation Matrix (DCCM)...")
    dccm = compute_dccm(ctx.coords_array)

    dccm_path = os.path.join(out_dir, "dccm_heatmap.png")
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        dccm,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=20,
        yticklabels=20,
        cbar_kws={"label": "Cross-Correlation Coefficient"},
    )
    plt.title("Dynamic Cross-Correlation Matrix (DCCM)")
    plt.xlabel("Residue Index")
    plt.ylabel("Residue Index")
    plt.savefig(dccm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved DCCM Heatmap to {dccm_path}")
    log_methodology(ctx, "Generated Dynamic Cross-Correlation Matrix (DCCM).")
    artifacts.append(dccm_path)

    if config.cavity_res and config.loop_res:
        cav_res = parse_res_string(config.cavity_res)
        loop_res = parse_res_string(config.loop_res)
        cav_str = " ".join(str(x) for x in cav_res)

        cav_indexers = []
        loop_indexers = []
        for idx, u_ds in enumerate(ctx.universes):
            c_atoms = []
            for r_num in cav_res:
                r_idx = r_num - 1
                if 0 <= r_idx < len(ctx.ds_residues_homologous[idx]):
                    t_res = ctx.ds_residues_homologous[idx][r_idx]
                    ca_m = t_res.atoms.select_atoms("name CA Ca ca CA* C-alpha Calpha")
                    c_atoms.append(ca_m[0] if len(ca_m) > 0 else t_res.atoms[0])
                else:
                    fb = u_ds.select_atoms(f"resid {r_num} and name CA")
                    if len(fb) > 0:
                        c_atoms.append(fb[0])
            l_atoms = []
            for r_num in loop_res:
                r_idx = r_num - 1
                if 0 <= r_idx < len(ctx.ds_residues_homologous[idx]):
                    t_res = ctx.ds_residues_homologous[idx][r_idx]
                    ca_m = t_res.atoms.select_atoms("name CA Ca ca CA* C-alpha Calpha")
                    l_atoms.append(ca_m[0] if len(ca_m) > 0 else t_res.atoms[0])
                else:
                    fb = u_ds.select_atoms(f"resid {r_num} and name CA")
                    if len(fb) > 0:
                        l_atoms.append(fb[0])
            cav_indexers.append(precompute_atom_indexer(mda.AtomGroup(c_atoms) if c_atoms else u_ds.atoms[:0]))
            loop_indexers.append(precompute_atom_indexer(mda.AtomGroup(l_atoms) if l_atoms else u_ds.atoms[:0]))

        def _indexer_len(idxer: Any) -> int:
            return max(0, idxer.stop - idxer.start) if isinstance(idxer, slice) else len(idxer)

        n_cav_atoms = _indexer_len(cav_indexers[0]) if cav_indexers else 0
        n_loop_atoms = _indexer_len(loop_indexers[0]) if loop_indexers else 0
        cav_coords = np.empty((len(valid_frames), n_cav_atoms, 3), dtype=np.float32)
        loop_coords = np.empty((len(valid_frames), n_loop_atoms, 3), dtype=np.float32)

        for idx, u_ds in enumerate(ctx.universes):
            ds_start = ctx.dataset_offsets[idx]
            ds_end = ctx.dataset_offsets[idx + 1]
            ds_valid_mask = (valid_frames >= ds_start) & (valid_frames < ds_end)
            if not np.any(ds_valid_mask):
                continue

            ds_valid_out_indices = np.where(ds_valid_mask)[0]
            ds_valid_local_frames = valid_frames[ds_valid_mask] - ds_start
            c_idxer = cav_indexers[idx]
            l_idxer = loop_indexers[idx]
            scale = ctx.dataset_scales[idx]
            ptr = 0
            n_needed = len(ds_valid_local_frames)

            for f_i, ts in enumerate(u_ds.trajectory):
                if ptr >= n_needed:
                    break
                if f_i == ds_valid_local_frames[ptr]:
                    out_i = ds_valid_out_indices[ptr]
                    if n_cav_atoms > 0:
                        cav_coords[out_i] = ts.positions[c_idxer]
                    if n_loop_atoms > 0:
                        loop_coords[out_i] = ts.positions[l_idxer]
                    ptr += 1

            if scale != 1.0:
                if n_cav_atoms > 0:
                    cav_coords[ds_valid_out_indices] *= scale
                if n_loop_atoms > 0:
                    loop_coords[ds_valid_out_indices] *= scale

        if cav_coords.shape[1] > 0 and loop_coords.shape[1] > 0:
            print("Calculating Targeted Sub-PCA for Allosteric Linkage...")
            cav_pca, sub_var_explained, loop_rmsd, sub_components = compute_cavity_subpca(cav_coords, loop_coords)

            subpca_plot_path = os.path.join(out_dir, "allostery_subpca_plot.png")
            plt.figure(figsize=(9, 7))
            scatter = plt.scatter(cav_pca[:, 0], cav_pca[:, 1], c=loop_rmsd, cmap="viridis", alpha=0.7, s=15)
            plt.colorbar(scatter, label="Loop RMSD to Mean (Å)")
            plt.title("Cavity PCA colored by Loop Fluctuation (Allostery Plot)")
            plt.xlabel(f"Cavity PC1 ({sub_var_explained[0]:.1f}%)")
            plt.ylabel(f"Cavity PC2 ({sub_var_explained[1]:.1f}%)")
            plt.savefig(subpca_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved Allostery Plot to {subpca_plot_path}")
            artifacts.append(subpca_plot_path)
            log_methodology(
                ctx,
                f"Targeted Sub-PCA applied to Cavity (res {cav_res}), colored by Loop RMSD (Å) (res {loop_res}). "
                f"Variance: PC1={sub_var_explained[0]:.1f}%, PC2={sub_var_explained[1]:.1f}%",
            )

            sub_n_atoms = cav_coords.shape[1]
            u_first, l_first = ctx.frame_universe_map[valid_frames[0]]
            u_first.trajectory[l_first]
            sub_resids = u_first.select_atoms(f"resid {cav_str} and name CA").resids
            if len(sub_resids) == sub_n_atoms:
                sub_atom_contrib_pc1 = np.sqrt(np.sum(sub_components[0].reshape(sub_n_atoms, 3) ** 2, axis=1))
                sub_atom_contrib_pc2 = (
                    np.sqrt(np.sum(sub_components[1].reshape(sub_n_atoms, 3) ** 2, axis=1))
                    if sub_components.shape[0] > 1
                    else np.zeros(sub_n_atoms)
                )

                sub_unique_resids, sub_res_inv = np.unique(sub_resids, return_inverse=True)
                sub_res_contrib_pc1 = np.bincount(sub_res_inv, weights=sub_atom_contrib_pc1)
                sub_res_contrib_pc2 = np.bincount(sub_res_inv, weights=sub_atom_contrib_pc2)

                loadings_path = os.path.join(out_dir, "allostery_subpca_loadings.png")
                plt.figure(figsize=(8, 4))
                width = 0.35
                x = np.arange(len(sub_unique_resids))
                plt.bar(x - width / 2, sub_res_contrib_pc1, width, label="PC1 Contribution", alpha=0.85, color="#0072B2")
                plt.bar(x + width / 2, sub_res_contrib_pc2, width, label="PC2 Contribution", alpha=0.85, color="#D55E00")
                plt.xticks(x, sub_unique_resids)
                plt.xlabel("Cavity Residue Number")
                plt.ylabel("Loading Magnitude (RMS, Å)")
                plt.title("Per-Residue Contribution to Cavity Sub-PCA")
                plt.legend()
                plt.grid(True, alpha=0.3, axis="y")
                plt.savefig(loadings_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Saved Sub-PCA Loadings Plot to {loadings_path}")
                artifacts.append(loadings_path)

    return artifacts


def run_stage3(ctx: TrajectoryContext) -> StageResult:
    """Execute Stage 3: streaming Kabsch export, global/ref PCA, and allostery."""
    coords_array, rmsd_to_ref = export_merged_trajectory(ctx)
    ctx.coords_array = coords_array
    ctx.rmsd_to_ref = rmsd_to_ref
    log_methodology(
        ctx,
        f"\n--- Trajectory Alignment & Dimensionality Reduction ---\n"
        f"Exported Kabsch-aligned merged trajectory ({len(ctx.valid_frames)} frames) to {ctx.merged_xtc_path}.",
    )

    X = coords_array.reshape(len(ctx.valid_frames), -1)
    ctx.X = X

    X_pca, pca_model, var_explained, method_note = compute_pca(
        X, ctx.dataset_tags, ctx.unique_tags, pca_ref=ctx.config.pca_ref, n_components=3
    )
    ctx.X_pca = X_pca
    ctx.pca_model = pca_model
    ctx.var_explained = var_explained
    if method_note:
        log_methodology(ctx, method_note)
    log_methodology(
        ctx,
        f"PCA Variance Explained: PC1={var_explained[0]:.1f}%, PC2={var_explained[1]:.1f}%, PC3={var_explained[2]:.1f}%",
    )

    local_counts = {}
    ds_totals = {}
    for tag in ctx.dataset_tags:
        ds_totals[tag] = ds_totals.get(tag, 0) + 1
    local_frame_idx = np.zeros(len(ctx.valid_frames), dtype=int)
    ds_progress_pct = np.zeros(len(ctx.valid_frames), dtype=float)
    for idx_f, tag in enumerate(ctx.dataset_tags):
        k_local = local_counts.get(tag, 0)
        local_frame_idx[idx_f] = k_local
        tot_ds = ds_totals[tag]
        ds_progress_pct[idx_f] = (100.0 * k_local / (tot_ds - 1)) if tot_ds > 1 else 0.0
        local_counts[tag] = k_local + 1

    ctx.df_data = {
        "Frame_Index": np.arange(len(ctx.valid_frames)),
        "Local_Frame_Index": local_frame_idx,
        "Dataset_Progress_Pct": np.round(ds_progress_pct, 2),
        "Dataset": ctx.dataset_tags,
        "RMSD_to_Reference": rmsd_to_ref,
        "PCA_1": X_pca[:, 0],
        "PCA_2": X_pca[:, 1],
        "PCA_3": X_pca[:, 2],
    }

    artifacts = [ctx.merged_xtc_path, ctx.unified_xtc_path]
    artifacts.extend(run_allostery(ctx))

    return StageResult(
        stage_name="stage3_reduction",
        success=True,
        exit_code=0,
        artifacts=artifacts,
        metrics={
            "n_frames": int(len(ctx.valid_frames)),
            "pca_var_explained": [float(v) for v in var_explained],
        },
    )
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage3_reduction/kabsch.py"
"""Batched Kabsch superposition and streaming unified trajectory export."""

import os
from typing import Tuple
import numpy as np

from metflow.models import TrajectoryContext
from metflow.stage1_ingestion.loader import make_alias


def batched_kabsch(
    P_raw: np.ndarray,
    Q_ref: np.ndarray,
    out: np.ndarray = None,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Superpose (N, A, 3) frames in P_raw onto centered (A, 3) reference Q_ref."""
    n_samples = len(P_raw)
    if n_samples == 0:
        return P_raw
    if out is None:
        out = np.empty_like(P_raw)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = P_raw[start:end]
        c_mean = chunk.mean(axis=1, keepdims=True)
        c_centered = chunk - c_mean
        H = np.matmul(c_centered.swapaxes(1, 2), Q_ref)
        U, _, Vt = np.linalg.svd(H)
        R = np.matmul(U, Vt)
        dets = (
            R[:, 0, 0] * (R[:, 1, 1] * R[:, 2, 2] - R[:, 1, 2] * R[:, 2, 1])
            - R[:, 0, 1] * (R[:, 1, 0] * R[:, 2, 2] - R[:, 1, 2] * R[:, 2, 0])
            + R[:, 0, 2] * (R[:, 1, 0] * R[:, 2, 1] - R[:, 1, 1] * R[:, 2, 0])
        )
        neg_mask = dets < 0
        if np.any(neg_mask):
            U[neg_mask, :, 2] *= -1
            R[neg_mask] = np.matmul(U[neg_mask], Vt[neg_mask])
        np.matmul(c_centered, R, out=out[start:end])
    return out


def export_merged_trajectory(
    ctx: TrajectoryContext,
    stream_chunk: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stream-align consensus heavy atoms, write merged_trajectory.xtc, and compute RMSD."""
    import MDAnalysis as mda

    out_dir = ctx.out_dir
    valid_frames = ctx.valid_frames
    u_consensus = ctx.u_consensus
    ref_consensus_mean = ctx.ref_consensus_mean
    pca_atom_indices = ctx.pca_atom_indices

    merged_xtc_path = os.path.join(out_dir, "merged_trajectory.xtc")
    unified_xtc_path = os.path.join(out_dir, "unified_trajectory.xtc")
    ctx.merged_xtc_path = merged_xtc_path
    ctx.unified_xtc_path = unified_xtc_path

    print("\n--- Exporting Physical Unified Trajectory ---")
    print(
        f"Writing {len(valid_frames)} aligned frames ({len(u_consensus.atoms)} atoms per frame) "
        f"to {merged_xtc_path}..."
    )

    coords_array = np.empty((len(valid_frames), len(pca_atom_indices), 3), dtype=np.float32)
    n_consensus_atoms = u_consensus.atoms.n_atoms
    c_batch_buf = np.empty((stream_chunk, n_consensus_atoms, 3), dtype=np.float32)
    dest_idx_buf = np.empty(stream_chunk, dtype=np.intp)

    with mda.Writer(merged_xtc_path, n_atoms=n_consensus_atoms) as w:
        def _flush_batch(count: int, scale: float) -> None:
            c_batch = c_batch_buf[:count]
            if scale != 1.0:
                c_batch *= scale
            batched_kabsch(c_batch, ref_consensus_mean, out=c_batch)
            for pos in c_batch:
                u_consensus.atoms.positions = pos
                w.write(u_consensus.atoms)
            coords_array[dest_idx_buf[:count]] = c_batch[:, pca_atom_indices, :]

        for ds_idx, (_, _, _, u_ds, _, _) in enumerate(ctx.temp_universes):
            ds_start = ctx.dataset_offsets[ds_idx]
            ds_end = ctx.dataset_offsets[ds_idx + 1]
            ds_valid_mask = (valid_frames >= ds_start) & (valid_frames < ds_end)
            if not np.any(ds_valid_mask):
                continue

            ds_valid_out_indices = np.where(ds_valid_mask)[0]
            ds_valid_local_frames = valid_frames[ds_valid_mask] - ds_start
            indexer = ctx.ds_consensus_indexers[ds_idx]
            scale = ctx.dataset_scales[ds_idx]

            buf_ptr = 0
            ptr = 0
            n_needed = len(ds_valid_local_frames)

            for f_i, ts in enumerate(u_ds.trajectory):
                if ptr >= n_needed:
                    break
                if f_i == ds_valid_local_frames[ptr]:
                    c_batch_buf[buf_ptr] = ts.positions[indexer]
                    dest_idx_buf[buf_ptr] = ds_valid_out_indices[ptr]
                    buf_ptr += 1
                    ptr += 1

                    if buf_ptr == stream_chunk:
                        _flush_batch(buf_ptr, scale)
                        buf_ptr = 0

            if buf_ptr > 0:
                _flush_batch(buf_ptr, scale)

    print(f"  -> Exported merged trajectory to {merged_xtc_path}")
    make_alias(merged_xtc_path, unified_xtc_path, out_dir)
    print(f"  -> Created unified trajectory alias at {unified_xtc_path}")

    diff_ref = coords_array - ctx.ref_bb
    rmsd_to_ref = np.sqrt(np.einsum("nai,nai->n", diff_ref, diff_ref) / coords_array.shape[1])
    return coords_array, rmsd_to_ref
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage3_reduction/pca.py"
"""Global and reference-projected PCA computation."""

from typing import List, Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA


def compute_pca(
    X: np.ndarray,
    dataset_tags: List[str],
    unique_tags: List[str],
    pca_ref: str = "combined",
    n_components: int = 3,
) -> Tuple[np.ndarray, PCA, np.ndarray, Optional[str]]:
    """Compute 3-component PCA (with safe padding when n_samples < 3 or n_features < 3)."""
    n_samples, n_features = X.shape
    actual_comp = max(1, min(n_components, n_samples, n_features))
    pca_model = PCA(n_components=actual_comp)
    method_note: Optional[str] = None

    use_ref = bool(
        pca_ref
        and (
            pca_ref.lower() in ["first", "1", "ref", "reference"]
            or pca_ref.lower() != "combined"
        )
    )
    if use_ref and unique_tags:
        target_tag = unique_tags[0]
        if pca_ref.lower() not in ["first", "1", "ref", "reference"]:
            for t in unique_tags:
                if pca_ref.lower() in t.lower():
                    target_tag = t
                    break
        fit_mask = np.array([t == target_tag for t in dataset_tags])
        if np.sum(fit_mask) >= max(2, actual_comp):
            pca_model.fit(X[fit_mask])
            X_pca_raw = pca_model.transform(X)
            print(
                f"\n[Reference PCA Projection] Fitted PCA model strictly on reference dataset "
                f"'{target_tag}' ({np.sum(fit_mask)} frames) and projected all datasets into "
                f"'{target_tag}'s PCA coordinate space."
            )
            method_note = (
                f"Fitted PCA on reference dataset '{target_tag}' ({np.sum(fit_mask)} frames) "
                f"and projected all datasets into reference PCA space."
            )
        else:
            X_pca_raw = pca_model.fit_transform(X)
    else:
        X_pca_raw = pca_model.fit_transform(X)

    var_raw = pca_model.explained_variance_ratio_ * 100.0

    # Pad to n_components columns so downstream PCA_1..PCA_3 always exist even for n_frames == 2
    if actual_comp < n_components:
        X_pca = np.zeros((n_samples, n_components), dtype=X_pca_raw.dtype)
        X_pca[:, :actual_comp] = X_pca_raw
        var_explained = np.zeros(n_components, dtype=np.float64)
        var_explained[:actual_comp] = var_raw
    else:
        X_pca = X_pca_raw
        var_explained = var_raw

    return X_pca, pca_model, var_explained, method_note
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage4_clustering/__init__.py"
"""Stage 4: Pairwise RMSD matrix and clustering algorithms (KMeans, Daura, CNN)."""

from metflow.stage4_clustering.cnn import run_stage4

__all__ = ["run_stage4"]
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage4_clustering/cnn.py"
"""CNN (DBSCAN) density clustering and Stage 4 orchestration."""

import os
from typing import List
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from metflow.models import StageResult, TrajectoryContext
from metflow.stage4_clustering.daura import run_daura
from metflow.stage4_clustering.kmeans import run_kmeans
from metflow.stage4_clustering.pairwise import compute_pairwise_rmsd, make_centroid_extractor
from metflow.stage6_visualization.reports import log_methodology


def run_cnn(rmsd_matrix: np.ndarray, eps: float, min_samples: int = 3) -> np.ndarray:
    """Run DBSCAN density clustering on precomputed pairwise RMSD matrix."""
    print(f"\n[ CNN ] Running Optimized DBSCAN with eps={eps}...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1)
    return dbscan.fit_predict(rmsd_matrix)


def run_stage4(ctx: TrajectoryContext) -> StageResult:
    """Execute Stage 4: pairwise RMSD, requested clustering methods, and centroid export."""
    config = ctx.config
    if config.cluster == "none":
        return StageResult(
            stage_name="stage4_clustering",
            success=True,
            exit_code=0,
            metrics={"skipped": True},
        )

    out_dir = ctx.out_dir
    valid_frames = ctx.valid_frames
    X = ctx.X
    X_pca = ctx.X_pca
    dataset_tags = ctx.dataset_tags
    rmsd_to_ref = ctx.rmsd_to_ref
    artifacts: List[str] = []

    print("\n--- Starting Clustering Modules ---")
    log_methodology(ctx, "\n--- Clustering Analysis ---")

    methods_to_run = ["pca_kmeans", "daura", "cnn"] if config.cluster == "all" else [config.cluster]
    allowed_methods = {"pca_kmeans", "daura", "cnn"}
    for m in methods_to_run:
        if m not in allowed_methods:
            raise ValueError(f"Unknown clustering method: {m}")

    if config.cluster in ["daura", "cnn", "all"]:
        print("Calculating pairwise distance matrix...")
        ctx.rmsd_matrix = compute_pairwise_rmsd(X, ctx.coords_array.shape[1])

    extract_c = make_centroid_extractor(
        ctx.merged_ref_path, ctx.merged_xtc_path, valid_frames, ctx.frame_universe_map
    )

    try:
        for m in methods_to_run:
            method_dir = os.path.join(out_dir, m)
            os.makedirs(method_dir, exist_ok=True)

            if m == "pca_kmeans":
                labels, centers, best_k, auto_det = run_kmeans(X_pca, config.k)
                ctx.df_data["PCA_KMeans_Cluster"] = labels
                for cid in range(best_k):
                    pts = X_pca[labels == cid]
                    if len(pts) > 0:
                        c_idx = int(np.where(labels == cid)[0][np.argmin(np.linalg.norm(pts - centers[cid], axis=1))])
                        artifacts.append(extract_c("pca_kmeans", c_idx, cid, method_dir))
                log_methodology(
                    ctx,
                    f"PCA+KMeans (k={best_k}) used."
                    + (" Optimal 'k' detected via Silhouette Score." if auto_det else ""),
                )
            elif m == "daura":
                labels, centroids = run_daura(ctx.rmsd_matrix, config.rmsd, config.k)
                ctx.df_data["Daura_Cluster"] = labels
                for cid, c_idx in enumerate(centroids):
                    artifacts.append(extract_c("daura", c_idx, cid, method_dir))
                log_methodology(
                    ctx,
                    f"Daura greedy clustering (RMSD cutoff {config.rmsd} Å) extracted {len(centroids)} centroids.",
                )
            elif m == "cnn":
                labels = run_cnn(ctx.rmsd_matrix, config.cnn_eps, min_samples=3)
                ctx.df_data["CNN_Cluster"] = labels
                for cid in set(labels):
                    if cid == -1:
                        continue
                    idxs = np.where(labels == cid)[0]
                    sums = ctx.rmsd_matrix[np.ix_(idxs, idxs)].sum(axis=1)
                    center = int(idxs[np.argmin(sums)])
                    artifacts.append(extract_c("cnn", center, int(cid), method_dir))
                log_methodology(
                    ctx,
                    f"CNN/DBSCAN: Density-based clustering (eps={config.cnn_eps} Å, min_samples=3) extracted "
                    f"{len(set(labels) - {-1})} clusters.",
                )
            else:
                raise ValueError(f"Unknown clustering method: {m}")

            df_method = pd.DataFrame({
                "Frame_Index": np.arange(len(valid_frames)),
                "Dataset": dataset_tags,
                "RMSD_to_Ref": rmsd_to_ref,
                "PCA_1": X_pca[:, 0],
                "PCA_2": X_pca[:, 1],
                "PCA_3": X_pca[:, 2],
                "Cluster_ID": labels,
            })
            csv_path = os.path.join(method_dir, f"{m}_metrics.csv")
            df_method.to_csv(csv_path, index=False)
            artifacts.append(csv_path)
    finally:
        if hasattr(extract_c, "close"):
            try:
                extract_c.close()
            except Exception:
                pass

    return StageResult(
        stage_name="stage4_clustering",
        success=True,
        exit_code=0,
        artifacts=artifacts,
        metrics={"methods": methods_to_run},
    )
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage4_clustering/daura.py"
"""Vectorized Daura / GROMOS greedy RMSD clustering."""

from typing import List, Tuple
import numpy as np


def run_daura(
    rmsd_matrix: np.ndarray,
    rmsd_cutoff: float,
    k: int,
) -> Tuple[np.ndarray, List[int]]:
    """Run vectorized Daura greedy clustering; k <= 0 allows unlimited clusters."""
    n_samples = rmsd_matrix.shape[0]
    if n_samples == 0:
        raise ValueError("Cannot run Daura clustering on 0 frames.")

    print(f"\n[ Daura ] Running Vectorized Daura Clustering with RMSD Cutoff {rmsd_cutoff} Å...")
    unclustered = np.ones(n_samples, dtype=bool)
    centroids: List[int] = []
    max_clusters = k if k > 0 else n_samples

    neighborhoods = rmsd_matrix < rmsd_cutoff
    while unclustered.any():
        counts = neighborhoods.dot(unclustered).astype(int)
        counts[~unclustered] = -1
        best_c = int(np.argmax(counts))
        best_h = neighborhoods[best_c] & unclustered
        centroids.append(best_c)
        unclustered[best_h] = False
        if len(centroids) >= max_clusters:
            break

    labels = np.argmin(rmsd_matrix[:, centroids], axis=1)
    return labels, centroids
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage4_clustering/kmeans.py"
"""PCA + KMeans clustering with Silhouette auto-k and small-sample safety."""

from typing import Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_kmeans(X_pca: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """Run KMeans on PCA coordinates; auto-detect k if k <= 0 and clamp to n_samples."""
    n_samples = len(X_pca)
    if n_samples == 0:
        raise ValueError("Cannot run KMeans on 0 samples.")

    auto_det = False
    if k <= 0:
        print("\n[ PCA + KMeans ] Auto-detecting optimal 'k' using Silhouette Analysis...")
        auto_det = True
        if n_samples < 3:
            best_k = min(2, n_samples)
            print(f"  -> Too few samples ({n_samples}) for Silhouette; using k = {best_k}")
        else:
            best_score = -1.0
            best_k = 2
            max_test_k = min(15, n_samples - 1)
            for k_test in range(2, max_test_k + 1):
                labels_test = KMeans(n_clusters=k_test, random_state=42, n_init="auto").fit_predict(X_pca)
                if len(set(labels_test)) < 2:
                    continue
                score = silhouette_score(X_pca, labels_test)
                if score > best_score:
                    best_score = score
                    best_k = k_test
            print(f"  -> Optimal clusters detected: k = {best_k} (Silhouette Score: {best_score:.3f})")
    else:
        best_k = min(k, n_samples)
        if best_k != k:
            print(f"\n[ PCA + KMeans ] Clamped k={k} to n_samples={n_samples}.")
        else:
            print(f"\n[ PCA + KMeans ] Running with explicit k={best_k}...")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_pca)
    return labels, kmeans.cluster_centers_, best_k, auto_det
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage4_clustering/pairwise.py"
"""Pairwise RMSD calculation and centroid PDB extraction."""

from typing import Any, Callable, List, Tuple
import numpy as np
from sklearn.metrics import pairwise_distances


def compute_pairwise_rmsd(X: np.ndarray, n_atoms: int) -> np.ndarray:
    """Compute pairwise RMSD matrix (N x N) from flattened aligned coordinates."""
    if n_atoms <= 0:
        raise ValueError("n_atoms must be > 0 to compute pairwise RMSD.")
    dists = pairwise_distances(X, metric="euclidean", n_jobs=-1)
    dists /= dists.dtype.type(np.sqrt(n_atoms))
    return dists


def make_centroid_extractor(
    merged_ref_path: str,
    merged_xtc_path: str,
    valid_frames: np.ndarray,
    frame_universe_map: List[Tuple[Any, int]],
) -> Callable[[str, int, int, str], str]:
    """Create a centroid PDB writer backed by the merged trajectory or source universes."""
    import MDAnalysis as mda

    try:
        u_centroid_src = mda.Universe(merged_ref_path, merged_xtc_path)

        def extract_c(pfx: str, c_idx: int, cid: int, out_dir: str) -> str:
            u_centroid_src.trajectory[c_idx]
            out_path = f"{out_dir}/{pfx}_centroid_{cid}.pdb"
            u_centroid_src.atoms.write(out_path)
            return out_path

        extract_c.close = u_centroid_src.trajectory.close  # type: ignore[attr-defined]
        return extract_c
    except Exception:

        def extract_c(pfx: str, c_idx: int, cid: int, out_dir: str) -> str:
            frame_idx = valid_frames[c_idx]
            u_curr, local_idx = frame_universe_map[frame_idx]
            u_curr.trajectory[local_idx]
            out_path = f"{out_dir}/{pfx}_centroid_{cid}.pdb"
            u_curr.atoms.write(out_path)
            return out_path

        extract_c.close = lambda: None  # type: ignore[attr-defined]
        return extract_c
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage5_energy/__init__.py"
"""Stage 5: OpenMM thermodynamic energy evaluation and CSV fusion."""

from metflow.stage5_energy.openmm_engine import run_stage5

__all__ = ["run_stage5"]
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage5_energy/csv_fusion.py"
"""Smart CSV fusion for preserving and merging OpenMM energy checkpoints."""

import math
import os
from typing import Any, Dict
import numpy as np
import pandas as pd


def is_valid_energy(val: Any) -> bool:
    """Return True if val is a finite, non-NaN numeric energy value."""
    if val is None or isinstance(val, bool):
        return False
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def fuse_energy_csv(
    df_data: Dict[str, Any],
    valid_frames: np.ndarray,
    energy_csv: str,
) -> int:
    """Fuse existing energy_checkpoint.csv into df_data if not already populated."""
    if "OpenMM_Energy_kcal_mol" in df_data or not os.path.exists(energy_csv):
        return 0
    try:
        df_resume = pd.read_csv(energy_csv)
        if "Frame" not in df_resume.columns:
            return 0
        ene_col = "Energy_kcal_mol" if "Energy_kcal_mol" in df_resume.columns else "Energy"
        if ene_col not in df_resume.columns:
            return 0

        energy_dict: Dict[int, float] = {}
        for raw_frame, raw_val in zip(df_resume["Frame"], df_resume[ene_col]):
            try:
                frame_id = int(raw_frame)
            except (TypeError, ValueError):
                continue
            if is_valid_energy(raw_val):
                energy_dict[frame_id] = float(raw_val)
            elif frame_id not in energy_dict:
                energy_dict[frame_id] = np.nan

        fused_values = []
        matched_valid = 0
        for f in valid_frames:
            val = energy_dict.get(int(f), np.nan)
            if not math.isnan(val):
                fused_values.append(val)
                matched_valid += 1
            else:
                fused_values.append(np.nan)

        df_data["OpenMM_Energy_kcal_mol"] = fused_values
        print(f"✅ Auto-fused {matched_valid} existing energy evaluations into clustering results!")
        return matched_valid
    except Exception as exc:
        print(f"⚠️ Could not fuse energy checkpoint: {exc}")
        return 0
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage5_energy/openmm_engine.py"
"""OpenMM Amber14/OBC2 frozen-heavy-atom energy evaluation with checkpointing."""

import csv
import math
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metflow.models import StageResult, TrajectoryContext
from metflow.stage1_ingestion.indexing import (
    PROTEIN_RESNAMES,
    heavy_atom_exclusion_expr,
    precompute_atom_indexer,
)
from metflow.stage5_energy.csv_fusion import fuse_energy_csv, is_valid_energy
from metflow.stage6_visualization.reports import log_methodology


def _write_checkpoint_atomic(
    energy_dict: Dict[int, float],
    energy_csv: str,
    return_df: bool = True,
) -> Optional[pd.DataFrame]:
    """Atomically write energy checkpoint CSV via a temporary file and os.replace."""
    tmp_csv = f"{energy_csv}.tmp"
    with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Energy_kcal_mol"])
        for frame_id, val in energy_dict.items():
            if val is None or (isinstance(val, float) and math.isnan(val)):
                writer.writerow([frame_id, ""])
            else:
                writer.writerow([frame_id, val])
    os.replace(tmp_csv, energy_csv)
    if return_df:
        return pd.DataFrame(list(energy_dict.items()), columns=["Frame", "Energy_kcal_mol"])
    return None


def _save_partial_plot(df_partial: pd.DataFrame, out_dir: str) -> str:
    """Generate partial_energy_progress.png on KeyboardInterrupt."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(df_partial["Frame"], df_partial["Energy_kcal_mol"], alpha=0.6, s=12, color="#D55E00")
    plt.xlabel("Frame Index")
    plt.ylabel("Potential Energy (kcal/mol)")
    plt.title("Partial Energy Timeline (kcal/mol)")

    plt.subplot(1, 2, 2)
    valid_partial = df_partial["Energy_kcal_mol"].dropna()
    if len(valid_partial) > 0:
        plt.hist(valid_partial, bins=30, color="#0072B2", alpha=0.75)
    plt.xlabel("Potential Energy (kcal/mol)")
    plt.ylabel("Frequency (Count)")
    plt.title("Partial Energy Distribution (kcal/mol)")

    plt.tight_layout()
    partial_plot = os.path.join(out_dir, "partial_energy_progress.png")
    plt.savefig(partial_plot, dpi=300)
    plt.close()
    print(f"📊 Generated a progress visualization of your partial results: {partial_plot}")
    return partial_plot


def run_openmm_energy(ctx: TrajectoryContext) -> Tuple[Dict[int, float], bool, List[str], str, int]:
    """Evaluate OpenMM potential energies with 50-frame atomic checkpointing and clean interrupt handling."""
    import MDAnalysis as mda

    config = ctx.config
    out_dir = ctx.out_dir
    valid_frames = ctx.valid_frames
    artifacts: List[str] = []
    energy_csv = os.path.join(out_dir, "energy_checkpoint.csv")
    energy_dict: Dict[int, float] = {}
    u_openmm = None
    platform_name = "Unknown"
    relax_steps = config.min_steps if config.min_steps > 0 else 25

    if os.path.exists(energy_csv):
        try:
            df_resume = pd.read_csv(energy_csv)
            ene_col = "Energy_kcal_mol" if "Energy_kcal_mol" in df_resume.columns else "Energy"
            raw_dict = dict(zip(df_resume["Frame"], df_resume[ene_col]))
            energy_dict = {int(f): float(v) for f, v in raw_dict.items() if is_valid_energy(v)}
            print(f"  -> Found checkpoint! Resuming {len(energy_dict)} valid previously minimized frames...")
            log_methodology(
                ctx,
                f"Resumed OpenMM energy checkpoint ({len(energy_dict)} valid frames already evaluated) from {energy_csv}.",
            )
        except Exception:
            energy_dict = {}

    try:
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
        ref_excl = heavy_atom_exclusion_expr(u_ref_clean)
        u_heavy = u_ref_clean.select_atoms(f"resname {prot_res_str} and {ref_excl}")
        if len(u_heavy) == 0:
            u_heavy = u_ref_clean.select_atoms(
                f"{ref_excl} and not (resname WAT or resname HOH or resname SOL or resname NA or resname CL)"
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

        tmp_dir = os.path.join(out_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_pdb = os.path.join(tmp_dir, "tmp_clean_heavy.pdb")
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
        # Use constraints=None because heavy atoms are frozen (mass=0.0);
        # OpenMM forbids HBonds constraints involving massless particles.
        system = forcefield.createSystem(ref_modeller.topology, nonbondedMethod=NoCutoff, constraints=None)

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
        platform_name = platform.getName()
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
        merged_heavy_indexer = None
        if os.path.exists(merged_ref_path) and os.path.exists(merged_xtc_path):
            try:
                u_openmm = mda.Universe(merged_ref_path, merged_xtc_path)
                use_merged_traj = len(u_openmm.trajectory) == len(valid_frames)
            except Exception:
                use_merged_traj = False

        if use_merged_traj and u_openmm is not None:
            openmm_excl = heavy_atom_exclusion_expr(u_openmm)
            u_openmm_heavy = u_openmm.select_atoms(f"resname {prot_res_str} and {openmm_excl}")
            if len(u_openmm_heavy) == 0:
                u_openmm_heavy = u_openmm.select_atoms(
                    f"{openmm_excl} and not (resname WAT or resname HOH or resname SOL or resname NA or resname CL)"
                )
            merged_heavy_indexer = precompute_atom_indexer(u_openmm_heavy)
            if len(u_openmm_heavy) != len(u_heavy):
                curr_map = {(a.resid, a.name.strip().upper()): i for i, a in enumerate(u_openmm_heavy)}
                merged_map_idx = np.zeros(len(u_heavy), dtype=int)
                for m_idx, m_atom in enumerate(u_heavy.atoms):
                    key = (m_atom.resid, m_atom.name.strip().upper())
                    merged_map_idx[m_idx] = curr_map.get(key, min(m_idx, max(0, len(u_openmm_heavy) - 1)))
                merged_heavy_indexer = merged_heavy_indexer[merged_map_idx]

        universe_heavy_info = {}
        if not use_merged_traj:
            for u_obj in set(u for u, _ in ctx.frame_universe_map):
                u_excl = heavy_atom_exclusion_expr(u_obj)
                u_h = u_obj.select_atoms(f"resname {prot_res_str} and {u_excl}")
                if len(u_h) == 0:
                    u_h = u_obj.select_atoms(
                        f"{u_excl} and not (resname WAT or resname HOH or resname SOL or resname NA or resname CL)"
                    )
                h_indexer = precompute_atom_indexer(u_h)
                if len(u_h) != len(u_heavy):
                    curr_map = {(a.resid, a.name.strip().upper()): i for i, a in enumerate(u_h)}
                    map_idx = np.zeros(len(u_heavy), dtype=int)
                    for m_idx, m_atom in enumerate(u_heavy.atoms):
                        key = (m_atom.resid, m_atom.name.strip().upper())
                        map_idx[m_idx] = curr_map.get(key, min(m_idx, max(0, len(u_h) - 1)))
                    h_indexer = h_indexer[map_idx]
                universe_heavy_info[id(u_obj)] = h_indexer

        if config.min_steps > 0:
            print(f"  -> Relaxing hydrogens and evaluating energy ({relax_steps} steps per frame, heavy atoms frozen)...")
        else:
            print(f"  -> Evaluating potential energy with hydrogen clash-relief ({relax_steps} steps per frame, heavy atoms frozen)...")

        log_methodology(
            ctx,
            f"OpenMM setup: Amber14/OBC2 implicit solvent, platform={platform_name}, "
            f"hydrogen relaxation={relax_steps} steps (heavy atoms frozen).",
        )

        full_heavy_pos = np.zeros((num_fixer_atoms, 3), dtype=np.float32)
        if n_fallback > 0:
            full_heavy_pos[fallback_mask] = ref_pos[fallback_mask]
        current_full_pos = ref_full_pos.copy()

        newly_evaluated = 0
        for local_idx, frame_idx in enumerate(tqdm(valid_frames, desc="OpenMM Energy")):
            frame_key = int(frame_idx)
            if frame_key in energy_dict and is_valid_energy(energy_dict[frame_key]):
                continue

            if use_merged_traj:
                ts = u_openmm.trajectory[local_idx]
                heavy_pos_src = ts.positions[merged_heavy_indexer] * np.float32(0.1)
            else:
                u_curr, raw_local_idx = ctx.frame_universe_map[frame_key]
                ts = u_curr.trajectory[raw_local_idx]
                h_indexer = universe_heavy_info[id(u_curr)]
                pos_angstroms = ts.positions[h_indexer]
                pos_centered = pos_angstroms - pos_angstroms.mean(axis=0)
                rms_dist = np.sqrt(np.mean(np.sum(pos_centered ** 2, axis=1)))
                scale = np.float32(0.1) if rms_dist >= 5.0 else np.float32(1.0)
                heavy_pos_src = pos_angstroms * scale

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
            newly_evaluated += 1
            if newly_evaluated % 50 == 0:
                _write_checkpoint_atomic(energy_dict, energy_csv, return_df=False)
    except KeyboardInterrupt:
        df_partial = _write_checkpoint_atomic(energy_dict, energy_csv, return_df=True)
        artifacts.append(energy_csv)
        print(f"\n⚠️  Energy calculation PAUSED by user! Saved checkpoint ({len(energy_dict)} frames) to {energy_csv}")
        log_methodology(
            ctx,
            f"OpenMM Energy calculation paused by user (KeyboardInterrupt). "
            f"Saved atomic partial checkpoint ({len(energy_dict)} frames) to {energy_csv}.",
        )
        if df_partial is not None and len(df_partial) > 1:
            artifacts.append(_save_partial_plot(df_partial, out_dir))
        print("Run the script again to resume exactly where you left off.")
        return energy_dict, True, artifacts, platform_name, relax_steps
    finally:
        if u_openmm is not None:
            try:
                u_openmm.trajectory.close()
            except Exception:
                pass

    _write_checkpoint_atomic(energy_dict, energy_csv, return_df=False)
    artifacts.append(energy_csv)
    return energy_dict, False, artifacts, platform_name, relax_steps


def run_stage5(ctx: TrajectoryContext) -> StageResult:
    """Execute Stage 5: optional OpenMM energy calculation and checkpoint CSV fusion."""
    config = ctx.config
    out_dir = ctx.out_dir
    valid_frames = ctx.valid_frames
    artifacts: List[str] = []
    energy_csv = os.path.join(out_dir, "energy_checkpoint.csv")

    if config.is_calc_energy:
        print("\n--- Calculating Thermodynamic Energy (OpenMM Physical Minimization) ---")
        log_methodology(ctx, "\n--- OpenMM Energy Calculation ---")
        try:
            energy_dict, interrupted, energy_artifacts, platform_name, relax_steps = run_openmm_energy(ctx)
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
            log_methodology(
                ctx,
                f"Evaluated Thermodynamic Potential Energy (OpenMM, Amber14/OBC2, platform={platform_name}, "
                f"{relax_steps}-step Minimization, {valid_cnt}/{len(valid_frames)} valid, unit: kcal/mol).",
            )
            print(f"✅ OpenMM Energy Evaluation complete ({valid_cnt}/{len(valid_frames)} valid frames).")
        except Exception as exc:
            print(f"❌ Error during OpenMM Energy calculation: {exc}")
            log_methodology(ctx, f"OpenMM Energy Calculation failed: {exc}")

    fused_count = fuse_energy_csv(ctx.df_data, valid_frames, energy_csv)
    if not config.is_calc_energy and fused_count > 0:
        log_methodology(
            ctx,
            f"Auto-fused {fused_count} existing OpenMM energy evaluations (kcal/mol) from {energy_csv}.",
        )
    return StageResult(
        stage_name="stage5_energy",
        success=True,
        exit_code=0,
        artifacts=artifacts,
        metrics={"interrupted": False, "fused_count": fused_count},
    )
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage6_visualization/__init__.py"
"""Stage 6: Static plots, interactive Art Nouveau dashboard, and reports."""

from metflow.stage6_visualization.reports import (
    init_methodology,
    log_methodology,
    run_stage6,
    write_cluster_summary,
    write_methodology,
)

__all__ = [
    "init_methodology",
    "log_methodology",
    "run_stage6",
    "write_cluster_summary",
    "write_methodology",
]
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage6_visualization/dashboard.py"
"""Art Nouveau HTML dashboard generation and --only_dashboard fast path."""

import os
from typing import Optional
import pandas as pd

DASHBOARD_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
        .close-btn { background: var(--gold); color: var(--dark-green); border: none; padding: 10px 20px; cursor: pointer; font-weight: bold; margin-top: 15px; border-radius: 5px;}
        .overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.55); z-index: 999; backdrop-filter: blur(2px); }

        /* Interactive Image Customization & Export Modal */
        #export-modal { display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: #ffffff; color: var(--text-color); padding: 25px 30px; border: 3px solid var(--gold); border-radius: 16px; z-index: 1001; box-shadow: 0 15px 45px rgba(0,0,0,0.45); width: 92%; max-width: 820px; max-height: 90vh; overflow-y: auto; font-family: 'Lato', sans-serif; }
        #export-modal h2 { font-family: 'Playfair Display', serif; color: var(--dark-green); margin: 0 0 15px 0; border-bottom: 2px solid var(--gold); padding-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
        .modal-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px 20px; }
        .modal-section { background: var(--bg-color); border: 1px solid rgba(212, 175, 55, 0.45); border-radius: 10px; padding: 12px 16px; }
        .modal-section h4 { margin: 0 0 12px 0; color: var(--dark-green); font-family: 'Playfair Display', serif; font-size: 1.05em; border-bottom: 1px solid rgba(212, 175, 55, 0.35); padding-bottom: 4px; display: flex; align-items: center; gap: 6px; }
        .field-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 9px; font-size: 0.88em; }
        .field-row label { font-weight: 600; color: #1b3a30; }
        .field-row input[type="text"], .field-row input[type="number"], .field-row select { width: 145px; padding: 5px 8px; border: 1px solid var(--gold); border-radius: 4px; background: #fff; font-size: 0.9em; font-family: inherit; }
        .field-row input[type="color"] { width: 45px; height: 28px; border: 1px solid var(--gold); border-radius: 4px; cursor: pointer; padding: 0; background: none; }
        .slider-row { margin-bottom: 10px; font-size: 0.88em; }
        .slider-header { display: flex; justify-content: space-between; margin-bottom: 3px; font-weight: 600; color: #1b3a30; }
        .slider-row input[type="range"] { width: 100%; cursor: pointer; accent-color: var(--dark-green); }
        .modal-actions { display: flex; justify-content: flex-end; gap: 12px; margin-top: 18px; padding-top: 14px; border-top: 2px solid var(--gold); }
        .btn-apply { background: #1b3a30; color: #d4af37; border: 1px solid #d4af37; padding: 9px 18px; border-radius: 6px; font-weight: bold; cursor: pointer; transition: 0.2s; font-size: 0.92em; }
        .btn-apply:hover { background: #234c3f; }
        .btn-download { background: var(--gold); color: var(--dark-green); border: none; padding: 9px 24px; border-radius: 6px; font-weight: bold; cursor: pointer; transition: 0.2s; font-size: 0.95em; box-shadow: 0 4px 10px rgba(0,0,0,0.15); }
        .btn-download:hover { filter: brightness(1.1); }
        .btn-cancel { background: #e2e8f0; color: #4a5568; border: none; padding: 9px 16px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.92em; }
        .btn-cancel:hover { background: #cbd5e1; }
        .modal-close-icon { background: transparent; border: 1px solid var(--gold); color: var(--dark-green); border-radius: 4px; cursor: pointer; font-size: 0.9em; padding: 4px 8px; }
        .modal-close-icon:hover { background: var(--bg-color); }
        #status-banner { display: none; padding: 10px 14px; border-radius: 6px; font-size: 0.9em; font-weight: 600; border: 1px solid var(--gold); background: #edf7ed; color: #1b3a30; }
        #status-banner.warning { background: #fef3c7; color: #78350f; border-color: #d97706; }
        .file-drop:focus, button:focus, select:focus, input:focus { outline: 2px solid var(--dark-green); outline-offset: 2px; }
        @media (max-width: 900px) {
            body { height: auto; }
            .container { flex-direction: column; overflow: visible; }
            .sidebar { width: auto; }
        }
    </style>
</head>
<body>
    <header>
        <h1>MET Flow</h1>
        <div style="font-style: italic;">Thermodynamic & Conformational Analytics</div>
    </header>
    <div class="container">
        <aside class="sidebar" aria-label="Controls and Data Ingestion">
            <h2>Data Ingestion</h2>
            <div class="file-drop" id="drop-zone" tabindex="0" role="button" aria-label="Drop CSV files here or press Enter to browse">
                Drop <b>global_cluster_metrics.csv</b>, <b>energy_checkpoint.csv</b>, or <b>pca_loadings.csv</b> here
                <input type="file" id="file-input" style="display: none;" accept=".csv" multiple aria-label="Upload CSV files">
            </div>
            <div id="status-banner" role="status" aria-live="polite"></div>
            <button type="button" class="export-btn" onclick="exportCSV()">💾 Export Merged CSV</button>
            <button type="button" class="export-btn" onclick="openExportModal()" style="margin-top: 10px; background: #1b3a30; color: #d4af37; border: 1px solid #d4af37;">🎨 Customize & Export Image</button>
            <h2>Plot Controls</h2>
            <label for="sel-filter-dataset">Filter Dataset:</label>
            <select id="sel-filter-dataset" aria-label="Filter Dataset" onchange="updatePlots()">
                <option value="all" selected>All Datasets (Combined)</option>
            </select>
            <label for="sel-x">X-Axis:</label>
            <select id="sel-x" aria-label="X-Axis"></select>
            <label for="sel-y">Y-Axis:</label>
            <select id="sel-y" aria-label="Y-Axis"></select>
            <label for="sel-z">Z-Axis (3D):</label>
            <select id="sel-z" aria-label="Z-Axis (3D)"><option value="none">None (2D Plot)</option></select>
            <label for="sel-color">Color By:</label>
            <select id="sel-color" aria-label="Color By"></select>
            <div id="color-mode-container" style="margin-top:8px;">
                <label for="sel-color-mode">Coloring Mode:</label>
                <select id="sel-color-mode" aria-label="Coloring Mode" onchange="updatePlots()">
                    <option value="binned" selected>Interval Ranges (Box-Plot with Outliers)</option>
                    <option value="continuous">Continuous Gradient</option>
                </select>
            </div>
            <div id="colorscale-container" style="display:none; margin-top:8px;">
                <label for="sel-colorscale">Colorscale:</label>
                <select id="sel-colorscale" aria-label="Colorscale" onchange="updatePlots()">
                    <option value="Viridis">Viridis</option>
                    <option value="Plasma">Plasma</option>
                    <option value="Inferno">Inferno</option>
                    <option value="Jet">Jet</option>
                    <option value="RdBu">Red-Blue (RdBu)</option>
                </select>
            </div>
            <div style="margin-top: 15px;">
                <label for="filter-energy" style="display:flex; align-items:center; gap: 8px;">
                    <input type="checkbox" id="filter-energy" aria-label="Only show frames with Energy data" onchange="updatePlots()"> Only show frames with Energy data
                </label>
            </div>
        </aside>
        <main class="main-content">
            <div class="plot-container" id="main-plot" style="height: 85vh; min-height: 700px; position: relative;">
                <div id="main-plot-placeholder" style="position: absolute; top: 40%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: #595959; pointer-events: none; z-index: 1;">
                    <div style="font-size: 1.3em; font-family: 'Playfair Display', serif; color: #1b3a30; margin-bottom: 8px;">Awaiting Trajectory Data</div>
                    <div>Drag & drop <b>global_cluster_metrics.csv</b> or wait for sample metrics to render.</div>
                </div>
            </div>
            <div class="plot-container" id="loadings-plot" style="display: none; height: 400px;"></div>
            <div style="display: flex; gap: 20px;">
                <div class="plot-container" id="timeseries-plot" style="flex: 1; height: 400px; overflow:hidden;">
                    <div style="text-align:center; padding: 100px; color: #595959;">No Energy Data Found.<br><br>Drag & drop <b>energy_checkpoint.csv</b> to dynamically merge and visualize energy (kcal/mol)!</div>
                </div>
                <div class="plot-container" id="hist-plot" style="flex: 1; height: 400px;"></div>
            </div>
        </main>
    </div>
    <div class="overlay" id="overlay" onclick="closeModal()"></div>
    <div id="pymol-modal" role="dialog" aria-modal="true" aria-labelledby="pymol-modal-title">
        <h3 id="pymol-modal-title">Frame Selected!</h3>
        <p>Frame ID: <span id="modal-frame"></span> | Cluster/Color: <span id="modal-cluster"></span> | Energy (kcal/mol): <span id="modal-energy"></span></p>
        <p>To extract this frame in PyMOL on your server, copy and paste this command:</p>
        <div id="pymol-code"></div>
        <button type="button" class="close-btn" onclick="closeModal()">Close</button>
    </div>

    <!-- Export & Styling Customization Modal -->
    <div id="export-modal" role="dialog" aria-modal="true" aria-labelledby="export-modal-title">
        <h2 id="export-modal-title">
            <span>🎨 Customize Image & Export</span>
            <button type="button" class="modal-close-icon" aria-label="Close export modal" onclick="closeModal()">✖</button>
        </h2>
        <div class="modal-grid">
            <!-- Section 1: Typography & Labels -->
            <div class="modal-section">
                <h4>🔤 Typography & Text</h4>
                <div class="field-row">
                    <label>Plot Title:</label>
                    <input type="text" id="exp-title" value="Structural Landscape" oninput="liveUpdateTitle()">
                </div>
                <div class="field-row">
                    <label>Font Family:</label>
                    <select id="exp-font-family">
                        <option value="Playfair Display, Georgia, serif" selected>Playfair Display (Serif)</option>
                        <option value="Lato, sans-serif">Lato (Clean Sans)</option>
                        <option value="Arial, Helvetica, sans-serif">Arial / Helvetica</option>
                        <option value="Times New Roman, Times, serif">Times New Roman (Academic)</option>
                        <option value="Courier New, monospace">Courier New (Monospace)</option>
                    </select>
                </div>
                <div class="slider-row">
                    <div class="slider-header"><span>Title Size:</span><span id="lbl-title-size">24 px</span></div>
                    <input type="range" id="exp-title-size" min="14" max="44" value="24" oninput="document.getElementById('lbl-title-size').innerText = this.value + ' px'">
                </div>
                <div class="slider-row">
                    <div class="slider-header"><span>Axis Label Size:</span><span id="lbl-axis-size">14 px</span></div>
                    <input type="range" id="exp-axis-size" min="10" max="30" value="14" oninput="document.getElementById('lbl-axis-size').innerText = this.value + ' px'">
                </div>
                <div class="slider-row">
                    <div class="slider-header"><span>Tick Numbers Size:</span><span id="lbl-tick-size">11 px</span></div>
                    <input type="range" id="exp-tick-size" min="8" max="22" value="11" oninput="document.getElementById('lbl-tick-size').innerText = this.value + ' px'">
                </div>
                <div class="field-row">
                    <label>Text & Label Color:</label>
                    <input type="color" id="exp-font-color" value="#1b3a30">
                </div>
            </div>

            <!-- Section 2: Data Points / Dots -->
            <div class="modal-section">
                <h4>🔴 Data Points & Dots</h4>
                <div class="slider-row">
                    <div class="slider-header"><span>Marker / Dot Size:</span><span id="lbl-marker-size">5 px</span></div>
                    <input type="range" id="exp-marker-size" min="1" max="22" value="5" oninput="document.getElementById('lbl-marker-size').innerText = this.value + ' px'">
                </div>
                <div class="slider-row">
                    <div class="slider-header"><span>Marker Opacity:</span><span id="lbl-marker-opacity">0.85</span></div>
                    <input type="range" id="exp-marker-opacity" min="0.1" max="1.0" step="0.05" value="0.85" oninput="document.getElementById('lbl-marker-opacity').innerText = this.value">
                </div>
                <div class="slider-row">
                    <div class="slider-header"><span>Dot Border Width:</span><span id="lbl-border-width">0 px</span></div>
                    <input type="range" id="exp-border-width" min="0" max="3" step="0.5" value="0" oninput="document.getElementById('lbl-border-width').innerText = this.value + ' px'">
                </div>
                <div class="field-row">
                    <label>Dot Border Color:</label>
                    <input type="color" id="exp-border-color" value="#ffffff">
                </div>
                <div class="field-row">
                    <label>Legend Position:</label>
                    <select id="exp-legend-pos">
                        <option value="right" selected>Right (Vertical)</option>
                        <option value="top">Top (Horizontal)</option>
                        <option value="bottom">Bottom (Horizontal)</option>
                        <option value="none">Hidden</option>
                    </select>
                </div>
            </div>

            <!-- Section 3: Colors & Background -->
            <div class="modal-section">
                <h4>🎨 Colors & Background</h4>
                <div class="field-row">
                    <label>Canvas Background:</label>
                    <select id="exp-bg-mode" onchange="toggleCustomBgInput(this.value)">
                        <option value="white" selected>Pure White (#ffffff)</option>
                        <option value="transparent">Transparent</option>
                        <option value="cream">Theme Cream (#f4f1ea)</option>
                        <option value="dark">Dark Forest (#0f251e)</option>
                        <option value="custom">Custom Color</option>
                    </select>
                </div>
                <div class="field-row" id="row-custom-bg" style="display: none;">
                    <label>Custom BG Color:</label>
                    <input type="color" id="exp-custom-bg" value="#ffffff">
                </div>
                <div class="field-row">
                    <label>Show Gridlines:</label>
                    <input type="checkbox" id="exp-show-grid" checked>
                </div>
                <div class="field-row">
                    <label>Gridline Color:</label>
                    <input type="color" id="exp-grid-color" value="#e2e8f0">
                </div>
                <div class="field-row">
                    <label>Show Zero Line:</label>
                    <input type="checkbox" id="exp-show-zeroline" checked>
                </div>
            </div>

            <!-- Section 4: Resolution & File Format -->
            <div class="modal-section">
                <h4>📐 Dimensions & Resolution</h4>
                <div class="field-row">
                    <label>Dimension Preset:</label>
                    <select id="exp-preset" onchange="applyPresetDimensions(this.value)">
                        <option value="1200x800">Standard Web (1200 x 800)</option>
                        <option value="1920x1080" selected>Full HD 1080p (1920 x 1080)</option>
                        <option value="2560x1440">2K QHD (2560 x 1440)</option>
                        <option value="3840x2160">4K Ultra HD (3840 x 2160)</option>
                        <option value="1050x800">Journal Single-Col (1050 x 800)</option>
                        <option value="2100x1400">Journal Double-Col (2100 x 1400)</option>
                        <option value="custom">Custom Width / Height</option>
                    </select>
                </div>
                <div class="field-row">
                    <label>Width (pixels):</label>
                    <input type="number" id="exp-width" value="1920" min="400" max="7680">
                </div>
                <div class="field-row">
                    <label>Height (pixels):</label>
                    <input type="number" id="exp-height" value="1080" min="300" max="4320">
                </div>
                <div class="field-row">
                    <label>Resolution / DPI Scale:</label>
                    <select id="exp-scale">
                        <option value="1">1x (Standard ~96 DPI)</option>
                        <option value="2" selected>2x (HiDPI / Retina ~192 DPI)</option>
                        <option value="3">3x (Publication Print ~300 DPI)</option>
                        <option value="4">4x (Ultra Publication ~400 DPI)</option>
                    </select>
                </div>
                <div class="field-row">
                    <label>Export Format:</label>
                    <select id="exp-format">
                        <option value="png" selected>PNG (.png - Lossless)</option>
                        <option value="svg">SVG (.svg - Vector Graphics)</option>
                        <option value="jpeg">JPEG (.jpg - Web)</option>
                        <option value="webp">WebP (.webp)</option>
                    </select>
                </div>
            </div>
        </div>
        <div class="modal-actions">
            <button class="btn-cancel" onclick="closeModal()">Cancel</button>
            <button class="btn-apply" onclick="applySettingsToLiveView()">👁️ Apply to Dashboard View</button>
            <button class="btn-download" onclick="downloadCustomImage()">💾 Download High-Res Image</button>
        </div>
    </div>
    <script>
        const EMBEDDED_METRICS_JSON = /*EMBEDDED_DATA_PLACEHOLDER*/[];
        let globalData = [];
        let globalLoadings = [];
        let cols = [];
        let pendingFiles = [];
        let mergeTimeout = null;

        function showStatus(message, isWarning = false) {
            const banner = document.getElementById('status-banner');
            if (!banner) return;
            if (!message) {
                banner.style.display = 'none';
                banner.innerText = '';
                banner.classList.remove('warning');
                return;
            }
            banner.innerText = message;
            banner.classList.toggle('warning', Boolean(isWarning));
            banner.style.display = 'block';
        }

        function formatAxisLabel(col) {
            if (!col) return '';
            const lower = col.toLowerCase();
            if (lower.includes('rmsd') && !lower.includes('å') && !lower.includes('angstrom')) {
                return `${col} (Å)`;
            }
            if (lower.includes('energy') && !lower.includes('kcal')) {
                return `${col} (kcal/mol)`;
            }
            return col;
        }

        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');

        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                fileInput.click();
            }
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
        });
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
            if (Array.isArray(EMBEDDED_METRICS_JSON) && EMBEDDED_METRICS_JSON.length > 0) {
                const ph = document.getElementById('main-plot-placeholder');
                if (ph) { ph.style.display = 'none'; ph.remove(); }
                globalData = EMBEDDED_METRICS_JSON;
                cols = Object.keys(globalData[0]);
                populateDropdowns();
                updatePlots();
                return;
            }
            fetch('global_cluster_metrics.csv')
                .then(r => { if (r.ok) return r.text(); throw new Error('Not found'); })
                .then(text => {
                    Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true, complete: function(res) {
                        if (res.meta && res.meta.fields) {
                            const ph = document.getElementById('main-plot-placeholder');
                            if (ph) { ph.style.display = 'none'; ph.remove(); }
                            globalData = res.data;
                            cols = res.meta.fields;
                            populateDropdowns();
                            updatePlots();
                        }
                    }});
                }).catch(() => {});
        });

        function processFile(file) {
            const ph = document.getElementById('main-plot-placeholder');
            if (ph) { ph.style.display = 'none'; ph.remove(); }
            Papa.parse(file, { header: true, dynamicTyping: true, skipEmptyLines: true,
                complete: function(results) {
                    if (results.meta.fields.includes('Residue') && results.meta.fields.includes('PC1_Contribution')) {
                        globalLoadings = results.data;
                        showStatus("✅ Loaded PCA Loadings Data!");
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
                        showStatus("✅ Successfully merged new dataset into existing plots!");
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

        function ensureLocalFrameColumns() {
            if (!globalData || globalData.length === 0) return;
            const hasLocal = cols.includes('Local_Frame_Index');
            const hasProg = cols.includes('Dataset_Progress_Pct');
            if (hasLocal && hasProg) return;
            const totals = {};
            globalData.forEach(d => {
                const ds = d['Dataset'] || 'Dataset';
                totals[ds] = (totals[ds] || 0) + 1;
            });
            const counts = {};
            globalData.forEach(d => {
                const ds = d['Dataset'] || 'Dataset';
                const k = counts[ds] || 0;
                const tot = totals[ds] || 1;
                if (!hasLocal) d['Local_Frame_Index'] = k;
                if (!hasProg) d['Dataset_Progress_Pct'] = tot > 1 ? Number((100.0 * k / (tot - 1)).toFixed(2)) : 0;
                counts[ds] = k + 1;
            });
            if (!hasLocal) cols.push('Local_Frame_Index');
            if (!hasProg) cols.push('Dataset_Progress_Pct');
        }

        function populateDropdowns() {
            ensureLocalFrameColumns();
            const sels = ['sel-x', 'sel-y', 'sel-z', 'sel-color'];
            const prevVals = {};
            sels.forEach(id => {
                prevVals[id] = document.getElementById(id).value;
            });

            sels.forEach(id => {
                const el = document.getElementById(id);
                if(id !== 'sel-z') el.innerHTML = '';
                else el.innerHTML = '<option value="none">None (2D Plot)</option>';
                cols.forEach(c => {
                    const opt = document.createElement('option');
                    opt.value = c;
                    opt.textContent = c;
                    el.appendChild(opt);
                });
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

        let customSettings = {
            title: 'Structural Landscape',
            fontFamily: 'Playfair Display, Georgia, serif',
            fontColor: '#1b3a30',
            titleSize: 24,
            axisLabelSize: 14,
            tickSize: 11,
            markerSize: 5,
            markerOpacity: 0.85,
            borderWidth: 0,
            borderColor: '#ffffff',
            bgMode: 'white',
            customBg: '#ffffff',
            showGrid: true,
            showZeroLine: true,
            gridColor: '#e2e8f0',
            legendPos: 'right',
            width: 1920,
            height: 1080,
            scale: 2,
            format: 'png'
        };

        function parseVal(v) {
            if (v === null || v === undefined) return null;
            if (typeof v === 'number') return (isNaN(v) || !isFinite(v)) ? null : v;
            let s = String(v).trim();
            if (s === '' || s.toLowerCase() === 'null' || s.toLowerCase() === 'nan' || s.toLowerCase() === 'n/a' || s.toLowerCase() === 'undefined') return null;
            let n = parseFloat(s);
            return (isNaN(n) || !isFinite(n)) ? null : n;
        }

        function updatePlots() {
            if(globalLoadings.length > 0) {
                document.getElementById('loadings-plot').style.display = 'block';
                Plotly.react('loadings-plot', [
                    { x: globalLoadings.map(d=>d.Residue), y: globalLoadings.map(d=>d.PC1_Contribution), name: 'PC1', type: 'scatter', mode: 'lines+markers', line: {color: '#1b3a30'} },
                    { x: globalLoadings.map(d=>d.Residue), y: globalLoadings.map(d=>d.PC2_Contribution), name: 'PC2', type: 'scatter', mode: 'lines+markers', line: {color: '#d4af37'} }
                ], { title: {text: 'PCA Loadings (Residue Contribution)', font: {family: customSettings.fontFamily, color: customSettings.fontColor}}, xaxis: {title: 'Residue Number'}, yaxis: {title: 'Loading Magnitude (RMS, Å)'}, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' }, {responsive: true});
            }

            if(!globalData.length) return;
            const ph = document.getElementById('main-plot-placeholder');
            if (ph) { ph.style.display = 'none'; ph.remove(); }

            let plotData = globalData;
            const selDataset = document.getElementById('sel-filter-dataset') ? document.getElementById('sel-filter-dataset').value : 'all';
            if (selDataset && selDataset !== 'all' && cols.includes('Dataset')) {
                plotData = plotData.filter(d => d['Dataset'] === selDataset);
            }
            if(document.getElementById('filter-energy') && document.getElementById('filter-energy').checked && cols.includes('OpenMM_Energy_kcal_mol')) {
                plotData = plotData.filter(d => parseVal(d['OpenMM_Energy_kcal_mol']) !== null);
            }
            if(!plotData.length) {
                showStatus("⚠️ No frames match the current filter criteria (0 matches).", true);
                Plotly.react('main-plot', [], { title: { text: 'No matching frames for current filter', font: { family: customSettings.fontFamily, color: customSettings.fontColor } } }, { responsive: true });
                return;
            }
            showStatus('');

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
                        marker: {
                            size: customSettings.markerSize,
                            color: c,
                            opacity: customSettings.markerOpacity,
                            line: { width: customSettings.borderWidth, color: customSettings.borderColor }
                        },
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
                                    size: isHighOutlier ? Math.max(3, customSettings.markerSize - 1) : (isNaNGroup ? Math.max(2.5, customSettings.markerSize - 1.5) : customSettings.markerSize),
                                    color: b.color,
                                    opacity: isNaNGroup ? 0.35 : customSettings.markerOpacity,
                                    line: { width: customSettings.borderWidth, color: customSettings.borderColor }
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
                const isIndexOrPct = ['frame', 'progress', 'pct'].some(k => cColor.toLowerCase().includes(k));
                if (sorted.length > 20 && !isIndexOrPct) {
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
                            size: customSettings.markerSize,
                            color: validIndices.map(i => numericVals[i]),
                            colorscale: cScale,
                            cmin: cmin,
                            cmax: cmax,
                            opacity: customSettings.markerOpacity,
                            line: { width: customSettings.borderWidth, color: customSettings.borderColor },
                            showscale: true,
                            colorbar: {
                                title: { text: cColor, font: { family: customSettings.fontFamily, size: customSettings.axisLabelSize, color: customSettings.fontColor } },
                                tickfont: { family: customSettings.fontFamily, size: customSettings.tickSize, color: customSettings.fontColor }
                            }
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
                        marker: {
                            size: Math.max(2.5, customSettings.markerSize - 1.5),
                            color: '#9ca3af',
                            opacity: 0.4,
                            line: { width: customSettings.borderWidth, color: customSettings.borderColor }
                        },
                        text: nanIndices.map(i => hovertext[i]),
                        hoverinfo: 'text+x+y+z',
                        customdata: nanIndices.map(i => plotData[i])
                    };
                    if (cz !== 'none') { nanTrace.type = 'scatter3d'; nanTrace.z = nanIndices.map(i => plotData[i][cz]); }
                    else { nanTrace.type = 'scatter'; }
                    traces.push(nanTrace);
                }
            }

            let liveBg = 'rgba(0,0,0,0)';
            if (customSettings.bgMode === 'white') liveBg = '#ffffff';
            else if (customSettings.bgMode === 'cream') liveBg = '#f4f1ea';
            else if (customSettings.bgMode === 'dark') liveBg = '#0f251e';
            else if (customSettings.bgMode === 'custom') liveBg = customSettings.customBg;

            let legendConfig = {
                title: { text: cColor, font: { family: customSettings.fontFamily, size: customSettings.axisLabelSize, color: customSettings.fontColor } },
                orientation: 'v',
                x: 1.02,
                y: 1,
                font: { family: customSettings.fontFamily, size: customSettings.tickSize, color: customSettings.fontColor }
            };
            if (customSettings.legendPos === 'none') {
                legendConfig = false;
            } else if (customSettings.legendPos === 'top') {
                legendConfig = { title: { text: cColor }, orientation: 'h', y: 1.12, x: 0, font: { family: customSettings.fontFamily, size: customSettings.tickSize, color: customSettings.fontColor } };
            } else if (customSettings.legendPos === 'bottom') {
                legendConfig = { title: { text: cColor }, orientation: 'h', y: -0.22, x: 0, font: { family: customSettings.fontFamily, size: customSettings.tickSize, color: customSettings.fontColor } };
            }

            const hasColorbar = traces.some(t => t.marker && t.marker.showscale);
            const showLegendEffective = (customSettings.legendPos !== 'none') && !(hasColorbar && traces.length === 1);

            let layout = {
                title: { text: customSettings.title, font: {family: customSettings.fontFamily, size: customSettings.titleSize, color: customSettings.fontColor} },
                paper_bgcolor: liveBg,
                plot_bgcolor: liveBg,
                margin: {l: 60, r: 60, b: 65, t: 80},
                font: {family: customSettings.fontFamily, size: customSettings.axisLabelSize, color: customSettings.fontColor},
                showlegend: showLegendEffective,
                legend: showLegendEffective ? legendConfig : undefined,
                xaxis: {
                    title: { text: formatAxisLabel(cx), font: { family: customSettings.fontFamily, size: customSettings.axisLabelSize, color: customSettings.fontColor } },
                    tickfont: { family: customSettings.fontFamily, size: customSettings.tickSize, color: customSettings.fontColor },
                    showgrid: customSettings.showGrid,
                    gridcolor: customSettings.gridColor,
                    zeroline: customSettings.showZeroLine
                },
                yaxis: {
                    title: { text: formatAxisLabel(cy), font: { family: customSettings.fontFamily, size: customSettings.axisLabelSize, color: customSettings.fontColor } },
                    tickfont: { family: customSettings.fontFamily, size: customSettings.tickSize, color: customSettings.fontColor },
                    showgrid: customSettings.showGrid,
                    gridcolor: customSettings.gridColor,
                    zeroline: customSettings.showZeroLine
                }
            };
            if (cz !== 'none') {
                layout.scene = {
                    xaxis: { title: formatAxisLabel(cx), showgrid: customSettings.showGrid, gridcolor: customSettings.gridColor },
                    yaxis: { title: formatAxisLabel(cy), showgrid: customSettings.showGrid, gridcolor: customSettings.gridColor },
                    zaxis: { title: formatAxisLabel(cz), showgrid: customSettings.showGrid, gridcolor: customSettings.gridColor }
                };
            }

            const plotConfig = {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToAdd: [{
                    name: '🎨 Customize & Export High-Res Image',
                    icon: Plotly.Icons.camera,
                    click: function() {
                        openExportModal();
                    }
                }],
                modeBarButtonsToRemove: ['toImage']
            };

            Plotly.react('main-plot', traces, layout, plotConfig);

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
                const e_vals = plotData.map(d => Number(d['OpenMM_Energy_kcal_mol'])).filter(v => v !== null && v !== undefined && !isNaN(v));
                if (e_vals.length > 0) {
                    document.getElementById('timeseries-plot').innerHTML = '';
                    document.getElementById('hist-plot').innerHTML = '';
                    const sorted_e = [...e_vals].sort((a,b) => a - b);
                    const q1 = sorted_e[Math.floor(sorted_e.length * 0.25)];
                    const q3 = sorted_e[Math.floor(sorted_e.length * 0.75)];
                    const iqr = q3 - q1;
                    const e_filt = (iqr > 0) ? e_vals.filter(v => v >= q1 - 2*iqr && v <= q3 + 2*iqr) : e_vals;
                    Plotly.react('timeseries-plot', [{ x: frames, y: plotData.map(d => d['OpenMM_Energy_kcal_mol']), type: 'scatter', mode: 'lines', line: {color: '#d4af37'} }], { title: {text: 'Energy Progression', font: {family: customSettings.fontFamily, color: customSettings.fontColor}}, xaxis: {title: 'Frame Index'}, yaxis: {title: 'Energy (kcal/mol)'}, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' }, {responsive: true});
                    Plotly.react('hist-plot', [{ x: e_filt, type: 'histogram', marker: {color: '#0f251e'} }], { title: {text: 'Energy Distribution (Representative Subset)', font: {family: customSettings.fontFamily, color: customSettings.fontColor}}, xaxis: {title: 'Energy (kcal/mol)'}, yaxis: {title: 'Frequency'}, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' }, {responsive: true});
                }
            }
        }

        function openExportModal() {
            document.getElementById('exp-title').value = customSettings.title;
            document.getElementById('exp-font-family').value = customSettings.fontFamily;
            document.getElementById('exp-title-size').value = customSettings.titleSize;
            document.getElementById('lbl-title-size').innerText = customSettings.titleSize + ' px';
            document.getElementById('exp-axis-size').value = customSettings.axisLabelSize;
            document.getElementById('lbl-axis-size').innerText = customSettings.axisLabelSize + ' px';
            document.getElementById('exp-tick-size').value = customSettings.tickSize;
            document.getElementById('lbl-tick-size').innerText = customSettings.tickSize + ' px';
            document.getElementById('exp-font-color').value = customSettings.fontColor;

            document.getElementById('exp-marker-size').value = customSettings.markerSize;
            document.getElementById('lbl-marker-size').innerText = customSettings.markerSize + ' px';
            document.getElementById('exp-marker-opacity').value = customSettings.markerOpacity;
            document.getElementById('lbl-marker-opacity').innerText = customSettings.markerOpacity;
            document.getElementById('exp-border-width').value = customSettings.borderWidth;
            document.getElementById('lbl-border-width').innerText = customSettings.borderWidth + ' px';
            document.getElementById('exp-border-color').value = customSettings.borderColor;
            document.getElementById('exp-legend-pos').value = customSettings.legendPos;

            document.getElementById('exp-bg-mode').value = customSettings.bgMode;
            toggleCustomBgInput(customSettings.bgMode);
            document.getElementById('exp-custom-bg').value = customSettings.customBg;
            document.getElementById('exp-show-grid').checked = customSettings.showGrid;
            document.getElementById('exp-grid-color').value = customSettings.gridColor;
            document.getElementById('exp-show-zeroline').checked = customSettings.showZeroLine;

            document.getElementById('exp-width').value = customSettings.width;
            document.getElementById('exp-height').value = customSettings.height;
            document.getElementById('exp-scale').value = customSettings.scale;
            document.getElementById('exp-format').value = customSettings.format;

            document.getElementById('overlay').style.display = 'block';
            document.getElementById('export-modal').style.display = 'block';
        }

        function toggleCustomBgInput(mode) {
            const row = document.getElementById('row-custom-bg');
            if (row) row.style.display = (mode === 'custom') ? 'flex' : 'none';
        }

        function applyPresetDimensions(preset) {
            if (preset === '1200x800') { document.getElementById('exp-width').value = 1200; document.getElementById('exp-height').value = 800; }
            else if (preset === '1920x1080') { document.getElementById('exp-width').value = 1920; document.getElementById('exp-height').value = 1080; }
            else if (preset === '2560x1440') { document.getElementById('exp-width').value = 2560; document.getElementById('exp-height').value = 1440; }
            else if (preset === '3840x2160') { document.getElementById('exp-width').value = 3840; document.getElementById('exp-height').value = 2160; }
            else if (preset === '1050x800') { document.getElementById('exp-width').value = 1050; document.getElementById('exp-height').value = 800; }
            else if (preset === '2100x1400') { document.getElementById('exp-width').value = 2100; document.getElementById('exp-height').value = 1400; }
        }

        function liveUpdateTitle() {
            const val = document.getElementById('exp-title').value;
            customSettings.title = val || 'Structural Landscape';
        }

        function updateSettingsFromModal() {
            customSettings.title = document.getElementById('exp-title').value || 'Structural Landscape';
            customSettings.fontFamily = document.getElementById('exp-font-family').value;
            customSettings.titleSize = parseInt(document.getElementById('exp-title-size').value) || 24;
            customSettings.axisLabelSize = parseInt(document.getElementById('exp-axis-size').value) || 14;
            customSettings.tickSize = parseInt(document.getElementById('exp-tick-size').value) || 11;
            customSettings.fontColor = document.getElementById('exp-font-color').value;

            customSettings.markerSize = parseFloat(document.getElementById('exp-marker-size').value) || 5;
            customSettings.markerOpacity = parseFloat(document.getElementById('exp-marker-opacity').value) || 0.85;
            customSettings.borderWidth = parseFloat(document.getElementById('exp-border-width').value) || 0;
            customSettings.borderColor = document.getElementById('exp-border-color').value;
            customSettings.legendPos = document.getElementById('exp-legend-pos').value;

            customSettings.bgMode = document.getElementById('exp-bg-mode').value;
            customSettings.customBg = document.getElementById('exp-custom-bg').value;
            customSettings.showGrid = document.getElementById('exp-show-grid').checked;
            customSettings.gridColor = document.getElementById('exp-grid-color').value;
            customSettings.showZeroLine = document.getElementById('exp-show-zeroline').checked;

            customSettings.width = parseInt(document.getElementById('exp-width').value) || 1920;
            customSettings.height = parseInt(document.getElementById('exp-height').value) || 1080;
            customSettings.scale = parseFloat(document.getElementById('exp-scale').value) || 2;
            customSettings.format = document.getElementById('exp-format').value || 'png';
        }

        function applySettingsToLiveView() {
            updateSettingsFromModal();
            updatePlots();
            closeModal();
        }

        function downloadCustomImage() {
            updateSettingsFromModal();

            let exportBg = '#ffffff';
            if (customSettings.bgMode === 'transparent') exportBg = 'rgba(0,0,0,0)';
            else if (customSettings.bgMode === 'cream') exportBg = '#f4f1ea';
            else if (customSettings.bgMode === 'dark') exportBg = '#0f251e';
            else if (customSettings.bgMode === 'custom') exportBg = customSettings.customBg;

            const mainEl = document.getElementById('main-plot');
            const cx = document.getElementById('sel-x') ? document.getElementById('sel-x').value : 'X';
            const cy = document.getElementById('sel-y') ? document.getElementById('sel-y').value : 'Y';
            const cleanTitle = (customSettings.title || 'plot').replace(/[^a-zA-Z0-9_-]/g, '_');
            const filename = `metflow_${cleanTitle}_${cx}_vs_${cy}`;

            Plotly.relayout(mainEl, {
                title: { text: customSettings.title, font: {family: customSettings.fontFamily, size: customSettings.titleSize, color: customSettings.fontColor} },
                paper_bgcolor: exportBg,
                plot_bgcolor: exportBg,
                font: {family: customSettings.fontFamily, size: customSettings.axisLabelSize, color: customSettings.fontColor},
                xaxis: {
                    title: { text: cx, font: { family: customSettings.fontFamily, size: customSettings.axisLabelSize, color: customSettings.fontColor } },
                    tickfont: { family: customSettings.fontFamily, size: customSettings.tickSize, color: customSettings.fontColor },
                    showgrid: customSettings.showGrid,
                    gridcolor: customSettings.gridColor,
                    zeroline: customSettings.showZeroLine
                },
                yaxis: {
                    title: { text: cy, font: { family: customSettings.fontFamily, size: customSettings.axisLabelSize, color: customSettings.fontColor } },
                    tickfont: { family: customSettings.fontFamily, size: customSettings.tickSize, color: customSettings.fontColor },
                    showgrid: customSettings.showGrid,
                    gridcolor: customSettings.gridColor,
                    zeroline: customSettings.showZeroLine
                }
            }).then(() => {
                return Plotly.downloadImage(mainEl, {
                    format: customSettings.format,
                    width: customSettings.width,
                    height: customSettings.height,
                    scale: customSettings.scale,
                    filename: filename
                });
            }).then(() => {
                closeModal();
            }).catch(err => {
                console.error("Download failed:", err);
                closeModal();
            });
        }

        function closeModal() {
            document.getElementById('overlay').style.display = 'none';
            document.getElementById('pymol-modal').style.display = 'none';
            const expM = document.getElementById('export-modal');
            if (expM) expM.style.display = 'none';
        }
    </script>
</body>
</html>"""


def generate_dashboard(out_dir: str, df: Optional[pd.DataFrame] = None) -> str:
    """Generate metflow_dashboard.html with embedded JSON metrics."""
    os.makedirs(out_dir, exist_ok=True)
    html = DASHBOARD_HTML_TEMPLATE
    if df is not None:
        df_json = df.to_json(orient="records").replace("</", "<\\/")
        html = html.replace("/*EMBEDDED_DATA_PLACEHOLDER*/[]", df_json)
    else:
        csv_file = os.path.join(out_dir, "global_cluster_metrics.csv")
        if os.path.exists(csv_file):
            try:
                df_dash = pd.read_csv(csv_file)
                df_json = df_dash.to_json(orient="records").replace("</", "<\\/")
                html = html.replace("/*EMBEDDED_DATA_PLACEHOLDER*/[]", df_json)
            except Exception:
                pass

    dash_file = os.path.join(out_dir, "metflow_dashboard.html")
    with open(dash_file, "w", encoding="utf-8") as f:
        f.write(html)
    return dash_file
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage6_visualization/plots.py"
"""Static matplotlib/seaborn plots, t-SNE, RMSD heatmaps, and PCA loadings."""

import os
from typing import Any, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE


OKABE_ITO_PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]


def _get_dataset_palette(n_ds: int) -> List[Any]:
    if n_ds <= len(OKABE_ITO_PALETTE):
        return OKABE_ITO_PALETTE[:max(1, n_ds)]
    if n_ds <= 10:
        return sns.color_palette("tab10", n_ds)
    if n_ds <= 20:
        return sns.color_palette("tab20", n_ds)
    return sns.color_palette("husl", n_ds)


def _plot_discrete_clusters(
    x_vals: Any,
    y_vals: Any,
    cluster_vals: Any,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
) -> None:
    """Plot discrete cluster scatter with colorblind-safe palette and explicit legend."""
    unique_cids = sorted(set(cluster_vals), key=lambda c: (c == -1, c))
    non_noise = [c for c in unique_cids if c != -1]
    palette = _get_dataset_palette(len(non_noise))
    color_map = {cid: palette[i % len(palette)] for i, cid in enumerate(non_noise)}
    color_map[-1] = "#7F7F7F"

    plt.figure(figsize=(10, 8))
    x_arr = np.asarray(x_vals)
    y_arr = np.asarray(y_vals)
    c_arr = np.asarray(cluster_vals)
    for cid in unique_cids:
        mask = c_arr == cid
        lbl = f"Noise / Unclustered (-1) (n={int(np.sum(mask))})" if cid == -1 else f"Cluster {cid} (n={int(np.sum(mask))})"
        plt.scatter(
            x_arr[mask],
            y_arr[mask],
            color=color_map[cid],
            label=lbl,
            alpha=0.45 if cid == -1 else 0.7,
            s=8 if cid == -1 else 14,
        )
    plt.legend(title=title, bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=1.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_comparison_plots(
    df: pd.DataFrame,
    out_dir: str,
) -> Tuple[List[str], List[Any], Any]:
    """Generate PCA dataset comparison, energy landscape, and 1D KDE/hist plots."""
    artifacts: List[str] = []
    unique_datasets = list(pd.unique(df["Dataset"]))
    n_ds = len(unique_datasets)
    palette = _get_dataset_palette(n_ds)

    pc1_col_name = [c for c in df.columns if c.startswith("PCA_1")][0]
    pc2_col_name = [c for c in df.columns if c.startswith("PCA_2")][0]

    # 1. PCA Dataset Origin Comparison
    plt.figure(figsize=(10, 8))
    for i, ds in enumerate(unique_datasets):
        sub = df[df["Dataset"] == ds]
        plt.scatter(
            sub[pc1_col_name],
            sub[pc2_col_name],
            label=f"{ds} (n={len(sub)})",
            color=palette[i % len(palette)],
            alpha=0.6,
            s=15,
        )
    plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Multi-Trajectory PCA Projections by Origin Dataset")
    plt.xlabel(pc1_col_name)
    plt.ylabel(pc2_col_name)
    pca_cmp_path = os.path.join(out_dir, "pca_dataset_comparison.png")
    plt.savefig(pca_cmp_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA Dataset Comparison Plot to {pca_cmp_path}")
    artifacts.append(pca_cmp_path)

    # 2. Energy Landscape Plot
    if "OpenMM_Energy_kcal_mol" in df.columns:
        en = df["OpenMM_Energy_kcal_mol"].dropna()
        if len(en) > 0:
            plt.figure(figsize=(10, 8))
            vmax = np.percentile(en, 95)
            vmin = np.min(en)
            scatter = plt.scatter(
                df[pc1_col_name],
                df[pc2_col_name],
                c=df["OpenMM_Energy_kcal_mol"],
                cmap="coolwarm",
                alpha=0.7,
                s=15,
                vmin=vmin,
                vmax=vmax,
            )
            plt.colorbar(scatter, label="Potential Energy (kcal/mol)", extend="max")
            plt.title("Free Energy Landscape (PCA colored by OpenMM Energy, kcal/mol)")
            plt.xlabel(pc1_col_name)
            plt.ylabel(pc2_col_name)
            en_plot_path = os.path.join(out_dir, "energy_landscape_plot.png")
            plt.savefig(en_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved Energy Landscape Plot to {en_plot_path}")
            artifacts.append(en_plot_path)

    # 3. PCA 1D Distributions partitioned by origin dataset
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for i, ds in enumerate(unique_datasets):
        sub = df[df["Dataset"] == ds]
        color = palette[i % len(palette)]
        lbl = f"{ds} (n={len(sub)})"
        if len(sub) > 1:
            try:
                sns.kdeplot(data=sub[pc1_col_name], ax=ax1, label=lbl, color=color, fill=True, alpha=0.25, warn_singular=False)
                sns.kdeplot(data=sub[pc2_col_name], ax=ax2, label=lbl, color=color, fill=True, alpha=0.25, warn_singular=False)
            except Exception:
                ax1.hist(sub[pc1_col_name], bins=30, density=True, alpha=0.4, color=color, label=lbl)
                ax2.hist(sub[pc2_col_name], bins=30, density=True, alpha=0.4, color=color, label=lbl)
        else:
            ax1.hist(sub[pc1_col_name], bins=30, density=True, alpha=0.4, color=color, label=lbl)
            ax2.hist(sub[pc2_col_name], bins=30, density=True, alpha=0.4, color=color, label=lbl)

    ax1.set_title("Conformational Density along PC1 by Dataset")
    ax1.set_xlabel(pc1_col_name)
    ax1.set_ylabel("Density")
    ax1.legend(title="Dataset")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Conformational Density along PC2 by Dataset")
    ax2.set_xlabel(pc2_col_name)
    ax2.set_ylabel("Density")
    ax2.legend(title="Dataset")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    dist_path = os.path.join(out_dir, "pca_1d_distributions.png")
    plt.savefig(dist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA 1D Distributions to {dist_path}")
    artifacts.append(dist_path)

    return artifacts, unique_datasets, palette


def generate_cluster_plots(
    df: pd.DataFrame,
    out_dir: str,
    var_explained: np.ndarray,
) -> List[str]:
    """Generate 2D PCA cluster scatter plots for each clustering method present."""
    artifacts: List[str] = []
    pc1_col = f"PCA_1 ({var_explained[0]:.1f}%)"
    pc2_col = f"PCA_2 ({var_explained[1]:.1f}%)"
    if pc1_col not in df.columns:
        pc1_col = [c for c in df.columns if c.startswith("PCA_1")][0]
    if pc2_col not in df.columns:
        pc2_col = [c for c in df.columns if c.startswith("PCA_2")][0]

    for cluster_col, title, out_name in [
        ("PCA_KMeans_Cluster", "K-Means Clusters", "pca_kmeans_plot.png"),
        ("Daura_Cluster", "Daura Clusters", "daura_plot.png"),
        ("CNN_Cluster", "CNN/DBSCAN Clusters", "cnn_plot.png"),
    ]:
        if cluster_col in df.columns:
            out_path = os.path.join(out_dir, out_name)
            _plot_discrete_clusters(
                df[pc1_col],
                df[pc2_col],
                df[cluster_col],
                title=f"PCA Projection ({title})",
                xlabel=f"Principal Component 1 ({var_explained[0]:.1f}%)",
                ylabel=f"Principal Component 2 ({var_explained[1]:.1f}%)",
                out_path=out_path,
            )
            print(f"Saved 2D Cluster Graph to {out_path}")
            artifacts.append(out_path)
    return artifacts


def generate_advanced_plots(
    df: pd.DataFrame,
    X: np.ndarray,
    rmsd_matrix: Optional[np.ndarray],
    out_dir: str,
    unique_datasets: List[str],
    palette: Any,
    methodology: List[str],
) -> List[str]:
    """Generate t-SNE projections (with safe perplexity) and RMSD cluster heatmaps."""
    artifacts: List[str] = []
    print("\n--- Generating Advanced Non-linear Projections & Heatmaps ---")
    methodology.append("\n--- Advanced Cluster Visualizations ---")

    n_samples = len(X)
    if n_samples >= 3:
        perplexity = min(30, max(1, (n_samples - 1) // 3))
        print(f"Calculating t-SNE projection (perplexity={perplexity}, this may take a minute)...")
        tsne_model = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
        X_tsne = tsne_model.fit_transform(X)
        df["tSNE_1"] = X_tsne[:, 0]
        df["tSNE_2"] = X_tsne[:, 1]

        plt.figure(figsize=(10, 8))
        for i, ds in enumerate(unique_datasets):
            sub = df[df["Dataset"] == ds]
            plt.scatter(
                sub["tSNE_1"],
                sub["tSNE_2"],
                label=f"{ds} (n={len(sub)})",
                color=palette[i % len(palette)],
                alpha=0.6,
                s=15,
            )
        plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title("Multi-Trajectory t-SNE Projections by Origin Dataset")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        tsne_cmp_path = os.path.join(out_dir, "tsne_dataset_comparison.png")
        plt.savefig(tsne_cmp_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved t-SNE Dataset Comparison Plot to {tsne_cmp_path}")
        artifacts.append(tsne_cmp_path)

        for cluster_col, title, out_name in [
            ("Daura_Cluster", "Daura Clusters", "tsne_daura_plot.png"),
            ("CNN_Cluster", "CNN/DBSCAN Clusters", "tsne_cnn_plot.png"),
        ]:
            if cluster_col in df.columns:
                out_path = os.path.join(out_dir, out_name)
                _plot_discrete_clusters(
                    df["tSNE_1"],
                    df["tSNE_2"],
                    df[cluster_col],
                    title=f"t-SNE Projection ({title})",
                    xlabel="t-SNE Dimension 1",
                    ylabel="t-SNE Dimension 2",
                    out_path=out_path,
                )
                print(f"Saved t-SNE Graph to {out_path}")
                artifacts.append(out_path)
                methodology.append(f"Generated t-SNE non-linear projection for {title}.")
    else:
        print(f"⚠️ Skipping t-SNE: need at least 3 samples (got {n_samples}).")

    if rmsd_matrix is not None:
        print("Generating Cluster RMSD Heatmaps...")
        for cluster_col, title, out_name in [
            ("Daura_Cluster", "Daura", "daura_rmsd_heatmap.png"),
            ("CNN_Cluster", "CNN/DBSCAN", "cnn_rmsd_heatmap.png"),
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
                out_path = os.path.join(out_dir, out_name)
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Saved RMSD Heatmap to {out_path}")
                artifacts.append(out_path)
                methodology.append(f"Generated RMSD pairwise distance block heatmap for {title} clusters.")

    return artifacts


def generate_pca_loadings(
    pca_model: Any,
    ref_bb_atoms: Any,
    out_dir: str,
) -> List[str]:
    """Save pca_loadings.csv and pca_loadings_plot.png."""
    n_atoms = len(ref_bb_atoms)
    resids = ref_bb_atoms.resids
    components = pca_model.components_

    atom_contrib_pc1 = np.sqrt(np.sum(components[0].reshape(n_atoms, 3) ** 2, axis=1))
    if components.shape[0] > 1:
        atom_contrib_pc2 = np.sqrt(np.sum(components[1].reshape(n_atoms, 3) ** 2, axis=1))
    else:
        atom_contrib_pc2 = np.zeros(n_atoms, dtype=np.float64)

    unique_resids, res_inv = np.unique(resids, return_inverse=True)
    res_contrib_pc1 = np.bincount(res_inv, weights=atom_contrib_pc1)
    res_contrib_pc2 = np.bincount(res_inv, weights=atom_contrib_pc2)

    all_resids = np.arange(unique_resids.min(), unique_resids.max() + 1)
    pc1_full = np.full(len(all_resids), np.nan)
    pc2_full = np.full(len(all_resids), np.nan)

    idx_arr = unique_resids - unique_resids.min()
    pc1_full[idx_arr] = res_contrib_pc1
    pc2_full[idx_arr] = res_contrib_pc2

    csv_path = os.path.join(out_dir, "pca_loadings.csv")
    pd.DataFrame({
        "Residue": all_resids,
        "PC1_Contribution": pc1_full,
        "PC2_Contribution": pc2_full,
    }).to_csv(csv_path, index=False)
    print(f"Saved PCA Loadings data to {csv_path}")

    plot_path = os.path.join(out_dir, "pca_loadings_plot.png")
    plt.figure(figsize=(12, 5))
    plt.plot(all_resids, pc1_full, label="PC1 Contribution", alpha=0.85, color="#0072B2", marker=".", markersize=4)
    plt.plot(all_resids, pc2_full, label="PC2 Contribution", alpha=0.85, color="#D55E00", marker=".", markersize=4)
    plt.xlabel("Residue Number")
    plt.ylabel("Loading Magnitude (RMS, Å)")
    plt.title("Per-Residue Contribution to Principal Components")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA Loadings Plot to {plot_path}")

    return [csv_path, plot_path]
EOF
cat << 'EOF' > "$INSTALL_DIR/metflow/stage6_visualization/reports.py"
"""Text reports (cluster summary, methodology) and Stage 6 orchestration."""

import datetime
import os
from typing import List, Optional
import pandas as pd

from metflow.models import PipelineConfig, StageResult, TrajectoryContext
from metflow.stage6_visualization.dashboard import generate_dashboard
from metflow.stage6_visualization.plots import (
    generate_advanced_plots,
    generate_cluster_plots,
    generate_comparison_plots,
    generate_pca_loadings,
)


def init_methodology(ctx: TrajectoryContext, mode_label: str = "Standard Analysis") -> str:
    """Initialize or append a timestamped session header to out_dir/methodology.txt."""
    out_dir = ctx.out_dir or (ctx.config.out_dir if ctx.config else "")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "methodology.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"=== MET Flow Session: {timestamp} | Mode: {mode_label} ==="

    ctx.methodology.append(header)
    file_exists = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    with open(out_path, "a", encoding="utf-8") as f:
        if file_exists:
            f.write("\n")
        f.write(header + "\n")
        f.flush()
    return out_path


def log_methodology(ctx: TrajectoryContext, entry: str) -> None:
    """Append an entry to ctx.methodology and immediately flush to out_dir/methodology.txt."""
    if not entry:
        return
    ctx.methodology.append(entry)
    out_dir = ctx.out_dir or (ctx.config.out_dir if ctx.config else "")
    if not out_dir:
        return
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "methodology.txt")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


def write_cluster_summary(
    df: pd.DataFrame,
    out_dir: str,
    unique_datasets: List[str],
    config: Optional[PipelineConfig] = None,
) -> str:
    """Write dataset_cluster_summary.txt with per-dataset stats and full cross-tabulations."""
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_lines = [
        "============================================================",
        "  MET Flow - Multi-Trajectory Dataset & Cluster Summary",
        "============================================================",
        f"Generated: {timestamp}",
        f"Total Datasets: {len(unique_datasets)}",
        f"Total Frames: {len(df)}",
    ]

    if config is not None:
        summary_lines.extend([
            "",
            "--- Clustering Parameters ---",
            f"  Method(s): {config.cluster}",
            f"  K-Means k: {config.k}",
            f"  Daura RMSD Cutoff: {config.rmsd} Å",
            f"  CNN/DBSCAN eps: {config.cnn_eps} Å",
        ])

    if len(df) == 0:
        summary_lines.append("\nNo frames available for summary statistics.")
        out_path = os.path.join(out_dir, "dataset_cluster_summary.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n")
        print(f"Saved dataset cluster summary report to {out_path}")
        return out_path

    summary_lines.append("")
    summary_lines.append("--- Per-Dataset Frame Counts ---")
    total_frames = len(df)
    ds_counts = df["Dataset"].value_counts().to_dict()
    for ds in unique_datasets:
        ds_count = int(ds_counts.get(ds, 0))
        summary_lines.append(f"Dataset '{ds}': {ds_count} frames ({ds_count / total_frames * 100:.1f}%)")

    rmsd_col = "RMSD_to_Reference" if "RMSD_to_Reference" in df.columns else (
        "RMSD_to_Ref" if "RMSD_to_Ref" in df.columns else None
    )
    energy_col = "OpenMM_Energy_kcal_mol" if "OpenMM_Energy_kcal_mol" in df.columns else None

    if rmsd_col or energy_col:
        summary_lines.append("\n--- Per-Dataset Quantitative Summary ---")
        rmsd_stats = None
        if rmsd_col:
            rmsd_stats = (
                pd.to_numeric(df[rmsd_col], errors="coerce")
                .groupby(df["Dataset"], sort=False)
                .agg(["count", "mean", "std", "min", "median", "max"])
            )
        en_stats = None
        if energy_col:
            en_stats = (
                pd.to_numeric(df[energy_col], errors="coerce")
                .groupby(df["Dataset"], sort=False)
                .agg(["count", "mean", "std", "min", "median", "max"])
            )
        for ds in unique_datasets:
            summary_lines.append(f"Dataset '{ds}':")
            if rmsd_stats is not None and ds in rmsd_stats.index:
                r = rmsd_stats.loc[ds]
                if r["count"] > 0:
                    summary_lines.append(
                        f"  RMSD to Reference (Å): mean={r['mean']:.3f} ± {r['std']:.3f}, "
                        f"min={r['min']:.3f}, median={r['median']:.3f}, max={r['max']:.3f}"
                    )
            if en_stats is not None and ds in en_stats.index:
                e = en_stats.loc[ds]
                if e["count"] > 0:
                    ds_total = int(ds_counts.get(ds, 0))
                    summary_lines.append(
                        f"  OpenMM Energy (kcal/mol): mean={e['mean']:.2f} ± {e['std']:.2f}, "
                        f"min={e['min']:.2f}, median={e['median']:.2f}, max={e['max']:.2f} "
                        f"(valid={int(e['count'])}/{ds_total})"
                    )

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        for cluster_col in ["PCA_KMeans_Cluster", "Daura_Cluster", "CNN_Cluster"]:
            if cluster_col in df.columns:
                summary_lines.append(f"\n--- Cross-Tabulation: {cluster_col} ---")
                ct_raw = pd.crosstab(df[cluster_col], df["Dataset"])
                row_sums = ct_raw.sum(axis=1)
                ct_pct = ct_raw.div(row_sums, axis=0) * 100
                ct = ct_raw.copy()
                ct["All"] = row_sums
                ct.loc["All"] = ct.sum(axis=0)
                if cluster_col == "CNN_Cluster":
                    ct = ct.rename(index={-1: "Noise / Unclustered (-1)"})
                    ct_pct = ct_pct.rename(index={-1: "Noise / Unclustered (-1)"})
                summary_lines.append("Counts:")
                summary_lines.append(ct.to_string())
                summary_lines.append("\nRow Percentages (%):")
                summary_lines.append(ct_pct.round(1).to_string())

    out_path = os.path.join(out_dir, "dataset_cluster_summary.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"Saved dataset cluster summary report to {out_path}")
    return out_path


def write_methodology(methodology: List[str], out_dir: str) -> str:
    """Write methodology.txt (backward-compatible batch writer)."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "methodology.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(methodology) + "\n")
    print("Saved methodology.txt")
    return out_path


def run_stage6(ctx: TrajectoryContext) -> StageResult:
    """Execute Stage 6: CSV export, HTML dashboard, plots, and text reports."""
    config = ctx.config
    out_dir = ctx.out_dir
    var_explained = ctx.var_explained
    artifacts: List[str] = []

    print("\n--- Generating Final Outputs ---")
    log_methodology(ctx, "\n--- Stage 6: Visualization & Reports ---")
    df = pd.DataFrame(ctx.df_data)
    try:
        df.rename(
            columns={
                "PCA_1": f"PCA_1 ({var_explained[0]:.1f}%)",
                "PCA_2": f"PCA_2 ({var_explained[1]:.1f}%)",
                "PCA_3": f"PCA_3 ({var_explained[2]:.1f}%)",
            },
            inplace=True,
        )
    except Exception:
        pass
    ctx.df = df

    csv_path = os.path.join(out_dir, "global_cluster_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved complete metrics to {csv_path}")
    artifacts.append(csv_path)

    dash_path = generate_dashboard(out_dir, df=df)
    print(f"🌟 Generated interactive Dashboard (with embedded data for zero-CORS offline browsing!): {dash_path}")
    artifacts.append(dash_path)
    log_methodology(ctx, f"Exported global_cluster_metrics.csv ({len(df)} frames) and metflow_dashboard.html.")

    cmp_artifacts, unique_datasets, palette = generate_comparison_plots(df, out_dir)
    artifacts.extend(cmp_artifacts)

    meth_path = os.path.join(out_dir, "methodology.txt")
    if config.is_only_energy:
        log_methodology(ctx, "Stage 6 completed (Energy Only mode).")
        artifacts.append(meth_path)
        print("==================================================")
        print(" ✅ Option [C] Energy ONLY Evaluation Complete!")
        print("==================================================")
        return StageResult(
            stage_name="stage6_visualization",
            success=True,
            exit_code=0,
            artifacts=artifacts,
            metrics={"only_energy": True, "n_frames": len(df)},
        )

    artifacts.append(write_cluster_summary(df, out_dir, unique_datasets, config=config))
    artifacts.extend(generate_cluster_plots(df, out_dir, var_explained))

    if config.is_advanced_plots:
        prev_len = len(ctx.methodology)
        artifacts.extend(
            generate_advanced_plots(
                df, ctx.X, ctx.rmsd_matrix, out_dir, unique_datasets, palette, ctx.methodology
            )
        )
        new_entries = ctx.methodology[prev_len:]
        if new_entries:
            with open(meth_path, "a", encoding="utf-8") as f:
                f.write("\n".join(new_entries) + "\n")

    artifacts.extend(generate_pca_loadings(ctx.pca_model, ctx.ref_bb_atoms, out_dir))
    log_methodology(ctx, "Stage 6 completed successfully (plots, PCA loadings, and cluster summary exported).")
    artifacts.append(meth_path)

    print(f"\n✅ Post-Generation Analysis Complete! Results saved to {out_dir}/ directory.")
    print("==================================================")
    print(" ✅ Post-Generation Analysis Complete!")
    print("==================================================")

    return StageResult(
        stage_name="stage6_visualization",
        success=True,
        exit_code=0,
        artifacts=artifacts,
        metrics={"n_frames": len(df), "n_datasets": len(unique_datasets)},
    )
EOF
# METFLOW_MODULES_END

# ----- BASH WRAPPER: metflow -----
cat << 'EOF' > "$INSTALL_DIR/bin/metflow"
#!/bin/bash
set -e

SCRIPT_PATH=$(readlink -f "$0" 2>/dev/null || readlink -f "$BASH_SOURCE")
BIN_DIR=$(dirname "$SCRIPT_PATH")
INSTALL_DIR=$(dirname "$BIN_DIR")
if [ ! -f "$INSTALL_DIR/analyze_trajectory.py" ]; then
    if [ -f "./analyze_trajectory.py" ]; then
        INSTALL_DIR="$(pwd)"
    elif [ -f "$BIN_DIR/analyze_trajectory.py" ]; then
        INSTALL_DIR="$BIN_DIR"
    elif [ -f "$HOME/metflow_pipeline/analyze_trajectory.py" ]; then
        INSTALL_DIR="$HOME/metflow_pipeline"
    fi
fi
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

validate_out_dir() {
    local target="$1"
    if [ -z "$target" ]; then
        return 0
    fi
    if [[ "$target" =~ (^|/)\.\.(/|$) ]]; then
        echo "❌ Error: Directory traversal blocked: $target" >&2
        exit 1
    fi
    local cwd
    cwd=$(pwd -P 2>/dev/null || pwd)
    local norm_target
    norm_target=$(realpath -m "$target" 2>/dev/null || echo "")
    if [ -n "$norm_target" ]; then
        if [[ "$norm_target" != "$cwd" ]] && [[ "$norm_target" != "$cwd/"* ]]; then
            echo "❌ Error: Directory traversal blocked: $target" >&2
            exit 1
        fi
    elif [[ "$target" = /* ]]; then
        if [[ "$target" != "$cwd" ]] && [[ "$target" != "$cwd/"* ]]; then
            echo "❌ Error: Directory traversal blocked: $target" >&2
            exit 1
        fi
    fi
}

run_analysis() {
    echo "================================================="
    echo "   MET Pipeline Analysis & Clustering Menu"
    echo "================================================="
    
    PARAM_FILE=""
    CHECK_OUT_DIR="${CUSTOM_OUT:-clustering_results}"
    if [ -f "$CHECK_OUT_DIR/registers/metflow_params.txt" ]; then
        PARAM_FILE="$CHECK_OUT_DIR/registers/metflow_params.txt"
    elif [ -f "metflow_params.txt" ]; then
        PARAM_FILE="metflow_params.txt"
    else
        PARAM_FILE=$(find . -maxdepth 3 -path "*/registers/metflow_params.txt" 2>/dev/null | head -n 1 || true)
    fi
    LOAD_PREV=false

    if [ "$2" == "true" ]; then
        OUT_DIR_VAL="${CUSTOM_OUT:-clustering_results}"
        validate_out_dir "$OUT_DIR_VAL"
        CMD_ARGS=(--only_dashboard "true" --out_dir "$OUT_DIR_VAL")
        mkdir -p "$OUT_DIR_VAL/registers"
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
dataset_configs=${CUSTOM_CONFIGS:-}
out_dir=${OUT_DIR_VAL}
multi_datasets=${IS_MULTI_DATASET:-false}
only_energy=false
only_dashboard=true"
        echo "$PARAM_CONTENT" > "$OUT_DIR_VAL/registers/metflow_params.txt"

        TARGET_SCRIPT="$INSTALL_DIR/analyze_trajectory.py"
        if [ -f "./analyze_trajectory.py" ]; then
            TARGET_SCRIPT="./analyze_trajectory.py"
        elif [ -f "$BIN_DIR/analyze_trajectory.py" ]; then
            TARGET_SCRIPT="$BIN_DIR/analyze_trajectory.py"
        fi

        micromamba run -n conf_sel_analysis python "$TARGET_SCRIPT" "${CMD_ARGS[@]}"
        exit 0
    fi

    if [ "$1" != "true" ] && [ "$2" != "true" ] && [ -n "$PARAM_FILE" ] && [ -f "$PARAM_FILE" ]; then
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
                pca_res) [ -n "$val" ] && CMD_ARGS+=(--pca_res "$val") ;;
                pdb) [ -n "$val" ] && CMD_ARGS+=(--pdb "$val") ;;
                xtc) [ -n "$val" ] && CMD_ARGS+=(--xtc "$val") ;;
                labels|dataset_labels) [ -n "$val" ] && CMD_ARGS+=(--dataset_labels "$val") ;;
                dataset_configs) [ -n "$val" ] && CMD_ARGS+=(--dataset_configs "$val") ;;
                out_dir) [ -n "$val" ] && validate_out_dir "$val" && CMD_ARGS+=(--out_dir "$val"); OUT_DIR_VAL="$val" ;;
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
        if [ -n "$CUSTOM_CONFIGS" ]; then
            CMD_ARGS+=(--dataset_configs "$CUSTOM_CONFIGS")
        fi
        OUT_DIR_VAL="${CUSTOM_OUT:-energy_results}"
        validate_out_dir "$OUT_DIR_VAL"
        CMD_ARGS+=(--out_dir "$OUT_DIR_VAL")
        if [ -n "$IS_MULTI_DATASET" ]; then
            CMD_ARGS+=(--multi_datasets "$IS_MULTI_DATASET")
        fi

        mkdir -p "$OUT_DIR_VAL/registers"
        PARAM_CONTENT="max_frames=${max_frames_input:-0}
skim=${SKIM_VAL}
cluster=${CLUSTER_OPT}
calc_energy=${CALC_ENERGY_VAL}
min_steps=${MIN_STEPS}
allostery=${ALLOSTERY_VAL}
pca_ref=${PCA_REF_VAL}
pca_res=${PCA_RES_VAL}
pdb=${CUSTOM_PDB:-}
xtc=${CUSTOM_XTC:-}
labels=${CUSTOM_LABELS:-}
dataset_configs=${CUSTOM_CONFIGS:-}
out_dir=${OUT_DIR_VAL}
multi_datasets=${IS_MULTI_DATASET:-false}
only_energy=${ONLY_ENERGY_VAL}
only_dashboard=${ONLY_DASHBOARD_VAL}"
        echo "$PARAM_CONTENT" > "$OUT_DIR_VAL/registers/metflow_params.txt"
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

        echo ""
        read -r -p "Restrict Alignment & PCA to specific residues/ranges? (e.g. 10-45, 60 or leave blank for ALL): " pca_res_input

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
        PCA_RES_VAL="${pca_res_input:-}"
        if [ -n "$PCA_RES_VAL" ]; then
            CMD_ARGS+=(--pca_res "$PCA_RES_VAL")
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
        if [ -n "$CUSTOM_CONFIGS" ]; then
            CMD_ARGS+=(--dataset_configs "$CUSTOM_CONFIGS")
        fi
        OUT_DIR_VAL="${CUSTOM_OUT:-clustering_results}"
        validate_out_dir "$OUT_DIR_VAL"
        CMD_ARGS+=(--out_dir "$OUT_DIR_VAL")
        if [ -n "$IS_MULTI_DATASET" ]; then
            CMD_ARGS+=(--multi_datasets "$IS_MULTI_DATASET")
        fi

        ONLY_ENERGY_VAL="false"
        ONLY_DASHBOARD_VAL="false"

        mkdir -p "$OUT_DIR_VAL/registers"
        PARAM_CONTENT="max_frames=${max_frames_input:-0}
skim=${SKIM_VAL}
cluster=${CLUSTER_OPT}
calc_energy=${CALC_ENERGY_VAL}
min_steps=${MIN_STEPS}
allostery=${ALLOSTERY_VAL}
pca_ref=${PCA_REF_VAL}
pca_res=${PCA_RES_VAL}
pdb=${CUSTOM_PDB:-}
xtc=${CUSTOM_XTC:-}
labels=${CUSTOM_LABELS:-}
dataset_configs=${CUSTOM_CONFIGS:-}
out_dir=${OUT_DIR_VAL}
multi_datasets=${IS_MULTI_DATASET:-false}
only_energy=${ONLY_ENERGY_VAL}
only_dashboard=${ONLY_DASHBOARD_VAL}"
        echo "$PARAM_CONTENT" > "$OUT_DIR_VAL/registers/metflow_params.txt"
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
        LOG_DIR="${OUT_DIR_VAL:-clustering_results}/logs"
        mkdir -p "$LOG_DIR"
        LOG_OUT="$LOG_DIR/metflow_analysis.log"
        nohup micromamba run -n conf_sel_analysis python "$TARGET_SCRIPT" "${CMD_ARGS[@]}" < /dev/null > "$LOG_OUT" 2>&1 &
        BG_PID=$!
        disown "$BG_PID" 2>/dev/null || true
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
    RUNNING_PID=""
    TARGET_SAMPLES=0
    while IFS='=' read -r skey sval || [ -n "$skey" ]; do
        skey="${skey%$'\r'}"
        sval="${sval%$'\r'}"
        case "$skey" in
            PID) [[ "$sval" =~ ^[0-9]+$ ]] && RUNNING_PID="$sval" ;;
            TARGET_SAMPLES) [[ "$sval" =~ ^[0-9]+$ ]] && TARGET_SAMPLES="$sval" ;;
        esac
    done < "$STATUS_FILE"

    if [ -n "$RUNNING_PID" ] && kill -0 "$RUNNING_PID" 2>/dev/null; then
        COMPLETED_SAMPLES=0
        if [ -d "$OUTPUT_DIR" ]; then
            pdb_files=("$OUTPUT_DIR"/*.pdb)
            if [ -e "${pdb_files[0]}" ]; then
                COMPLETED_SAMPLES=${#pdb_files[@]}
            fi
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
            2) kill -STOP "$RUNNING_PID"; echo "❄️  Paused!"; exit 0 ;;
            3) kill -CONT "$RUNNING_PID"; echo "🔥 Resumed!"; exit 0 ;;
            4) kill -9 "$RUNNING_PID"; rm -f "$STATUS_FILE"; echo "💀 Killed." ;;
            5) DO_ANALYSIS=true ;;
            *) echo "Check progress with: tail -f $LOG_FILE"; exit 0 ;;
        esac
    else
        rm -f "$STATUS_FILE"
    fi
elif [ -n "$RUNNING_PID" ] && [[ "$RUNNING_PID" =~ ^[0-9]+$ ]]; then
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
        2) kill -STOP "$RUNNING_PID"; echo "❄️  Paused!"; exit 0 ;;
        3) kill -CONT "$RUNNING_PID"; echo "🔥 Resumed!"; exit 0 ;;
        4) kill -9 "$RUNNING_PID"; rm -f "$STATUS_FILE"; echo "💀 Killed." ;;
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
    echo "   MET Pipeline Launch Configuration (v2.8 MULTI-SOURCE)"
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
                    if [ -n "$input_val" ] && [[ "$input_val" =~ ^[0-9]+$ ]]; then NUM_SAMPLES="$input_val"; fi
                    ;;
                2)
                    read -p "Value (Default: 50): " input_val
                    if [ -n "$input_val" ] && [[ "$input_val" =~ ^[0-9]+$ ]]; then DENOISING_STEPS="$input_val"; fi
                    ;;
                3)
                    read -p "Value (dpm/euler): " input_val
                    if [ "$input_val" == "dpm" ] || [ "$input_val" == "euler" ]; then DENOISER_SOLVER="$input_val"; fi
                    ;;
            esac
            ;;
        [Aa])
            unset CUSTOM_PDB CUSTOM_XTC CUSTOM_LABELS CUSTOM_OUT IS_MULTI_DATASET CUSTOM_CONFIGS
            echo -e "\n--- Option [A]: Analyze BioEmu Results ---"
            echo "Select Master Reference Topology PDB for BioEmu analysis:"
            echo "  [1] ./bioemu_ensemble/topology.pdb [DEFAULT]"
            echo "  [2] Custom PDB file in working directory..."
            read -p "Selection [1-2] (Default: 1): " bioemu_pdb_choice
            if [[ "$bioemu_pdb_choice" == "2" ]]; then
                read -p "Enter path to Custom Reference PDB file: " bioemu_pdb
            else
                bioemu_pdb="./bioemu_ensemble/topology.pdb"
            fi
            bioemu_xtc="./bioemu_ensemble/samples.xtc"
            if [ ! -f "$bioemu_xtc" ] && [ -f "samples.xtc" ]; then
                bioemu_xtc="samples.xtc"
            fi

            echo ""
            read -p "Do you want to compare BioEmu against an external reference trajectory (e.g. Holo/Apo simulation)? [y/N]: " compare_ext_ans
            if [[ "$compare_ext_ans" =~ ^[Yy]$ ]]; then
                read -p "Enter external reference PDB file: " ext_ref_pdb
                read -p "Enter external reference XTC file: " ext_ref_xtc
                read -p "Enter label for external reference [Default: External_Ref]: " ext_ref_lbl
                ext_ref_lbl="${ext_ref_lbl:-External_Ref}"
                read -p "Enter label for BioEmu [Default: BioEmu_Ensemble]: " bioemu_lbl
                bioemu_lbl="${bioemu_lbl:-BioEmu_Ensemble}"
                read -p "Output folder name [Default: bioemu_comparative_results]: " out_dir_in
                CUSTOM_OUT="${out_dir_in:-bioemu_comparative_results}"
                CUSTOM_PDB="$bioemu_pdb,$ext_ref_pdb"
                CUSTOM_XTC="$bioemu_xtc,$ext_ref_xtc"
                CUSTOM_LABELS="$bioemu_lbl,$ext_ref_lbl"
                IS_MULTI_DATASET="true"
            else
                CUSTOM_PDB="$bioemu_pdb"
                CUSTOM_XTC="$bioemu_xtc"
                CUSTOM_LABELS="BioEmu_Ensemble"
            fi
            run_analysis
            ;;
        [Cc])
            unset CUSTOM_PDB CUSTOM_XTC CUSTOM_LABELS CUSTOM_OUT IS_MULTI_DATASET CUSTOM_CONFIGS
            echo -e "\n--- Option [C]: Calculate Energy ONLY ---"
            echo "  [1] Single Trajectory Dataset (Default: topology.pdb & samples.xtc)"
            echo "  [2] Multi-Trajectory Comparative Analysis (2+ Datasets)"
            echo "  [3] Auto-Detect Matching Datasets in Directory (PDB & XTC with identical stems)"
            echo "  [4] Batch / Comma-Separated Pairs (PDB1,XTC1,Label1;PDB2,XTC2,Label2)"
            read -p "Selection [1-4] (Default: 1): " c_mode

            if [[ "$c_mode" == "4" ]]; then
                echo "Paste dataset configuration string in format: PDB1,XTC1,Label1;PDB2,XTC2,Label2"
                read -p "Dataset configs: " CUSTOM_CONFIGS
                read -p "Output folder name [Default: energy_batch_results]: " out_dir_in
                CUSTOM_OUT="${out_dir_in:-energy_batch_results}"
                IS_MULTI_DATASET="true"
            elif [[ "$c_mode" == "3" ]]; then
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
                old_ifs="$IFS"
                IFS=,
                CUSTOM_PDB="${PDB_ARRAY[*]}"
                CUSTOM_XTC="${XTC_ARRAY[*]}"
                CUSTOM_LABELS="${LABEL_ARRAY[*]}"
                IFS="$old_ifs"
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
                old_ifs="$IFS"
                IFS=,
                CUSTOM_PDB="${PDB_ARRAY[*]}"
                CUSTOM_XTC="${XTC_ARRAY[*]}"
                CUSTOM_LABELS="${LABEL_ARRAY[*]}"
                IFS="$old_ifs"
                IS_MULTI_DATASET="true"
            else
                read -p "Output folder name [Default: energy_results]: " out_dir_in
                CUSTOM_OUT="${out_dir_in:-energy_results}"
            fi
            run_analysis "true"
            ;;
        [Dd])
            unset CUSTOM_PDB CUSTOM_XTC CUSTOM_LABELS CUSTOM_OUT IS_MULTI_DATASET CUSTOM_CONFIGS
            read -p "Enter results folder containing global_cluster_metrics.csv [Default: clustering_results]: " custom_dash_out
            CUSTOM_OUT="${custom_dash_out:-clustering_results}"
            run_analysis "false" "true"
            ;;
        [Ee])
            unset CUSTOM_PDB CUSTOM_XTC CUSTOM_LABELS CUSTOM_OUT IS_MULTI_DATASET CUSTOM_CONFIGS
            echo -e "\n--- Analyze External Trajectory ---"
            echo "[1] Single Trajectory Dataset"
            echo "[2] Multi-Trajectory Comparative Analysis (2+ Datasets)"
            echo "[3] Auto-Detect Matching Datasets in Directory (PDB & XTC sharing same prefix/stem)"
            echo "[4] Batch / Comma-Separated Pairs (PDB1,XTC1,Label1;PDB2,XTC2,Label2)"
            echo "(Note: PDB and XTC pairs must share the same filename stem to be auto-detected in [3])"
            read -p "Selection [1-4] (Default: 1): " ext_mode
            
            if [[ "$ext_mode" == "4" ]]; then
                echo "Paste dataset configuration string in format: PDB1,XTC1,Label1;PDB2,XTC2,Label2"
                read -p "Dataset configs: " CUSTOM_CONFIGS
                read -p "Output folder name [Default: batch_comparative_results]: " out_dir_in
                CUSTOM_OUT="${out_dir_in:-batch_comparative_results}"
                IS_MULTI_DATASET="true"
            elif [[ "$ext_mode" == "3" ]]; then
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
                old_ifs="$IFS"
                IFS=,
                CUSTOM_PDB="${PDB_ARRAY[*]}"
                CUSTOM_XTC="${XTC_ARRAY[*]}"
                CUSTOM_LABELS="${LABEL_ARRAY[*]}"
                IFS="$old_ifs"
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
                old_ifs="$IFS"
                IFS=,
                CUSTOM_PDB="${PDB_ARRAY[*]}"
                CUSTOM_XTC="${XTC_ARRAY[*]}"
                CUSTOM_LABELS="${LABEL_ARRAY[*]}"
                IFS="$old_ifs"
                IS_MULTI_DATASET="true"
            else
                read -p "External PDB file: " CUSTOM_PDB
                read -p "External XTC trajectory file: " CUSTOM_XTC
                read -p "Dataset Label [Default: External]: " CUSTOM_LABELS
                CUSTOM_LABELS="${CUSTOM_LABELS:-External}"
                read -p "Output folder name [Default: external_results]: " CUSTOM_OUT
                CUSTOM_OUT="${CUSTOM_OUT:-external_results}"
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

EXTRA_ARGS=(--num_samples "$NUM_SAMPLES" --denoising_steps "$DENOISING_STEPS" --denoiser_type "$DENOISER_SOLVER")
CPU_CORES=$(nproc 2>/dev/null || echo 1)

case $load_choice in
    2) THREADS=$(( CPU_CORES * 85 / 100 )); EXTRA_ARGS+=(--batch_size_100 40) ;;
    3) THREADS=$(( CPU_CORES * 70 / 100 )); EXTRA_ARGS+=(--batch_size_100 35) ;;
    4) THREADS=$(( CPU_CORES * 55 / 100 )); EXTRA_ARGS+=(--batch_size_100 25) ;;
    5) THREADS=$(( CPU_CORES * 40 / 100 )); EXTRA_ARGS+=(--batch_size_100 20) ;;
    6) THREADS=$(( CPU_CORES * 25 / 100 )); EXTRA_ARGS+=(--batch_size_100 10); [ "$THREADS" -lt 2 ] && THREADS=2 ;;
    7) THREADS=1; EXTRA_ARGS+=(--batch_size_100 5) ;;
    *) THREADS=$CPU_CORES; EXTRA_ARGS+=(--batch_size_100 50) ;;
esac

export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
echo "⚙️ Set Load to ${THREADS} CPU threads."

echo ""
read -p "Do you want to start the pipeline in the BACKGROUND for the PDB in this directory? [Y/n]: " answer
if [[ "$answer" =~ ^[Yy]$ || -z "$answer" ]]; then
    nohup python "$INSTALL_DIR/run_pipeline.py" "${EXTRA_ARGS[@]}" < /dev/null > "$LOG_FILE" 2>&1 &
    BG_PID=$!
    disown "$BG_PID" 2>/dev/null || true
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
