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
