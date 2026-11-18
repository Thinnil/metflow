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
