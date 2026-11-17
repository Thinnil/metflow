"""Text reports (cluster summary, methodology) and Stage 6 orchestration."""

import os
from typing import List
import pandas as pd

from metflow.models import StageResult, TrajectoryContext
from metflow.stage6_visualization.dashboard import generate_dashboard
from metflow.stage6_visualization.plots import (
    generate_advanced_plots,
    generate_cluster_plots,
    generate_comparison_plots,
    generate_pca_loadings,
)


def write_cluster_summary(df: pd.DataFrame, out_dir: str, unique_datasets: List[str]) -> str:
    """Write dataset_cluster_summary.txt with per-dataset counts and cross-tabulations."""
    summary_lines = [
        "=== Multi-Trajectory Dataset Cluster Distribution Summary ===",
        f"Total Datasets: {len(unique_datasets)}",
        f"Total Frames: {len(df)}\n",
    ]
    for ds in unique_datasets:
        ds_count = len(df[df["Dataset"] == ds])
        summary_lines.append(f"Dataset '{ds}': {ds_count} frames ({ds_count / len(df) * 100:.1f}%)")

    for cluster_col in ["PCA_KMeans_Cluster", "Daura_Cluster", "CNN_Cluster"]:
        if cluster_col in df.columns:
            summary_lines.append(f"\n--- Cross-Tabulation: {cluster_col} ---")
            ct = pd.crosstab(df[cluster_col], df["Dataset"], margins=True)
            ct_pct = pd.crosstab(df[cluster_col], df["Dataset"], normalize="index") * 100
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
    """Write methodology.txt."""
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

    cmp_artifacts, unique_datasets, palette = generate_comparison_plots(df, out_dir)
    artifacts.extend(cmp_artifacts)

    if config.is_only_energy:
        artifacts.append(write_methodology(ctx.methodology, out_dir))
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

    artifacts.append(write_cluster_summary(df, out_dir, unique_datasets))
    artifacts.extend(generate_cluster_plots(df, out_dir, var_explained))

    if config.is_advanced_plots:
        artifacts.extend(
            generate_advanced_plots(
                df, ctx.X, ctx.rmsd_matrix, out_dir, unique_datasets, palette, ctx.methodology
            )
        )

    artifacts.extend(generate_pca_loadings(ctx.pca_model, ctx.ref_bb_atoms, out_dir))
    artifacts.append(write_methodology(ctx.methodology, out_dir))

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
