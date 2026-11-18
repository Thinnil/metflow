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
