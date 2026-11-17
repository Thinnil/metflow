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
