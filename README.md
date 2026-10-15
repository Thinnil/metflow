# MET Flow (Thermodynamic & Conformational Analytics)

**MET Flow** is an integrated pipeline for conformational ensemble generation, thermodynamic potential energy evaluation, and dimensionality reduction analysis across multi-trajectory datasets.

---

## Key Features

- **Consensus Sequence & Coordinate Alignment**: Automatically matches homologous residues across diverse conformational datasets (e.g., BioEmu generative models, unbiased MD with/without ligand), filters non-protein heteroatoms/ions, and generates a unified physical trajectory (`merged_trajectory.xtc` & `merged_reference.pdb`).
- **Thermodynamic Potential Energy (OpenMM Physical Minimization)**:
  - GPU-accelerated (CUDA) implicit solvent energy evaluation (`amber14-all.xml`, `implicit/obc2.xml`).
  - Adaptive clash-relief micro-relaxation for generative structures.
  - Periodic checkpointing (`energy_checkpoint.csv`) with automatic resumption.
- **Dimensionality Reduction & Clustering**:
  - Global C-$\alpha$ PCA, residue loading vectors, and cross-dataset projection.
  - Daura RMSD clustering, PCA k-means, and CNN clustering algorithms.
- **Interactive Art Nouveau Dashboard**:
  - Standalone HTML visualizer (`metflow_dashboard.html`).
  - Dataset filtering dropdown (`bioemu`, `mixed_frames_wo`, `mixed_frames_w`).
  - Discrete quantile interval bins (box-plot style) and smooth continuous gradient coloring.
  - Integrated PyMOL extraction helper snippet for any selected frame.

---

## Installation

Clone the repository and run the self-contained installer:

```bash
git clone https://github.com/Thinnil/metflow.git
cd metflow
chmod +x restore_metflow.sh
./restore_metflow.sh
```

The installer automatically:
1. Detects or installs `micromamba`.
2. Creates and configures the required conda environments (`conf_sel_gen` and `conf_sel_analysis`).
3. Installs all GPU (CUDA) and CPU dependencies (PyTorch, OpenMM, MDAnalysis, PDBFixer, scikit-learn).
4. Deploys the CLI executable `metflow` to `~/.local/bin/metflow` and your system path.

---

## Usage

### Interactive Menu
Run `metflow` from your terminal:
```bash
metflow
```

Options include:
- **[A] Run BioEmu Sampling Pipeline**: Generate conformational ensembles from sequence.
- **[B] Run Skimming on Conformations**: Filter by distance thresholds.
- **[C] Calculate Thermodynamic Energy ONLY**: Direct/minimized OpenMM potential energy evaluation across single or multi-dataset trajectories.
- **[D] Generate Dashboard HTML ONLY**: Re-render the interactive visualizer in ~20 ms.
- **[E] Run Full Analysis Pipeline**: End-to-end PCA, clustering, DCCM heatmaps, and energy calculations.

---

## Repository Structure

```text
metflow/
├── bin/
│   └── metflow               # Main CLI wrapper executable
├── analyze_trajectory.py     # Trajectory analysis, OpenMM energy engine, & dashboard
├── run_pipeline.py           # Generation pipeline orchestrator
├── restore_metflow.sh        # Universal installer and deployment script
├── .gitignore                # Trajectory, checkpoint, and log exclusions
└── README.md                 # Project documentation
```
