# Technical Specification: MET Flow (v2.7 MERGED)

**Document Version**: 2.7.0  
**Target Files**:
- `C:/Users/g_san/.gemini/antigravity/scratch/conf_sel/analyze_trajectory.py`
- `C:/Users/g_san/.gemini/antigravity/scratch/conf_sel/bin/metflow`
- `C:/Users/g_san/.gemini/antigravity/scratch/conf_sel/restore_metflow.sh`
- `C:/Users/g_san/.gemini/antigravity/scratch/conf_sel/sync_heredoc.py`

---

## 1. Executive Summary & Objective

In multi-trajectory comparative mode (e.g., comparing wild-type vs. mutant, apo vs. holo, or BioEmu generative ensembles vs. molecular dynamics trajectories), MET Flow previously performed alignment strictly on C-alpha atom subsets in memory and only exported a C-alpha-only reference structure (`unified_reference.pdb`). No physical merged trajectory (`.xtc`) was ever produced.

The objective of **`v2.7 MERGED`** is to provide an end-to-end physical trajectory fusion and comparative analytics engine:
1. **Physical Unified Trajectory & Reference Export**: Generate an aligned, physically complete merged trajectory (`out_dir/merged_trajectory.xtc` + symlink/alias `out_dir/unified_trajectory.xtc`) and master reference PDB (`out_dir/merged_reference.pdb` + symlink/alias `out_dir/unified_reference.pdb`).
2. **Atom & Sequence Alignment / Non-Protein Stripping**: Filter out solvent, ions, buffer molecules, and heteroatoms; enforce a $\ge 95\%$ sequence identity threshold; calculate consensus homologous protein boundaries; and guarantee 1-to-1 atom name and residue numbering alignment.
3. **Comprehensive Dimensionality Reduction & Origin Tracking**: Perform PCA and clustering directly on the unified trajectory, preserving the `Dataset` origin identifier in all tabular outputs (`global_cluster_metrics.csv`), static figures (`pca_dataset_comparison.png`), and the interactive HTML dashboard (`metflow_dashboard.html`).
4. **Version Consistency (`v2.7 MERGED`)**: Synchronize all version headers across `restore_metflow.sh`, `bin/metflow`, and `analyze_trajectory.py`.
5. **Universal Installer Heredoc Parity**: Maintain 100.0% line and byte parity between standalone scripts and embedded heredocs within `restore_metflow.sh`.

---

## 2. Architectural & Technical Specifications

### Requirement 1: Physical Unified PDB and XTC Export

#### 1.1 File Contract
When comparative analysis is run (single or multi-dataset), the output directory (`out_dir`) will contain:
- `out_dir/merged_reference.pdb`: Clean, renumbered reference PDB containing all common protein heavy atoms of the consensus sequence.
- `out_dir/unified_reference.pdb`: Symlink (or physical copy fallback) pointing to `merged_reference.pdb`.
- `out_dir/merged_trajectory.xtc`: Superimposed, aligned binary XTC trajectory containing the exact coordinates for all analyzed frames.
- `out_dir/unified_trajectory.xtc`: Symlink (or physical copy fallback) pointing to `merged_trajectory.xtc`.

#### 1.2 Coordinate System & Superposition
- All frames written to `merged_trajectory.xtc` are centered and superposed via batched Kabsch alignment onto the consensus reference coordinates.
- Units are verified and scaled to Ångströms (MDAnalysis default).
- The atom count $N_{\text{atoms}}$ in `merged_reference.pdb` strictly equals the number of atoms per frame in `merged_trajectory.xtc`.
- Any external tool (PyMOL, VMD, ChimeraX) loading `merged_reference.pdb` followed by `merged_trajectory.xtc` can immediately step through frames without coordinate displacement or topology mismatches.

#### 1.3 1-to-1 Frame Index Correlation
- Frame index $k \in [0, N_{\text{frames}}-1]$ in `merged_trajectory.xtc` corresponds directly to row $k$ (`Frame_Index == k`) in `global_cluster_metrics.csv`.
- In PyMOL, frame $k$ is accessed by `state = k + 1`.

---

### Requirement 2: Atom & Sequence Alignment / Non-Protein Filtering

#### 2.1 Non-Protein Filtering
Before merging or sequence comparison, every input dataset's topology is inspected and filtered:
- Stripped elements:
  - Water/Solvent: `resname HOH, WAT, TIP3, TIP3P, SOL`
  - Counter-ions & Salts: `resname NA, CL, K, MG, ZN, CA, SOD, CLA, POT`
  - Crystallization agents & buffers: `resname EDO, GOL, PEG, DMS, ACT, TRS, PO4, SO4`
  - Hydrogens: `name H* or type H` (ensuring compatibility between all-atom and heavy-atom ensembles)
  - Heteroatoms: non-protein residues
- Retained elements:
  - Canonical protein amino acids and standard protonation/mutation variants:
    `ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL`
    plus variants `HIE, HID, HIP, HSD, HSE, HSP, CYX, CYM, MSE`.

#### 2.2 Sequence Alignment & Homology Verification ($\ge 95\%$)
- Each dataset's protein residues are converted into canonical 1-letter amino acid sequences.
- For all dataset pairs $(i, j)$, sequence similarity is calculated:
  $$\text{Identity}(s_i, s_j) = \frac{\sum_{b \in \text{matching\_blocks}} \text{length}(b)}{\max(|s_i|, |s_j|)}$$
- If $\text{Identity} < 0.95$, the pipeline halts immediately with a clear diagnostic:
  ```
  ValueError: Sequence similarity across datasets '<name_A>' and '<name_B>' is XX.X%, which is below the 95% threshold.
  ```

#### 2.3 Consensus Homologous Protein Sequence
- Using `difflib.SequenceMatcher`, find the longest common contiguous matching sequence between the master reference (Dataset 0) and each subsequent dataset.
- Calculate the global consensus start and end indices:
  $$\text{consensus\_start} = \max_{k}(\text{ref\_start}_k)$$
  $$\text{consensus\_end} = \min_{k}(\text{ref\_end}_k)$$
  $$\text{consensus\_len} = \text{consensus\_end} - \text{consensus\_start}$$
- Require $\text{consensus\_len} \ge 10$ residues; otherwise raise an error.

#### 2.4 Strict 1-to-1 Atom Name and Residue Mapping
- For each homologous residue position $r \in [0, \text{consensus\_len}-1]$:
  - Intersect the set of heavy atom names across all datasets:
    $$\mathcal{A}_r = \bigcap_{k=1}^D \text{AtomNames}(k, r)$$
  - Sort $\mathcal{A}_r$ in standard PDB backbone-first order (`N`, `CA`, `C`, `O`, `CB`, `CG`, ...).
  - Construct an identical atom template $\mathcal{T} = \bigcup_{r} \mathcal{A}_r$.
- In the exported `merged_reference.pdb`:
  - Residue numbers are sequential: $1, 2, \dots, \text{consensus\_len}$.
  - Atom serial numbers are sequential: $1, 2, \dots, |\mathcal{T}|$.
- In each dataset universe:
  - An `AtomGroup` of exactly $|\mathcal{T}|$ atoms is selected, matching atom names and homologous residue positions.

---

### Requirement 3: PCA, Clustering & Analysis on the Merged Trajectory

#### 3.1 Superposition and Dimensionality Reduction
- C-alpha backbone atoms (or specified `--pca_res`) of the consensus sequence are extracted for all valid frames across all datasets.
- Batched Kabsch algorithm rotates and aligns all conformations to the consensus reference structure.
- PCA is executed:
  - **Combined PCA** (Default): Fits PCA across all datasets simultaneously to capture the shared conformational landscape.
  - **Reference PCA** (`--pca_ref first`): Fits PCA on Dataset 1 (e.g. BioEmu baseline) and projects subsequent datasets into this basis.

#### 3.2 Metadata Preservation
- The column `Dataset` is explicitly retained in:
  - `out_dir/global_cluster_metrics.csv`
  - `out_dir/pca_kmeans/pca_kmeans_metrics.csv`
  - `out_dir/daura/daura_metrics.csv`
  - `out_dir/cnn/cnn_metrics.csv`
  - `out_dir/dataset_cluster_summary.txt`

#### 3.3 Comparative Visualizations & Art Nouveau Dashboard
- `pca_dataset_comparison.png` and `tsne_dataset_comparison.png`: Generated with high-contrast categorical palettes (`Set1` / `tab10`) clearly delineating dataset distributions.
- `metflow_dashboard.html`:
  - Defaults color dropdown to `Dataset`.
  - PyMOL modal code snippet updated for unified trajectory loading:
    ```python
    # Load unified trajectory in PyMOL:
    load merged_reference.pdb
    load_traj merged_trajectory.xtc, merged_reference
    cmd.frame({frame + 1})
    ```

---

### Requirement 4: Version Bump to `v2.7 MERGED`

All banner messages, log headers, and CLI menus are upgraded from `v2.6 PERSISTENT` to `v2.7 MERGED`:
- `restore_metflow.sh`:
  - Banner: `MET Workflow - SMART UNIVERSAL INSTALLER (v2.7 MERGED)`
  - Embedded `analyze_trajectory.py`: `Post-Generation Analysis (v2.7 MERGED)`
  - Embedded `bin/metflow`: `MET Pipeline Launch Configuration (v2.7 MERGED)`
- `bin/metflow`:
  - Banner: `MET Pipeline Launch Configuration (v2.7 MERGED)`
- `analyze_trajectory.py`:
  - Banner: `Post-Generation Analysis (v2.7 MERGED)`

---

### Requirement 5: Universal Installer & Heredoc Parity

- `restore_metflow.sh` embeds the full contents of `analyze_trajectory.py` and `bin/metflow`.
- `sync_heredoc.py` is upgraded to:
  1. Synchronize standalone `analyze_trajectory.py` into the heredoc block `cat << 'EOF' > "$INSTALL_DIR/analyze_trajectory.py"`.
  2. Synchronize standalone `bin/metflow` into the heredoc block `cat << 'EOF' > "$INSTALL_DIR/bin/metflow"`.
  3. Validate 100.0% line-by-line and byte-for-byte exact equality between standalone files and extracted heredocs.

---

## 3. Concrete Implementation Plan & Step-by-Step Edits

### Step 1: Update `analyze_trajectory.py`
- Modify header banner to `(v2.7 MERGED)`.
- Replace loading logic with per-dataset universe initialization (`mda.Universe(pdb_p, xtc_p)`).
- Implement non-protein filtering (`heavy_protein_atoms`).
- Implement 1-to-1 atom name consensus extraction across homologous residues.
- Implement physical export of `merged_reference.pdb` and `merged_trajectory.xtc` (with `unified_*` symlinks/copies).
- Update PyMOL modal helper in HTML template.

### Step 2: Update `bin/metflow`
- Bump version string in menu to `(v2.7 MERGED)`.
- Verify multi-dataset CLI arguments and parameter persistence in `metflow_params.txt`.

### Step 3: Update `sync_heredoc.py`
- Extend synchronization to both `analyze_trajectory.py` and `bin/metflow`.
- Add automated verification asserting zero byte diffs.

### Step 4: Update `restore_metflow.sh`
- Bump version string in installer banner to `(v2.7 MERGED)`.
- Execute `sync_heredoc.py` logic to ensure perfect heredoc embedding.

### Step 5: Verification
- Execute automated parity test asserting exact equality.
- Verify Python syntax via compilation checks.
