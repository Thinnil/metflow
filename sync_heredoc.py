import os

base_dir = os.path.dirname(os.path.abspath(__file__))
py_file = os.path.join(base_dir, "analyze_trajectory.py")
sh_file = os.path.join(base_dir, "restore_metflow.sh")
bin_file = os.path.join(base_dir, "bin", "metflow")


def read_lf(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().replace("\r\n", "\n")
    if not content.endswith("\n"):
        content += "\n"
    return content


def collect_module_files() -> list:
    """Collect scripts/run_pipeline.py and all .py files under metflow/ in deterministic order."""
    files = []
    orch_path = os.path.join(base_dir, "scripts", "run_pipeline.py")
    if os.path.exists(orch_path):
        files.append("scripts/run_pipeline.py")

    metflow_dir = os.path.join(base_dir, "metflow")
    for root, dirs, filenames in os.walk(metflow_dir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for fname in sorted(filenames):
            if fname.endswith(".py"):
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, base_dir).replace("\\", "/")
                files.append(rel)
    return files


py_content = read_lf(py_file)
bin_content = read_lf(bin_file)
sh_content = read_lf(sh_file)

py_start_marker = "cat << 'EOF' > \"$INSTALL_DIR/analyze_trajectory.py\"\n"
bin_start_marker = "cat << 'EOF' > \"$INSTALL_DIR/bin/metflow\"\n"
mod_start_marker = "# METFLOW_MODULES_START\n"
mod_end_marker = "# METFLOW_MODULES_END\n"

# Locate analyze_trajectory.py heredoc boundaries
py_start_idx = sh_content.find(py_start_marker)
if py_start_idx == -1:
    raise ValueError(f"Marker not found: {py_start_marker}")
py_content_start = py_start_idx + len(py_start_marker)
py_end_idx = sh_content.find("\nEOF\n", py_content_start)
if py_end_idx == -1:
    raise ValueError("EOF marker after analyze_trajectory.py heredoc not found")

# Locate module block boundaries
mod_start_idx = sh_content.find(mod_start_marker, py_end_idx)
if mod_start_idx == -1:
    raise ValueError(f"Marker not found: {mod_start_marker}")
mod_content_start = mod_start_idx + len(mod_start_marker)
mod_end_idx = sh_content.find(mod_end_marker, mod_content_start)
if mod_end_idx == -1:
    raise ValueError(f"Marker not found: {mod_end_marker}")

# Locate bin/metflow heredoc boundaries
bin_start_idx = sh_content.find(bin_start_marker, mod_end_idx)
if bin_start_idx == -1:
    raise ValueError(f"Marker not found: {bin_start_marker}")
bin_content_start = bin_start_idx + len(bin_start_marker)
bin_end_idx = sh_content.find("\nEOF\n", bin_content_start)
if bin_end_idx == -1:
    raise ValueError("EOF marker after bin/metflow heredoc not found")

# Build module heredoc block
module_rel_paths = collect_module_files()
module_dirs = sorted({os.path.dirname(p) for p in module_rel_paths if os.path.dirname(p)})
mod_lines = []
for d in module_dirs:
    mod_lines.append(f'mkdir -p "$INSTALL_DIR/{d}"\n')

module_contents = {}
for rel_p in module_rel_paths:
    content = read_lf(os.path.join(base_dir, rel_p))
    module_contents[rel_p] = content
    mod_lines.append(f"cat << 'EOF' > \"$INSTALL_DIR/{rel_p}\"\n")
    mod_lines.append(content)
    mod_lines.append("EOF\n")

mod_block = "".join(mod_lines)

# Assemble updated restore_metflow.sh
prefix = sh_content[:py_content_start]
between_py_and_mod = sh_content[py_end_idx + 1:mod_content_start]
between_mod_and_bin = sh_content[mod_end_idx:bin_content_start]
suffix = sh_content[bin_end_idx + 1:]

new_sh_content = prefix + py_content + between_py_and_mod + mod_block + between_mod_and_bin + bin_content + suffix

with open(sh_file, "w", encoding="utf-8", newline="\n") as f:
    f.write(new_sh_content)

# --- Round-Trip Parity Verification ---
with open(sh_file, "rb") as f:
    sh_bytes = f.read()

assert b"\r" not in sh_bytes, "Error: Carriage returns (CR) detected in restore_metflow.sh! Strict Unix (LF) required."
sh_verified = sh_bytes.decode("utf-8")

# Verify analyze_trajectory.py
rt_py_start = sh_verified.find(py_start_marker) + len(py_start_marker)
rt_py_end = sh_verified.find("\nEOF\n", rt_py_start) + 1
extracted_py = sh_verified[rt_py_start:rt_py_end]
assert extracted_py == py_content, f"Byte parity mismatch in analyze_trajectory.py! ({len(extracted_py)} vs {len(py_content)} bytes)"

# Verify every module file
for rel_p, expected_content in module_contents.items():
    marker = f"cat << 'EOF' > \"$INSTALL_DIR/{rel_p}\"\n"
    m_start = sh_verified.find(marker)
    assert m_start != -1, f"Missing module heredoc in restore_metflow.sh: {rel_p}"
    c_start = m_start + len(marker)
    c_end = sh_verified.find("\nEOF\n", c_start) + 1
    extracted = sh_verified[c_start:c_end]
    assert extracted == expected_content, f"Byte parity mismatch in {rel_p}! ({len(extracted)} vs {len(expected_content)} bytes)"

# Verify bin/metflow
rt_bin_start = sh_verified.find(bin_start_marker) + len(bin_start_marker)
rt_bin_end = sh_verified.find("\nEOF\n", rt_bin_start) + 1
extracted_bin = sh_verified[rt_bin_start:rt_bin_end]
assert extracted_bin == bin_content, f"Byte parity mismatch in bin/metflow! ({len(extracted_bin)} vs {len(bin_content)} bytes)"

print("[SUCCESS] Multi-Module Heredoc Synchronization Successful!")
print(f"   - analyze_trajectory.py: {len(py_content)} bytes (100.0% exact parity)")
print(f"   - bin/metflow:           {len(bin_content)} bytes (100.0% exact parity)")
print(f"   - bundled modules:       {len(module_contents)} files (100.0% exact parity)")
print("   - Strict Unix line endings (LF only): VERIFIED")
