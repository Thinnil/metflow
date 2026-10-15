import os

base_dir = os.path.dirname(os.path.abspath(__file__))
py_file = os.path.join(base_dir, 'analyze_trajectory.py')
sh_file = os.path.join(base_dir, 'restore_metflow.sh')
bin_file = os.path.join(base_dir, 'bin', 'metflow')

with open(py_file, 'r', encoding='utf-8') as f:
    py_content = f.read().replace('\r\n', '\n')

with open(bin_file, 'r', encoding='utf-8') as f:
    bin_content = f.read().replace('\r\n', '\n')

with open(sh_file, 'r', encoding='utf-8') as f:
    sh_content = f.read().replace('\r\n', '\n')

if not py_content.endswith('\n'):
    py_content += '\n'

if not bin_content.endswith('\n'):
    bin_content += '\n'

py_start_marker = "cat << 'EOF' > \"$INSTALL_DIR/analyze_trajectory.py\"\n"
bin_start_marker = "cat << 'EOF' > \"$INSTALL_DIR/bin/metflow\"\n"

# Locate analyze_trajectory.py heredoc boundaries
py_start_idx = sh_content.find(py_start_marker)
if py_start_idx == -1:
    raise ValueError(f"Marker not found: {py_start_marker}")
py_content_start = py_start_idx + len(py_start_marker)

py_end_idx = sh_content.find("\nEOF\n", py_content_start)
if py_end_idx == -1:
    raise ValueError("EOF marker after analyze_trajectory.py heredoc not found")

# Locate bin/metflow heredoc boundaries
bin_start_idx = sh_content.find(bin_start_marker, py_end_idx)
if bin_start_idx == -1:
    raise ValueError(f"Marker not found: {bin_start_marker}")
bin_content_start = bin_start_idx + len(bin_start_marker)

bin_end_idx = sh_content.find("\nEOF\n", bin_content_start)
if bin_end_idx == -1:
    raise ValueError("EOF marker after bin/metflow heredoc not found")

# Extract segments
prefix = sh_content[:py_content_start]
middle = sh_content[py_end_idx + 1 : bin_content_start]  # starts with 'EOF\n...' and ends with bin_start_marker
suffix = sh_content[bin_end_idx + 1 :]                   # starts with 'EOF\n...' and contains remainder of installer

# Assemble updated restore_metflow.sh
new_sh_content = prefix + py_content + middle + bin_content + suffix

# Write with strict Unix \n line endings
with open(sh_file, 'w', encoding='utf-8', newline='\n') as f:
    f.write(new_sh_content)

# --- Round-Trip Parity Verification ---
with open(sh_file, 'rb') as f:
    sh_bytes = f.read()

assert b'\r' not in sh_bytes, "Error: Carriage returns (CR) detected in restore_metflow.sh! Strict Unix (LF) required."

sh_verified = sh_bytes.decode('utf-8')

# Verify analyze_trajectory.py heredoc exact byte-for-byte parity
rt_py_start = sh_verified.find(py_start_marker) + len(py_start_marker)
rt_py_end = sh_verified.find("\nEOF\n", rt_py_start) + 1
extracted_py = sh_verified[rt_py_start:rt_py_end]
assert extracted_py == py_content, f"Byte parity mismatch in analyze_trajectory.py! ({len(extracted_py)} vs {len(py_content)} bytes)"

# Verify bin/metflow heredoc exact byte-for-byte parity
rt_bin_start = sh_verified.find(bin_start_marker) + len(bin_start_marker)
rt_bin_end = sh_verified.find("\nEOF\n", rt_bin_start) + 1
extracted_bin = sh_verified[rt_bin_start:rt_bin_end]
assert extracted_bin == bin_content, f"Byte parity mismatch in bin/metflow! ({len(extracted_bin)} vs {len(bin_content)} bytes)"

print("[SUCCESS] Dual Heredoc Synchronization Successful!")
print(f"   - analyze_trajectory.py: {len(py_content)} bytes (100.0% exact parity)")
print(f"   - bin/metflow:           {len(bin_content)} bytes (100.0% exact parity)")
print("   - Strict Unix line endings (LF only): VERIFIED")
