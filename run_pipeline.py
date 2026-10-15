import os, sys, glob

# MOCK: Bypass TensorFlow 2.14's strict requirement for Protobuf 4.x runtime_version
import google.protobuf
if not hasattr(google.protobuf, 'runtime_version'):
    class DummyRuntimeVersion:
        Domain = type('Domain', (), {'PUBLIC': 1})
        def ValidateProtobufRuntimeVersion(*args, **kwargs): pass
    google.protobuf.runtime_version = DummyRuntimeVersion
    import sys
    sys.modules['google.protobuf.runtime_version'] = DummyRuntimeVersion

# Apply JAX namespace package monkeypatch BEFORE any other imports!
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
    print("Error: Could not import bioemu. Make sure the conf_sel environment is activated!")
    print(e)
    sys.exit(1)

# Find PDB file
pdb_files = glob.glob("*.pdb")
if not pdb_files:
    print("Error: No .pdb file found in this directory!")
    sys.exit(1)

pdb_file = pdb_files[0]
print(f"Found '{pdb_file}'. Automatically selecting as Monomer PDB Structure.")

parser = PDB.PDBParser(QUIET=True)
structure = parser.get_structure("protein", pdb_file)
sequence = ""
for model in structure:
    for chain in model:
        seq = [seq1(res.resname) for res in chain if PDB.is_aa(res, standard=True)]
        sequence = "".join(seq)
        break
    break

if not sequence:
    print("Error: Could not extract amino acid sequence from PDB!")
    sys.exit(1)

print(f"Extracted Sequence ({len(sequence)} aa): {sequence[:50]}...")

# Parse overriding args
kwargs = {
    "sequence": sequence,
    "num_samples": 2000,
    "output_dir": "./bioemu_ensemble"
}

for i in range(1, len(sys.argv)):
    arg = sys.argv[i]
    if arg.startswith("--") and "=" in arg:
        key, val = arg[2:].split("=", 1)
        kwargs[key] = val
    elif arg.startswith("--") and i + 1 < len(sys.argv) and not sys.argv[i+1].startswith("--"):
        key = arg[2:]
        val = sys.argv[i+1]
        if val.isdigit():
            val = int(val)
        else:
            try:
                val = float(val)
            except ValueError:
                pass
        kwargs[key] = val

# Extract core parameters
num_samples = int(kwargs.pop("num_samples", 2000))
denoising_steps = int(kwargs.pop("denoising_steps", 50))
denoiser_type = kwargs.pop("denoiser_type", "dpm")

solver_target_map = {
    "dpm": "bioemu.shortcuts.dpm_solver",
    "euler": "bioemu.shortcuts.euler_maruyama_solver"
}

kwargs["denoiser_config"] = {
    "_target_": solver_target_map.get(denoiser_type, "bioemu.shortcuts.dpm_solver"),
    "_partial_": True,
    "eps_t": 0.001,
    "max_t": 0.99,
    "N": denoising_steps,
    "noise": 0.0
}
kwargs["denoiser_type"] = denoiser_type
kwargs["num_samples"] = num_samples

print(f"Parameters: samples={num_samples}, steps={denoising_steps}, solver={denoiser_type}")
print("Generating conformations...")

# Write status metadata
status_file = ".metflow_status"
try:
    with open(status_file, "w") as f:
        f.write(f"PID={os.getpid()}\n")
        f.write(f"TARGET_SAMPLES={num_samples}\n")
        if "batch_size_100" in kwargs:
            f.write(f"BATCH_SIZE_100={kwargs['batch_size_100']}\n")
except Exception as e:
    print(f"Warning: Could not write status file: {e}")

# Run BioEmu in-process
try:
    bioemu_main(**kwargs)
finally:
    try:
        if os.path.exists(status_file):
            os.remove(status_file)
    except Exception:
        pass
