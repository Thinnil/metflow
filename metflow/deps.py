"""Dependency verification and installation helper."""

import subprocess
import sys


def install_deps() -> None:
    """Install missing analysis dependencies if needed.

    Only invoked from the CLI/orchestrator entrypoint when running full analysis,
    never at module import time.
    """
    mod_map = {
        "scikit-learn": "sklearn",
        "scipy": "scipy",
        "MDAnalysis": "MDAnalysis",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "setuptools": "pkg_resources",
    }
    pins = {
        "scikit-learn": "scikit-learn==1.4.0",
        "scipy": "scipy==1.12.0",
        "MDAnalysis": "MDAnalysis==2.7.0",
        "pandas": "pandas==2.2.0",
        "matplotlib": "matplotlib==3.8.3",
        "seaborn": "seaborn==0.13.2",
        "setuptools": "setuptools==69.0.0",
    }
    missing = []
    for pkg_name, mod_name in mod_map.items():
        try:
            __import__(mod_name)
        except ImportError:
            missing.append(pins[pkg_name])
    if missing:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir"] + missing,
                check=True,
            )
        except Exception as exc:
            print(f"⚠️ Dependency notice: {exc}")
            print("👉 Please make sure the MetFlow environment is active: micromamba activate conf_sel_gen")
