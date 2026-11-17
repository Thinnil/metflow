"""Smart CSV fusion for preserving and merging OpenMM energy checkpoints."""

import os
from typing import Any, Dict
import numpy as np
import pandas as pd


def is_valid_energy(val: Any) -> bool:
    """Return True if val is a finite, non-NaN numeric energy value."""
    if pd.isna(val):
        return False
    try:
        f = float(val)
        return not (np.isnan(f) or np.isinf(f))
    except Exception:
        return False


def fuse_energy_csv(
    df_data: Dict[str, Any],
    valid_frames: np.ndarray,
    energy_csv: str,
) -> int:
    """Fuse existing energy_checkpoint.csv into df_data if not already populated."""
    if "OpenMM_Energy_kcal_mol" in df_data or not os.path.exists(energy_csv):
        return 0
    try:
        df_resume = pd.read_csv(energy_csv)
        ene_col = "Energy_kcal_mol" if "Energy_kcal_mol" in df_resume.columns else "Energy"
        energy_dict = dict(zip(df_resume["Frame"], df_resume[ene_col]))
        df_data["OpenMM_Energy_kcal_mol"] = [energy_dict.get(f, np.nan) for f in valid_frames]
        print(f"✅ Auto-fused {len(energy_dict)} existing energy evaluations into clustering results!")
        return len(energy_dict)
    except Exception as exc:
        print(f"⚠️ Could not fuse energy checkpoint: {exc}")
        return 0
