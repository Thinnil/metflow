"""Smart CSV fusion for preserving and merging OpenMM energy checkpoints."""

import math
import os
from typing import Any, Dict
import numpy as np
import pandas as pd


def is_valid_energy(val: Any) -> bool:
    """Return True if val is a finite, non-NaN numeric energy value."""
    if val is None or isinstance(val, bool):
        return False
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
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
        if "Frame" not in df_resume.columns:
            return 0
        ene_col = "Energy_kcal_mol" if "Energy_kcal_mol" in df_resume.columns else "Energy"
        if ene_col not in df_resume.columns:
            return 0

        energy_dict: Dict[int, float] = {}
        for raw_frame, raw_val in zip(df_resume["Frame"], df_resume[ene_col]):
            try:
                frame_id = int(raw_frame)
            except (TypeError, ValueError):
                continue
            if is_valid_energy(raw_val):
                energy_dict[frame_id] = float(raw_val)
            elif frame_id not in energy_dict:
                energy_dict[frame_id] = np.nan

        fused_values = []
        matched_valid = 0
        for f in valid_frames:
            val = energy_dict.get(int(f), np.nan)
            if not math.isnan(val):
                fused_values.append(val)
                matched_valid += 1
            else:
                fused_values.append(np.nan)

        df_data["OpenMM_Energy_kcal_mol"] = fused_values
        print(f"✅ Auto-fused {matched_valid} existing energy evaluations into clustering results!")
        return matched_valid
    except Exception as exc:
        print(f"⚠️ Could not fuse energy checkpoint: {exc}")
        return 0
