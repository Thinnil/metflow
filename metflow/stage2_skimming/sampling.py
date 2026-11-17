"""Per-dataset uniform strided downsampling."""

from typing import List, Tuple
import numpy as np


def downsample_per_dataset(
    valid_frames: np.ndarray,
    dataset_tags: List[str],
    max_frames: int,
) -> Tuple[np.ndarray, List[str], str]:
    """Uniformly sample up to max_frames per dataset, preserving order."""
    unique_tags = list(dict.fromkeys(dataset_tags))
    if max_frames > 0:
        dataset_tags_arr = np.array(dataset_tags)[valid_frames]
        sampled_parts = []
        for t in unique_tags:
            t_idxs = valid_frames[dataset_tags_arr == t]
            if len(t_idxs) > max_frames:
                st = max(1, len(t_idxs) // max_frames)
                sampled_parts.append(t_idxs[::st][:max_frames])
            else:
                sampled_parts.append(t_idxs)
        sampled_frames = np.sort(np.concatenate(sampled_parts)) if sampled_parts else np.array([], dtype=int)
        msg = (
            f"Per-dataset sampling: up to {max_frames} frames per dataset "
            f"({len(sampled_frames)} total)."
        )
        print(
            f"\n[Per-Dataset Sampling] Uniformly sampled up to {max_frames} frames per dataset "
            f"({len(sampled_frames)} total frames across {len(unique_tags)} dataset(s))."
        )
    else:
        sampled_frames = valid_frames
        msg = f"No downsampling applied: 100% of frames retained ({len(sampled_frames)} total)."
        print(
            f"\n[No Downsampling] Preserved 100% of frames ({len(sampled_frames)} total frames "
            f"across {len(unique_tags)} dataset(s))."
        )

    filtered_tags = [dataset_tags[i] for i in sampled_frames]
    return sampled_frames, filtered_tags, msg
