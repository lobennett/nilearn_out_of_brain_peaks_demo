# src/nilearn_oob/peaks.py
"""Extract peak coordinates from a t-map, optionally masking first, and flag in/out of brain."""
import numpy as np
from nilearn.image import math_img, resample_to_img
from nilearn.reporting import get_clusters_table

from .brain_mask import is_in_brain


def extract_peaks(tmap, mask_img, ref_mask, stat_threshold=2.0, cluster_threshold=0, min_distance_mm=8.0):
    """Run ``get_clusters_table`` on ``tmap``, optionally masked by ``mask_img``.

    Returns a DataFrame with an added boolean ``in_brain`` column based on ``ref_mask``
    (which is the MNI152 reference, not the analysis mask).
    """
    if mask_img is not None:
        mask_resampled = resample_to_img(mask_img, tmap, interpolation="nearest")
        masked = math_img("img1 * (img2 > 0)", img1=tmap, img2=mask_resampled)
    else:
        masked = tmap
    df = get_clusters_table(
        masked,
        stat_threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        min_distance=min_distance_mm,
    )
    # get_clusters_table emits sub-peaks with NaN cluster IDs; keep all rows
    coords = df[["X", "Y", "Z"]].to_numpy(dtype=float)
    if len(coords):
        flags = is_in_brain(coords, ref_mask)
        # Coerce scalar -> 1-element list
        df["in_brain"] = np.atleast_1d(flags)
    else:
        df["in_brain"] = []
    return df
