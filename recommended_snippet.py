"""Recommended pattern for in-brain peak extraction with nilearn.

Drop-in replacement for the pipeline that produced out-of-brain peaks.

Key changes:
  1. Build the analysis mask with ``intersect_masks(threshold=1.0)`` — strict
     intersection across runs. The default of 0.5 is too permissive when you
     have 3 or more runs (and is functionally identical to 1.0 when you have
     exactly 2 runs — see the README for details).
  2. Pass the same mask to ``FirstLevelModel(mask_img=...)``.
  3. Multiply the contrast t-map by the mask BEFORE ``get_clusters_table`` —
     the function does not accept a mask argument, so the caller must mask
     the input image themselves.
"""
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import math_img
from nilearn.masking import compute_epi_mask, intersect_masks
from nilearn.reporting import get_clusters_table


def fit_and_report(bold_files, event_files, t_r, contrast):
    # 1) Per-run masks (use fmriprep's brain mask in production, not compute_epi_mask)
    run_masks = [compute_epi_mask(b) for b in bold_files]

    # 2) Strict intersection — every voxel must be in every run's mask
    mask_img = intersect_masks(run_masks, threshold=1.0)

    # 3) Fit with the explicit mask
    events = [pd.read_csv(e, sep="\t") for e in event_files]
    model = FirstLevelModel(t_r=t_r, hrf_model="spm", mask_img=mask_img,
                            noise_model="ar1", standardize=False)
    model = model.fit(bold_files, events=events)

    t_map = model.compute_contrast(contrast, output_type="stat")

    # 4) MASK BEFORE PEAK EXTRACTION — this is the fix
    t_map_masked = math_img("img1 * (img2 > 0)", img1=t_map, img2=mask_img)
    return get_clusters_table(t_map_masked, stat_threshold=2.0,
                              cluster_threshold=0, min_distance=8.0)
