#!/usr/bin/env python3
"""test_template_mismatch.py — quantify the MNI152 release-a vs release-c mismatch.

``nilearn.datasets.load_mni152_template()`` returns the MNI152 ICBM 152
nonlinear 2009 *asymmetric release a* T1w template [1]. fMRIPrep normalizes
BOLD data into the ``MNI152NLin2009cAsym`` (*release c*) space — a different,
though related, normalization with a slightly different brain surface.

When ``plot_stat_map(stat_map)`` is called with the default ``bg_img=None``,
nilearn uses its release-a template as the backdrop [2]. If your stat_map is
in release-c space (the fMRIPrep output), the two brain outlines do not align
exactly — voxels can sit inside your release-c analysis mask but outside the
release-a silhouette the plot draws. The visible result is "floating dots"
that aren't real signal-outside-brain, just a backdrop mismatch.

This script:

  1. Loads nilearn's default (release a) and templateflow's MNI152NLin2009cAsym
     (release c) T1w templates side by side.
  2. Generates a 3-row sagittal comparison of a single stat_map rendered
     against (a) the nilearn default, (b) the templateflow release-c T1w,
     and (c) the subject's own MNI-normalized T1w. All three rows use the
     same ``cut_coords``, ``threshold``, and ``vmax``, so the only thing
     changing is the backdrop.
  3. Resamples both templates to the stat_map grid, binarizes each at a
     fraction of its 99th-percentile intensity (robust against single high
     voxels), and reports the symmetric volume difference in mm³. Then
     counts how many suprathreshold stat voxels fall inside the fMRIPrep
     subject brain mask but OUTSIDE the binarized release-a silhouette —
     i.e. the exact set of "floating" voxels that the rendering produces.

References (for the docstring rather than runtime):
  [1] https://nilearn.github.io/dev/modules/generated/nilearn.datasets.load_mni152_template.html
      load_mni152_template — release a, MNI ICBM 152 nonlinear asymmetric
  [2] https://nilearn.github.io/dev/modules/generated/nilearn.plotting.plot_stat_map.html
      plot_stat_map — bg_img default is nilearn's MNI152 template
  [3] https://www.templateflow.org/python-client/ — templateflow API

Outputs:
  - outputs/template_mismatch_demo.png — 3-row sagittal comparison figure
  - stdout summary with voxel counts and mm³ deltas

Install templateflow if you don't have it:
  uv add templateflow      # if inside a uv project
  pip install templateflow # otherwise
"""

# ======================== CONFIG: EDIT FOR YOUR DATA ========================

# z- or t-stat NIfTI from your nilearn pipeline (any single subject, single contrast).
STAT_MAP_PATH = "/REPLACE/with/sub-1012_narrative-probes_z_map.nii.gz"

# That same subject's fMRIPrep brain mask (space-MNI152NLin2009cAsym_desc-brain_mask).
SUBJECT_MASK_PATH = (
    "/REPLACE/with/sub-1012_..._space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
)

# That subject's MNI-normalized T1w (space-MNI152NLin2009cAsym_desc-preproc_T1w).
SUBJECT_T1W_PATH = (
    "/REPLACE/with/sub-1012_..._space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
)

# Sagittal cuts for the comparison figure.
CUT_COORDS = [-40, -20, 0, 20, 40]

# Threshold (|t| or |z|) applied to the stat overlay in the figure.
STAT_THRESHOLD = 2.0

# Color-scale upper bound; identical across all 3 panels.
STAT_VMAX = 8.0

# Templateflow query for the release-c brain T1w.
TFLOW_TEMPLATE = "MNI152NLin2009cAsym"
TFLOW_RESOLUTION = 1
TFLOW_DESC = "brain"
TFLOW_SUFFIX = "T1w"

# Intensity-binarization threshold for each template: frac * 99th-percentile.
# 0.10 gives a generous "brain region" silhouette that includes both white
# and gray matter while excluding near-zero background.
BINARIZE_FRAC = 0.10

OUT_DIR = "outputs"

# ====================== END CONFIG (don't edit below) =======================

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import templateflow.api as tf
from nilearn.datasets import load_mni152_template
from nilearn.image import math_img, resample_to_img
from nilearn.plotting import plot_stat_map


def _voxel_volume_mm3(img):
    """Volume of one voxel of ``img`` in mm³ (works for any affine)."""
    return float(np.abs(np.linalg.det(img.affine[:3, :3])))


def _binarize_at_p99(img, frac):
    """Binarize an intensity image at ``frac * 99th-percentile`` of its values.

    Using the 99th percentile (not max) keeps a single high voxel from
    dragging the threshold up.
    """
    arr = img.get_fdata()
    p99 = float(np.percentile(arr, 99))
    threshold = frac * p99
    return math_img(f"img > {threshold}", img=img)


def _three_panel_figure(stat_map, panels, out_path):
    """Render three vertically-stacked sagittal strips with matched cuts/threshold/vmax."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 9))
    for ax, (bg, title) in zip(axes, panels):
        plot_stat_map(
            stat_map,
            bg_img=bg,
            display_mode="x",
            cut_coords=CUT_COORDS,
            threshold=STAT_THRESHOLD,
            vmax=STAT_VMAX,
            axes=ax,
            title=title,
            colorbar=True,
            symmetric_cbar=True,
        )
    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight", dpi=120)
    plt.close(fig)


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(exist_ok=True)

    # --- (1) Load templates and subject data ----------------------------------
    print("Loading templates and subject data...")

    nilearn_default = load_mni152_template()
    print(f"  nilearn default (release a):      shape={nilearn_default.shape}, "
          f"voxel={_voxel_volume_mm3(nilearn_default):.3f} mm^3")

    tflow_path = tf.get(
        TFLOW_TEMPLATE,
        resolution=TFLOW_RESOLUTION,
        desc=TFLOW_DESC,
        suffix=TFLOW_SUFFIX,
    )
    templateflow_c = nib.load(tflow_path)
    print(f"  templateflow {TFLOW_TEMPLATE} (release c): "
          f"shape={templateflow_c.shape}, "
          f"voxel={_voxel_volume_mm3(templateflow_c):.3f} mm^3")
    print(f"    (path: {tflow_path})")

    stat_map = nib.load(STAT_MAP_PATH)
    subject_mask = nib.load(SUBJECT_MASK_PATH)
    subject_t1w = nib.load(SUBJECT_T1W_PATH)
    print(f"  stat_map:                         shape={stat_map.shape}, "
          f"voxel={_voxel_volume_mm3(stat_map):.3f} mm^3")

    # --- (2) Three-panel comparison figure ------------------------------------
    print("Rendering 3-panel sagittal comparison...")
    panels = [
        (None,
         "(a) nilearn default backdrop  —  MNI152 ICBM 152 release a"),
        (templateflow_c,
         "(b) templateflow MNI152NLin2009cAsym (release c)  —  fmriprep output space"),
        (subject_t1w,
         "(c) subject's own MNI-normalized T1w  —  from fmriprep"),
    ]
    fig_path = out_dir / "template_mismatch_demo.png"
    _three_panel_figure(stat_map, panels, fig_path)
    print(f"  wrote {fig_path}")

    # --- (3) Quantify mismatch -----------------------------------------------
    print("Quantifying release-a vs release-c silhouette mismatch...")

    # Resample both templates to the stat_map grid (linear: continuous intensities).
    nl_resampled = resample_to_img(nilearn_default, stat_map, interpolation="linear")
    tf_resampled = resample_to_img(templateflow_c, stat_map, interpolation="linear")

    # Binarize each at frac * 99th-percentile intensity.
    nl_bin = _binarize_at_p99(nl_resampled, BINARIZE_FRAC)
    tf_bin = _binarize_at_p99(tf_resampled, BINARIZE_FRAC)
    nl_arr = nl_bin.get_fdata() > 0
    tf_arr = tf_bin.get_fdata() > 0

    voxel_mm3 = _voxel_volume_mm3(stat_map)
    nl_count = int(nl_arr.sum())
    tf_count = int(tf_arr.sum())
    sym_diff = int((nl_arr ^ tf_arr).sum())
    only_in_c = int((tf_arr & ~nl_arr).sum())
    only_in_a = int((nl_arr & ~tf_arr).sum())

    # Suprathreshold stat voxels that fall in the subject's brain mask
    # but OUTSIDE the nilearn-default silhouette — i.e., the floaters that
    # plot_stat_map(bg_img=None) renders.
    stat_data = stat_map.get_fdata()
    supra = np.abs(stat_data) > STAT_THRESHOLD
    n_supra = int(supra.sum())

    subj_mask_resampled = resample_to_img(
        subject_mask, stat_map, interpolation="nearest"
    )
    subj_arr = subj_mask_resampled.get_fdata() > 0
    floating = supra & subj_arr & ~nl_arr
    n_floating = int(floating.sum())

    # --- (4) Print summary ----------------------------------------------------
    print()
    print("=" * 70)
    print("BINARIZED SILHOUETTE VOXEL COUNTS (resampled to stat_map grid;")
    print(f"binarized at {BINARIZE_FRAC * 100:.0f}% of each template's 99th percentile):")
    print(f"  nilearn default (release a):       {nl_count:>8} vox  "
          f"= {nl_count * voxel_mm3:>10.0f} mm^3")
    print(f"  templateflow MNI...cAsym (rel c):  {tf_count:>8} vox  "
          f"= {tf_count * voxel_mm3:>10.0f} mm^3")
    print()
    print("SYMMETRIC DIFFERENCE:")
    print(f"  voxels in c but NOT in a:          {only_in_c:>8} vox  "
          f"= {only_in_c * voxel_mm3:>10.0f} mm^3")
    print(f"  voxels in a but NOT in c:          {only_in_a:>8} vox  "
          f"= {only_in_a * voxel_mm3:>10.0f} mm^3")
    print(f"  total symmetric difference:        {sym_diff:>8} vox  "
          f"= {sym_diff * voxel_mm3:>10.0f} mm^3")
    print()
    print("FLOATING-VOXEL POPULATION (suprathreshold AND inside fmriprep")
    print("subject brain mask AND outside the nilearn-default silhouette):")
    print(f"  total suprathreshold voxels (|t| > {STAT_THRESHOLD}): {n_supra}")
    print(f"  of which 'floating' under the nilearn-default backdrop: {n_floating}")
    if n_supra > 0:
        print(f"  -> {n_floating / n_supra * 100:.2f}% of suprathreshold voxels "
              f"would appear to sit outside the brain in a default plot")
    print("=" * 70)


if __name__ == "__main__":
    main()
