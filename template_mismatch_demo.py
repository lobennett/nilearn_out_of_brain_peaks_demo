#!/usr/bin/env python3
"""template_mismatch_demo.py — show why nilearn's default plot_stat_map
backdrop and fMRIPrep's output space disagree at the brain surface.

Both rows overlay the SAME two brain-mask contours on sagittal T1w slices:

    cyan  = nilearn default brain mask   (MNI152 release a)
    red   = fmriprep brain mask          (MNI152NLin2009cAsym, release c,
                                          from templateflow)

Top row: backdrop is the release-a T1w (nilearn's default for plot_stat_map
when bg_img is not passed).
Bottom row: backdrop is the release-c T1w (what fmriprep registers to).

The cyan contour hugs the release-a backdrop; the red contour hugs the
release-c backdrop. Where the two contours diverge is the strip of voxels
that are in-brain for fmriprep but out-of-brain for the default plot.
Real, in-mask peaks that fall in that strip render 'outside the brain'
when you call plot_stat_map(stat_map) without passing bg_img.

Writes figures/template_mismatch_demo.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import templateflow.api as tf
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
from nilearn.plotting import plot_anat


RESOLUTION_MM = 1
CUT_COORDS = [-40, -10, 30]
OUT_PATH = "figures/template_mismatch_demo.png"


def main():
    t1w_a = load_mni152_template(resolution=RESOLUTION_MM)
    mask_a = load_mni152_brain_mask(resolution=RESOLUTION_MM)

    t1w_c = nib.load(
        tf.get(
            "MNI152NLin2009cAsym",
            resolution=RESOLUTION_MM,
            desc="brain",
            suffix="T1w",
        )
    )
    mask_c = nib.load(
        tf.get(
            "MNI152NLin2009cAsym",
            resolution=RESOLUTION_MM,
            desc="brain",
            suffix="mask",
        )
    )

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    disp_a = plot_anat(
        t1w_a,
        display_mode="x",
        cut_coords=CUT_COORDS,
        axes=axes[0],
        title="nilearn default backdrop (release a) — plot_stat_map without bg_img",
        annotate=True,
        draw_cross=False,
        colorbar=False,
    )
    disp_a.add_contours(mask_a, levels=[0.5], colors="cyan", linewidths=2.0)
    disp_a.add_contours(mask_c, levels=[0.5], colors="red", linewidths=2.0)

    disp_c = plot_anat(
        t1w_c,
        display_mode="x",
        cut_coords=CUT_COORDS,
        axes=axes[1],
        title="fmriprep backdrop (MNI152NLin2009cAsym, release c) — your stat_map's space",
        annotate=True,
        draw_cross=False,
        colorbar=False,
    )
    disp_c.add_contours(mask_a, levels=[0.5], colors="cyan", linewidths=2.0)
    disp_c.add_contours(mask_c, levels=[0.5], colors="red", linewidths=2.0)

    fig.suptitle(
        "cyan = nilearn default mask (release a)  |  red = fmriprep mask (release c)\n"
        "Same two contours on both backdrops. Where red extends beyond cyan in the top row, "
        "an in-mask peak there renders 'outside the brain' under the default plot_stat_map backdrop.",
        fontsize=11,
        y=1.02,
    )

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Fix: pass bg_img= explicitly so the backdrop matches your stat_map's space.
#
# Option 1 — templateflow's release-c T1w (group-level reference for fmriprep):
#
#     import templateflow.api as tf
#     from nilearn.plotting import plot_stat_map
#
#     bg = str(tf.get("MNI152NLin2009cAsym", resolution=2,
#                     desc="brain", suffix="T1w"))
#     plot_stat_map(stat_map, bg_img=bg, threshold=2.0, display_mode="x")
#
# Option 2 — the subject's own preproc T1w from fmriprep (best per-subject
# alignment; also masks the backdrop to the subject's brain):
#
#     bg = (f"{fmriprep_dir}/{sub}/{ses}/anat/"
#           f"{sub}_{ses}_..._space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz")
#     plot_stat_map(stat_map, bg_img=bg, threshold=2.0, display_mode="x")
# ---------------------------------------------------------------------------
