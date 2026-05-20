"""Figures for the README:

- ``plot_will_style``: 3 rows (A/B/C), each a horizontal strip of sagittal
  slices in the same style as the colleague's _visualization.pdf files.
  Direct visual analog to the contrast maps they're producing.
- ``plot_smoking_gun``: A vs B vs C at the worst NATURAL (non-synthetic)
  out-of-brain peak from variant A, with the strict-intersection mask drawn
  as a red contour so "outside the brain" is visible.
- ``plot_peak_scatter``: 3-row x 3-col grid (rows = variants, cols = three
  orthogonal projections) with the brain-mask projection as a gray backdrop
  in every panel. OOB peaks plotted as red X, in-brain peaks as colored dots.
- ``plot_masks``: both analysis masks overlaid as contours of different
  colors on the mean-BOLD anatomical backdrop.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nilearn.image import math_img, mean_img, resample_to_img
from nilearn.plotting import plot_anat, plot_stat_map


_BG_IMG_CACHE = None


def _displayed_tmap(variant_data):
    """Return the t-map a viewer would see after the variant's analysis mask
    is applied. For variant A (no mask) this is the raw t-map; for B and C
    the t-map is multiplied by the analysis mask so the figure matches what
    ``get_clusters_table`` ultimately receives.
    """
    tmap = variant_data["tmap"]
    mask = variant_data["mask"]
    if mask is None:
        return tmap
    mask_resampled = resample_to_img(mask, tmap, interpolation="nearest")
    return math_img("img1 * (img2 > 0)", img1=tmap, img2=mask_resampled)


def _backdrop():
    """Mean of run-1 BOLD, used as a subject-space anatomical backdrop.

    Nilearn's default backdrop is the MNI152 T1 in MNI space, but this
    dataset is in subject space — using the MNI default makes contours and
    t-maps appear displaced from the visible brain. The mean BOLD shares
    the t-map's affine so everything aligns. Cached at module level so
    repeated figure calls don't re-load the dataset.
    """
    global _BG_IMG_CACHE
    if _BG_IMG_CACHE is None:
        from .data import load_dataset
        _BG_IMG_CACHE = mean_img(load_dataset().bold_imgs[0])
    return _BG_IMG_CACHE


def _natural_oob_cut(df_a, margin_mm=25.0):
    """Return MNI mm of the highest-t out-of-brain peak that lies within
    ``margin_mm`` of the bounding box of the in-brain peaks.

    This filters out the synthetic injection placed at the image bounding-box
    corner (which is far from any brain content and produces empty plots).
    Falls back to (0, 0, 0) if no natural OOB peak exists.
    """
    in_brain = df_a[df_a["in_brain"]]
    if len(in_brain) == 0:
        return (0.0, 0.0, 0.0)
    bounds = {
        "X": (in_brain["X"].min() - margin_mm, in_brain["X"].max() + margin_mm),
        "Y": (in_brain["Y"].min() - margin_mm, in_brain["Y"].max() + margin_mm),
        "Z": (in_brain["Z"].min() - margin_mm, in_brain["Z"].max() + margin_mm),
    }
    oob_sorted = df_a[~df_a["in_brain"]].sort_values("Peak Stat", ascending=False)
    for _, row in oob_sorted.iterrows():
        if all(bounds[ax][0] <= row[ax] <= bounds[ax][1] for ax in ("X", "Y", "Z")):
            return float(row["X"]), float(row["Y"]), float(row["Z"])
    return (0.0, 0.0, 0.0)


def plot_will_style(variants, out_path):
    """Horizontal sagittal-slice montage per variant, matching the layout of
    Will's ``_visualization.pdf`` files (one row of cuts per stat map, default
    diverging colormap, threshold = 2). Intended as a direct visual analog
    to the contrast maps Will is generating, but rendered for each of the
    three masking variants so the effect of masking on a familiar layout is
    apparent.
    """
    bg = _backdrop()
    # Cuts span the FOV (not just the brain) so the synthetic ghosts at the
    # lateral FOV edges (~X=±85 mm) are visible alongside the brain interior.
    cut_coords = [-85, -54, -27, 0, 27, 54, 85]

    titles = {
        "A": "Variant A: no mask -- faces > scrambled",
        "B": "Variant B: union mask (intersect_masks threshold=0.0)",
        "C": "Variant C: strict intersection mask (threshold=1.0)",
    }

    fig, axes = plt.subplots(3, 1, figsize=(15, 7))
    for ax, label in zip(axes, ("A", "B", "C")):
        plot_stat_map(
            _displayed_tmap(variants[label]),
            bg_img=bg,
            display_mode="x",
            cut_coords=cut_coords,
            threshold=2.0,
            axes=ax,
            title=titles[label],
            colorbar=True,
            symmetric_cbar=True,
            cmap="cold_hot",
        )
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_smoking_gun(variants, out_path):
    """Three rows (A, B, C) at the worst natural OOB peak from variant A.

    The strict-intersection mask (variant C's mask) is overlaid as a red
    contour in every panel so the brain boundary is visible. Variant A
    should show a hot spot (`t > 2`) outside the red contour; variant C
    should not.
    """
    cut_coords = _natural_oob_cut(variants["A"]["peaks"])
    coord_label = tuple(round(c) for c in cut_coords)

    titles = {
        "A": "Variant A: no mask anywhere (bug)",
        "B": "Variant B: intersect_masks(threshold=0.0), union",
        "C": "Variant C: intersect_masks(threshold=1.0), strict intersection (fix)",
    }

    bg = _backdrop()
    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    for ax, label in zip(axes, ("A", "B", "C")):
        display = plot_stat_map(
            _displayed_tmap(variants[label]),
            bg_img=bg,
            cut_coords=cut_coords,
            threshold=2.0,
            display_mode="ortho",
            axes=ax,
            title=titles[label],
            colorbar=True,
        )
        display.add_contours(variants["C"]["mask"], colors="r", linewidths=1.2)

    fig.suptitle(
        f"Cut at variant-A out-of-brain peak {coord_label} mm. "
        "Red contour = strict-intersection (recommended) mask boundary.",
        y=1.00,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_peak_scatter(variants, out_path):
    """3 rows (variants) x 3 columns (orthogonal projections) of peak coords.

    Each panel has the brain-mask voxels projected as a faint gray hexbin
    backdrop so the in/out distinction is visually obvious.
    """
    mask = variants["C"]["mask"].get_fdata() > 0
    affine = variants["C"]["mask"].affine
    ijk = np.array(np.where(mask)).T
    mm = (affine @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]
    mm_axes = {"X": 0, "Y": 1, "Z": 2}

    plane_pairs = [
        ("X", "Y", "axial (X vs Y)"),
        ("X", "Z", "coronal (X vs Z)"),
        ("Y", "Z", "sagittal (Y vs Z)"),
    ]
    variant_colors = {"A": "C0", "B": "C1", "C": "C2"}

    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    for row_idx, label in enumerate(("A", "B", "C")):
        df = variants[label]["peaks"]
        inside = df[df["in_brain"]]
        outside = df[~df["in_brain"]]
        for col_idx, (a, b, title) in enumerate(plane_pairs):
            ax = axes[row_idx, col_idx]
            ax.hexbin(mm[:, mm_axes[a]], mm[:, mm_axes[b]],
                      cmap="Greys", gridsize=50, alpha=0.4, mincnt=1)
            ax.scatter(inside[a], inside[b], c=variant_colors[label],
                       marker="o", s=12, alpha=0.55, label="in-brain")
            ax.scatter(outside[a], outside[b], c="red", marker="x",
                       s=70, linewidths=2, label="OUT-of-brain")
            if row_idx == 0:
                ax.set_title(title)
            ax.set_xlabel(f"{a} (mm)")
            if col_idx == 0:
                ax.set_ylabel(f"Variant {label}\n{b} (mm)")
            else:
                ax.set_ylabel(f"{b} (mm)")
            ax.axhline(0, color="gray", lw=0.3)
            ax.axvline(0, color="gray", lw=0.3)
    axes[0, -1].legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Peak coordinates per variant. "
        "Gray backdrop = strict-intersection brain mask; red X = out-of-brain.",
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_masks(variants, out_path):
    """Overlay both analysis masks (union vs strict intersection) as contours
    of different colors on the mean-BOLD anatomical backdrop. The rim between
    them is the difference the threshold parameter buys you.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    display = plot_anat(
        _backdrop(), axes=ax, display_mode="z", cut_coords=5,
        title="Analysis masks: blue = union (threshold=0.0); red = strict intersection (threshold=1.0)",
    )
    display.add_contours(variants["B"]["mask"], colors="b", linewidths=1.5)
    display.add_contours(variants["C"]["mask"], colors="r", linewidths=1.5)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
