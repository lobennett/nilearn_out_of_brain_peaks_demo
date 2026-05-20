#!/usr/bin/env python3
"""debug_mask.py — diagnose floating-voxel artifacts in nilearn first-level GLM.

Edit the CONFIG block below to point at your data, then run:

    python debug_mask.py

Outputs land in ./debug_mask_out/:

  - 01_mask_on_stat_map.png ........ current mask overlaid on stat_map (#1)
  - 02_mask_on_template.png ........ current mask overlaid on MNI152 anatomy (#2)
  - 03_run_NN_mask.png ............. each per-run mask separately (#3)
  - 04_intersected_mask.png ........ intersect_masks(threshold=1.0) result (#4)
  - 08_stat_map_unthresholded.png .. unthresholded stat_map sanity check (#8)
  - 09_stat_map_continuous.png ..... thresholded plot with cubic interpolation (#9)
  - 09_stat_map_nearest.png ........ thresholded plot with nearest interpolation (#9)
  - summary.txt .................... voxel counts, masker PASS/FAIL, breakdown (#5-#7)

THE HEADLINE DIAGNOSTIC is #7 (in summary.txt). For every voxel above
|t|>STAT_THRESHOLD it counts whether the voxel falls inside the current
[0] mask, the strict-intersection mask, the MNI152 template brain mask,
or outside all three. It ALSO reports the distance from the template
brain surface for the outside-all-three voxels, binned 0-2 / 2-5 / 5-10
/ 10-20 / 20+ mm. The shape of that distribution distinguishes:

  - (d) ≈ 0 in raw data but visible floaters in plots -> plot_stat_map
    rendering artifact (cubic-spline bleed across the mask boundary);
    compare #9_continuous vs #9_nearest to confirm.
  - mostly 0-2 mm out  -> analysis-vs-template rim mismatch (cosmetic)
  - concentrated 2-5 mm out -> draining veins / dural sinus signal
  - substantial fraction 5+ mm out -> registration error or multiband
    slice-leakage ghosts (NOT cosmetic — real artifact)

Provide the fitted FirstLevelModel via FLM_JOBLIB (saved with
``joblib.dump(first_level_model, "flm.joblib")``) to enable the masker
consistency check (#5). If not provided, #5 is skipped.
"""

# ======================== CONFIG: EDIT FOR YOUR DATA ========================

BASE_DIR = "/REPLACE/with/your/derivatives/path"
SUB = "sub-1012"
SES = "ses-1"
TASK = "empath"

# Full path to the t-statistic image from FirstLevelModel.compute_contrast.
STAT_MAP_PATH = "/REPLACE/with/your/contrast_t_map.nii.gz"

# Any one preprocessed BOLD path (for the affine check, #6).
BOLD_PATH = "/REPLACE/with/one/preproc_bold.nii.gz"

# Optional: path to a joblib-saved FirstLevelModel. Set to None to skip #5.
FLM_JOBLIB = None

STAT_THRESHOLD = 2.0
OUT_DIR = "./debug_mask_out"

# ====================== END CONFIG (don't edit below) =======================

import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import nibabel as nib
import numpy as np
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
from nilearn.image import resample_to_img
from nilearn.masking import intersect_masks
from nilearn.plotting import plot_roi, plot_stat_map
from scipy.ndimage import distance_transform_edt


MASK_GLOB = (
    f"{BASE_DIR}/fmriprep/{SUB}/{SES}/func/"
    f"{SUB}_{SES}_task-{TASK}_*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
)


def _voxel_count(img):
    """Count nonzero voxels in a binary-ish image."""
    return int((img.get_fdata() > 0).sum())


def _coords_in_mask(coords_mm, mask_img):
    """For each MNI mm coordinate (N x 3), return whether it sits inside ``mask_img``."""
    mask = mask_img.get_fdata() > 0
    shape = np.array(mask.shape)
    inv = np.linalg.inv(mask_img.affine)
    coords = np.atleast_2d(np.asarray(coords_mm, dtype=float))
    ijk = np.round(
        (inv @ np.c_[coords, np.ones(len(coords))].T).T[:, :3]
    ).astype(int)
    in_bounds = np.all((ijk >= 0) & (ijk < shape), axis=1)
    out = np.zeros(len(ijk), dtype=bool)
    for k, valid in enumerate(in_bounds):
        if valid:
            out[k] = mask[ijk[k, 0], ijk[k, 1], ijk[k, 2]]
    return out


def diagnostic_1(mask_img, stat_map, out_dir, log):
    """#1: current mask overlaid on the stat_map.

    If you see suprathreshold colour outside the red mask, your t-map has
    non-zero values that aren't being filtered by the mask (i.e. the
    masking step before get_clusters_table wasn't applied).
    """
    display = plot_roi(
        mask_img, bg_img=stat_map, display_mode="ortho",
        title="#1: current mask_img overlaid on stat_map",
    )
    path = out_dir / "01_mask_on_stat_map.png"
    display.savefig(str(path), dpi=120)
    display.close()
    log(f"  #1: wrote {path}")


def diagnostic_2(mask_img, out_dir, log):
    """#2: current mask overlaid on the MNI152 anatomical template.

    Any red voxels that sit outside the visible brain on the template are
    the 'analysis-vs-display rim' — voxels fmriprep considers in-brain
    but the canonical MNI152 template draws as outside-brain.
    """
    display = plot_roi(
        mask_img, bg_img=load_mni152_template(), display_mode="ortho",
        title="#2: current mask_img overlaid on MNI152 anatomy",
    )
    path = out_dir / "02_mask_on_template.png"
    display.savefig(str(path), dpi=120)
    display.close()
    log(f"  #2: wrote {path}")


def diagnostic_3(mask_paths, out_dir, log):
    """#3: each per-run mask separately; flag voxel-count outliers (>5% from median).

    If one run's mask is much larger than the others, picking [0] (or any
    fixed index) may have anchored the analysis to the most permissive run.
    """
    template = load_mni152_template()
    counts = []
    for i, path in enumerate(mask_paths):
        img = nib.load(path)
        c = _voxel_count(img)
        counts.append(c)
        display = plot_roi(
            img, bg_img=template, display_mode="z", cut_coords=5,
            title=f"#3: run {i + 1} mask ({c} voxels)",
        )
        out = out_dir / f"03_run_{i + 1:02d}_mask.png"
        display.savefig(str(out), dpi=120)
        display.close()

    if not counts:
        log("  #3: no per-run masks found")
        return

    median = float(np.median(counts))
    log(f"  #3: per-run voxel counts (median = {median:.0f}):")
    for i, (p, c) in enumerate(zip(mask_paths, counts)):
        pct = abs(c - median) / median * 100 if median else 0.0
        flag = "  <-- OUTLIER (>5% from median)" if pct > 5 else ""
        log(f"      run {i + 1}: {c} ({pct:.1f}% from median){flag}"
            f"  [{os.path.basename(p)}]")


def diagnostic_4(mask_paths, current_mask, out_dir, log):
    """#4: intersect_masks(threshold=1.0, connected=False); compare to current; plot.

    Strict intersection keeps only voxels present in *every* run's mask.
    Compare voxel count to the [0] mask: large delta means the [0] mask
    contains voxels that aren't reliable across runs.
    """
    intersected = intersect_masks(mask_paths, threshold=1.0, connected=False)
    ic, cc = _voxel_count(intersected), _voxel_count(current_mask)
    log(f"  #4: intersect_masks(threshold=1.0) voxel count = {ic}")
    log(f"      current [0] mask voxel count             = {cc}")
    delta_pct = (cc - ic) / max(cc, 1) * 100
    log(f"      delta: {cc - ic} voxels ({delta_pct:.1f}% larger in current [0])")

    display = plot_roi(
        intersected, bg_img=load_mni152_template(), display_mode="ortho",
        title="#4: intersect_masks(threshold=1.0) overlaid on MNI152 anatomy",
    )
    path = out_dir / "04_intersected_mask.png"
    display.savefig(str(path), dpi=120)
    display.close()
    return intersected


def diagnostic_5(flm, supplied_mask, log):
    """#5: confirm FLM.masker_.mask_img_ matches the mask we passed in.

    If FAIL, the GLM was actually fit against a different mask than the
    one being applied before get_clusters_table — likely cause of mismatch.
    """
    if flm is None:
        log("  #5: SKIPPED (set FLM_JOBLIB to enable)")
        return

    fitted = flm.masker_.mask_img_
    fc = _voxel_count(fitted)
    sc = _voxel_count(supplied_mask)
    affines_ok = np.allclose(fitted.affine, supplied_mask.affine)
    diff_sum = float(np.abs(
        (fitted.get_fdata() > 0).astype(int)
        - (supplied_mask.get_fdata() > 0).astype(int)
    ).sum())
    status = "PASS" if (fc == sc and affines_ok and diff_sum == 0) else "FAIL"
    log(f"  #5: masker consistency check: {status}")
    log(f"      fitted mask voxels = {fc}; supplied mask voxels = {sc}")
    log(f"      affines match: {affines_ok}; sum-of-abs-diff = {diff_sum:.0f}")


def diagnostic_6(mask_img, bold_path, log):
    """#6: print the mask and one BOLD affine side by side; flag mismatch."""
    bold = nib.load(bold_path)
    match = np.allclose(mask_img.affine, bold.affine)
    log("  #6: affines")
    log("      mask_img:")
    for r in mask_img.affine:
        log(f"        {r}")
    log(f"      {os.path.basename(bold_path)}:")
    for r in bold.affine:
        log(f"        {r}")
    log(f"      affines match: {match}")


def diagnostic_7(stat_map, current_mask, intersected, threshold, log):
    """#7 (HEADLINE): suprathreshold-voxel breakdown by mask membership +
    distance-from-template-brain histogram for the outside-all-three voxels.

    For every voxel with |t| > threshold, report how many fall inside:
      (a) current [0] mask
      (b) intersected (threshold=1.0) mask
      (c) MNI152 template brain mask
      (d) outside ALL three

    Then for the (d) voxels, compute distance from the template brain
    surface and report a histogram. The shape of that distribution is
    the discriminator:
      - mostly 0-2 mm out  -> analysis-vs-display rim mismatch (cosmetic)
      - concentrated 2-5 mm out -> draining veins / dural sinus signal
      - substantial fraction 5+ mm out -> registration error or
        multiband slice-leakage ghosts (NOT cosmetic — real artifact)
    """
    data = stat_map.get_fdata()
    above = np.abs(data) > threshold
    n = int(above.sum())
    log(f"  #7: suprathreshold voxels (|t| > {threshold}): {n}")
    if n == 0:
        return

    ijk_all = np.array(np.where(above)).T
    mm_all = (stat_map.affine @ np.c_[ijk_all, np.ones(len(ijk_all))].T).T[:, :3]
    t_all = data[above]

    # Resample all reference masks to the stat_map grid for fair comparison.
    mni_r = resample_to_img(load_mni152_brain_mask(), stat_map, interpolation="nearest")
    cur_r = resample_to_img(current_mask, stat_map, interpolation="nearest")
    ix_r = resample_to_img(intersected, stat_map, interpolation="nearest")

    in_cur = _coords_in_mask(mm_all, cur_r)
    in_ix = _coords_in_mask(mm_all, ix_r)
    in_mni = _coords_in_mask(mm_all, mni_r)
    in_any = in_cur | in_ix | in_mni

    def pct(x):
        return f"{x.mean() * 100:5.1f}%"

    log(f"      (a) inside current [0] mask:            "
        f"{int(in_cur.sum()):>7} ({pct(in_cur)})")
    log(f"      (b) inside intersected (threshold=1.0): "
        f"{int(in_ix.sum()):>7} ({pct(in_ix)})")
    log(f"      (c) inside MNI152 template brain mask:  "
        f"{int(in_mni.sum()):>7} ({pct(in_mni)})")
    log(f"      (d) outside ALL three:                  "
        f"{int((~in_any).sum()):>7} ({pct(~in_any)})")

    rim = int((in_cur & ~in_mni).sum())
    log(f"      'analysis-vs-display rim' (in current but outside template): {rim}")

    # ---- distance-from-template-brain histogram for (d) voxels ----
    d_mask = ~in_any
    if d_mask.sum() == 0:
        log("      no (d) voxels — distance histogram skipped")
        return

    template_arr = mni_r.get_fdata() > 0
    # voxel sizes in mm (per-axis); abs(diag) handles any LR-flip in the affine
    voxel_sizes_mm = np.abs(np.diag(mni_r.affine))[:3]
    # distance_transform_edt(~template) gives, for each outside-brain voxel,
    # the Euclidean distance to the nearest in-brain voxel — in mm when we
    # pass voxel sizes via ``sampling``.
    dist_mm = distance_transform_edt(~template_arr, sampling=voxel_sizes_mm)

    d_ijk = ijk_all[d_mask]
    d_distances = dist_mm[d_ijk[:, 0], d_ijk[:, 1], d_ijk[:, 2]]
    d_mm = mm_all[d_mask]
    d_t = t_all[d_mask]

    log(f"      (d) distance-from-template-brain (mm): "
        f"n={len(d_distances)}, min={d_distances.min():.1f}, "
        f"median={np.median(d_distances):.1f}, max={d_distances.max():.1f}")
    log(f"      (d) distance histogram:")
    bins = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, np.inf)]
    for lo, hi in bins:
        in_bin = (d_distances >= lo) & (d_distances < hi)
        n_in = int(in_bin.sum())
        pct_in = n_in / len(d_distances) * 100 if len(d_distances) > 0 else 0
        hi_str = "inf" if hi == np.inf else f"{hi:>4.0f}"
        log(f"          {lo:>4.1f} - {hi_str} mm: {n_in:>6} ({pct_in:5.1f}%)")
        if n_in > 0:
            sample_ix = np.where(in_bin)[0][:3]
            for k in sample_ix:
                coord = d_mm[k]
                log(f"            sample: t={d_t[k]:+.2f}  "
                    f"MNI [{coord[0]:>5.0f}, {coord[1]:>5.0f}, {coord[2]:>5.0f}]  "
                    f"({d_distances[k]:.1f} mm out)")

    pct_le2 = (d_distances < 2).mean() * 100
    pct_5plus = (d_distances >= 5).mean() * 100
    log(f"      INTERPRETATION:")
    log(f"        - {pct_le2:.0f}% of (d) voxels are within 2 mm of the template "
        f"brain (rim/vein story)")
    log(f"        - {pct_5plus:.0f}% are 5+ mm out (suggests registration error "
        f"or multiband slice-leakage)")


def diagnostic_8(stat_map, out_dir, log):
    """#8: plot the unthresholded stat_map for visual sanity."""
    display = plot_stat_map(
        stat_map, threshold=None, display_mode="ortho",
        title="#8: unthresholded stat_map (sanity check)",
    )
    path = out_dir / "08_stat_map_unthresholded.png"
    display.savefig(str(path), dpi=120)
    display.close()
    log(f"  #8: wrote {path}")


def diagnostic_9(stat_map, threshold, out_dir, log):
    """#9: thresholded stat_map at default vs nearest interpolation.

    plot_stat_map's default ``resampling_interpolation='continuous'`` is
    cubic-spline. When the stat_map and the (1 mm) MNI152 backdrop are at
    different resolutions, the cubic-spline kernel can bleed bright voxels
    by ~1-3 voxels (~2-6 mm) BEFORE the threshold is applied. That can
    create visible floaters in the rendered image that don't correspond
    to any actual suprathreshold voxel in the source data.

    If #9 nearest looks much cleaner than #9 continuous, the floaters
    were rendering bleed, not real data. If both look identical, the
    floaters are in the data and #7 tells you where.
    """
    for interp in ("continuous", "nearest"):
        display = plot_stat_map(
            stat_map,
            threshold=threshold,
            display_mode="ortho",
            resampling_interpolation=interp,
            title=f"#9: stat_map (t > {threshold}), interpolation='{interp}'",
        )
        path = out_dir / f"09_stat_map_{interp}.png"
        display.savefig(str(path), dpi=120)
        display.close()
        log(f"  #9: wrote {path}")


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(exist_ok=True)

    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    log("=== debug_mask.py ===")
    log(f"sub={SUB} ses={SES} task={TASK}")

    mask_paths = sorted(glob.glob(MASK_GLOB))
    if not mask_paths:
        raise RuntimeError(f"No per-run masks matched glob:\n  {MASK_GLOB}")
    log(f"Found {len(mask_paths)} per-run mask(s)")

    current_mask = nib.load(mask_paths[0])
    stat_map = nib.load(STAT_MAP_PATH)

    flm = None
    if FLM_JOBLIB:
        import joblib
        flm = joblib.load(FLM_JOBLIB)

    diagnostic_1(current_mask, stat_map, out_dir, log)
    diagnostic_2(current_mask, out_dir, log)
    diagnostic_3(mask_paths, out_dir, log)
    intersected = diagnostic_4(mask_paths, current_mask, out_dir, log)
    diagnostic_5(flm, current_mask, log)
    diagnostic_6(current_mask, BOLD_PATH, log)
    diagnostic_7(stat_map, current_mask, intersected, STAT_THRESHOLD, log)
    diagnostic_8(stat_map, out_dir, log)
    diagnostic_9(stat_map, STAT_THRESHOLD, out_dir, log)

    summary = out_dir / "summary.txt"
    summary.write_text("\n".join(lines) + "\n")
    print(f"\nSummary -> {summary}")


if __name__ == "__main__":
    main()
