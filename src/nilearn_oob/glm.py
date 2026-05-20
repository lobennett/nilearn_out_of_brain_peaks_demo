"""Fit a two-run FirstLevelModel on the SPM faces task and return a contrast t-map.

The contrast is faces > scrambled (the canonical contrast for this dataset).
"""
import nibabel as nib
import numpy as np
from nilearn.glm.first_level import FirstLevelModel


# Synthetic ghost peak locations, in voxel coordinates for the SPM-multimodal
# 4-D BOLD shape (64, 64, 32). Each entry is (center_i, center_j, center_k,
# peak_t). The locations sit on FOV faces or corners — far enough from the
# brain center that they fall outside any run's EPI mask.
#
# These are intended as visual analogs to multi-band slice leakage ghosts
# (which deposit task-correlated signal in air voxels at half-FOV
# displacements). The dataset isn't multi-band, so the constellation is
# stamped in by hand. Variants B and C zero these out at peak extraction;
# variant A reports them as out-of-brain peaks.
_SYNTHETIC_GHOSTS = (
    (32, 60, 16, 7.5),   # posterior FOV edge (analog of MNI Y=-108)
    (32, 3, 16, 5.0),    # anterior FOV edge
    (3, 32, 16, 6.0),    # left lateral FOV edge (analog of MNI X=-72)
    (60, 32, 16, 6.5),   # right lateral FOV edge (analog of MNI X=+72)
    (32, 32, 1, 4.8),    # inferior FOV edge (analog of MNI Z=-69)
    (32, 32, 30, 5.8),   # superior FOV edge (analog of MNI Z=+87)
)


def fit_variant(dataset, mask_img, inject_synthetic_outlier=False):
    """Fit FirstLevelModel across both runs with the given mask and return faces>scrambled t-map.

    ``mask_img`` may be a Nifti1Image, a path, or None (nilearn auto-computes via compute_epi_mask).
    If ``inject_synthetic_outlier`` is True, a constellation of high-t voxels is
    stamped into the t-map at far-from-brain FOV edges to mimic the kind of
    multi-band slice-leakage ghosts the colleague is seeing. Used only because
    this demo dataset isn't multi-band and doesn't naturally produce visible
    out-of-brain artifacts.
    """
    model = FirstLevelModel(
        t_r=dataset.t_r,
        hrf_model="spm",
        mask_img=mask_img,
        standardize=False,
        noise_model="ar1",
        minimize_memory=True,
    )
    model = model.fit(list(dataset.bold_imgs), events=list(dataset.event_dfs))
    # faces > scrambled — passing as a formula expression lets nilearn match
    # condition names across both runs' design matrices regardless of order
    tmap = model.compute_contrast("faces - scrambled", output_type="stat")
    if inject_synthetic_outlier:
        tmap = _add_synthetic_outliers(tmap)
    return tmap


def _add_synthetic_outliers(tmap):
    """Stamp synthetic multi-band-ghost-like peaks at FOV-edge voxels.

    For each entry in ``_SYNTHETIC_GHOSTS``, paints a 3x3x3 block of voxels
    centered on the specified voxel coordinate. The center voxel gets the
    full peak t-value; neighbors falloff linearly. This produces a small
    blob (~5-7 mm across, one peak after min_distance merging) at each
    location instead of an invisible single voxel.
    """
    data = np.array(tmap.get_fdata(), copy=True)
    shape = np.array(data.shape)
    for ci, cj, ck, peak_t in _SYNTHETIC_GHOSTS:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    i, j, k = ci + di, cj + dj, ck + dk
                    if not (0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]):
                        continue
                    dist = abs(di) + abs(dj) + abs(dk)
                    value = peak_t * (1.0 - 0.18 * dist)
                    if value > data[i, j, k]:
                        data[i, j, k] = value
    return nib.Nifti1Image(data, tmap.affine, tmap.header)
