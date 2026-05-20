# src/nilearn_oob/brain_mask.py
"""Reference brain mask + in/out-of-brain coordinate classification."""
from __future__ import annotations

import numpy as np
from nilearn.datasets import load_mni152_brain_mask
from nilearn.image import resample_to_img


def load_reference_mask(target_img=None):
    """Return the MNI152 brain mask, optionally resampled to ``target_img``."""
    mask = load_mni152_brain_mask()
    if target_img is not None:
        mask = resample_to_img(mask, target_img, interpolation="nearest")
    return mask


def is_in_brain(coords, ref_mask):
    """Return whether each MNI coordinate (mm) is inside ``ref_mask``.

    ``coords`` may be a single (x, y, z) tuple or an (N, 3) array.
    Returns a bool scalar or bool array of length N.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    affine = ref_mask.affine
    data = ref_mask.get_fdata() > 0
    inv = np.linalg.inv(affine)
    homog = np.c_[coords, np.ones(len(coords))]
    voxels = (inv @ homog.T).T[:, :3]
    voxels = np.round(voxels).astype(int)
    shape = np.array(data.shape)
    in_bounds = np.all((voxels >= 0) & (voxels < shape), axis=1)
    result = np.zeros(len(coords), dtype=bool)
    valid_voxels = voxels[in_bounds]
    result[in_bounds] = data[valid_voxels[:, 0], valid_voxels[:, 1], valid_voxels[:, 2]]
    return result if len(result) > 1 else bool(result[0])
