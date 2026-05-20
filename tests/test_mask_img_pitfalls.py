"""Behavior of ``FirstLevelModel`` when ``mask_img`` is given the wrong type.

The most common footgun in this area is forgetting to collapse a list of
per-run mask paths (from ``glob.glob(...)``) into a single combined mask
before passing to ``FirstLevelModel(mask_img=...)``. This test documents
that nilearn raises ``TypeError`` in that case — the bug is loud, not silent.

If you reach ``fit()`` without an error, your ``mask_img`` is at least a
single image (not a list); the next places to check are whether the mask's
voxel count is sensible and whether it actually covers the brain.
"""
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import FirstLevelModel


def _toy_inputs():
    """Return a tiny 4-D BOLD, two trivial masks, and a 2-condition events DF."""
    shape3 = (8, 8, 8)
    affine = np.eye(4)
    rng = np.random.RandomState(0)
    bold = nib.Nifti1Image(rng.randn(*shape3, 12).astype(np.float32), affine)
    mask1 = nib.Nifti1Image(np.ones(shape3, dtype=np.int8), affine)
    mask2 = nib.Nifti1Image(np.ones(shape3, dtype=np.int8), affine)
    events = pd.DataFrame({
        "onset": [2.0, 8.0, 14.0, 20.0],
        "duration": [1.0, 1.0, 1.0, 1.0],
        "trial_type": ["a", "b", "a", "b"],
    })
    return bold, mask1, mask2, events


def test_mask_img_as_list_raises_typeerror():
    """Passing ``mask_img=[mask1, mask2]`` triggers a TypeError on ``fit()``."""
    bold, mask1, mask2, events = _toy_inputs()
    flm = FirstLevelModel(t_r=2.0, mask_img=[mask1, mask2], hrf_model="spm")
    with pytest.raises(TypeError, match="(?i)mask"):
        flm.fit(bold, events=events)


def test_mask_img_as_single_image_succeeds():
    """Sanity check: a single Nifti mask is accepted (this is the correct usage)."""
    bold, mask1, _, events = _toy_inputs()
    flm = FirstLevelModel(t_r=2.0, mask_img=mask1, hrf_model="spm")
    flm.fit(bold, events=events)
    tmap = flm.compute_contrast("a - b", output_type="stat")
    assert tmap.shape == bold.shape[:3]
