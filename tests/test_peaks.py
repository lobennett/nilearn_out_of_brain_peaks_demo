# tests/test_peaks.py
import numpy as np
import nibabel as nib
import pandas as pd
import pytest
from nilearn_oob.brain_mask import load_reference_mask
from nilearn_oob.peaks import extract_peaks


def _toy_tmap_with_outlier():
    """Build a tiny 3-D t-map in MNI mm space with a hot spot at (0,0,0) and one at (0,-108,0)."""
    shape = (91, 109, 91)  # MNI152 2mm
    affine = np.array([
        [-2, 0, 0, 90],
        [0, 2, 0, -126],
        [0, 0, 2, -72],
        [0, 0, 0, 1],
    ], dtype=float)
    data = np.zeros(shape, dtype=np.float32)
    # In-brain peak at (0,0,0) mm and out-of-brain peak at (0,-108,0) mm
    inv = np.linalg.inv(affine)
    for mm in [(0, 0, 0), (0, -108, 0)]:
        ijk = np.round(inv @ np.r_[mm, 1])[:3].astype(int)
        data[ijk[0], ijk[1], ijk[2]] = 5.0
    return nib.Nifti1Image(data, affine)


def test_unmasked_extraction_finds_out_of_brain_peak():
    tmap = _toy_tmap_with_outlier()
    df = extract_peaks(tmap, mask_img=None, ref_mask=load_reference_mask())
    assert (df["in_brain"] == False).any()


def test_masked_extraction_drops_out_of_brain_peak():
    tmap = _toy_tmap_with_outlier()
    # Use the MNI152 reference mask as the masking image
    ref = load_reference_mask(target_img=tmap)
    df = extract_peaks(tmap, mask_img=ref, ref_mask=load_reference_mask())
    assert (df["in_brain"] == False).sum() == 0


def test_dataframe_has_expected_columns():
    tmap = _toy_tmap_with_outlier()
    df = extract_peaks(tmap, mask_img=None, ref_mask=load_reference_mask())
    for col in ("X", "Y", "Z", "Peak Stat", "in_brain"):
        assert col in df.columns
