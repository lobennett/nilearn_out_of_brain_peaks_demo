"""Demonstrate that masking is what eliminates out-of-brain peaks."""
import numpy as np


def test_variant_A_produces_out_of_brain_peaks(variants):
    """With no mask, at least one peak falls outside the reference brain mask."""
    df = variants["A"]["peaks"]
    out_of_brain = (~df["in_brain"]).sum()
    assert out_of_brain >= 1, (
        f"variant A should reproduce the bug but had {out_of_brain} out-of-brain peaks"
    )


def test_variant_B_reduces_out_of_brain_peaks(variants):
    """Union mask (threshold=0.0) reduces but may not eliminate out-of-brain peaks."""
    a = (~variants["A"]["peaks"]["in_brain"]).sum()
    b = (~variants["B"]["peaks"]["in_brain"]).sum()
    assert b < a, f"variant B ({b}) should have fewer out-of-brain peaks than A ({a})"


def test_variant_C_produces_no_out_of_brain_peaks(variants):
    """Strict intersection mask (threshold=1.0) eliminates all out-of-brain peaks."""
    df = variants["C"]["peaks"]
    out_of_brain = (~df["in_brain"]).sum()
    assert out_of_brain == 0, (
        f"variant C with strict mask still had {out_of_brain} out-of-brain peaks"
    )


def test_union_mask_is_superset_of_intersection_mask(variants):
    """The threshold=0.0 mask must contain every voxel of the threshold=1.0 mask."""
    d_union = variants["B"]["mask"].get_fdata() > 0
    d_intersect = variants["C"]["mask"].get_fdata() > 0
    assert np.all(d_union[d_intersect]), "every voxel in intersection mask should be in union mask"
    assert d_union.sum() > d_intersect.sum(), "union mask should be strictly larger"
