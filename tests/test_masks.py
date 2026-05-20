# tests/test_masks.py
import numpy as np
import pytest
from nilearn_oob.data import load_dataset
from nilearn_oob.masks import build_run_masks, combine_masks


@pytest.fixture(scope="module")
def run_masks():
    ds = load_dataset()
    return build_run_masks(ds.bold_imgs)


def test_one_mask_per_run(run_masks):
    assert len(run_masks) == 2


def test_run_masks_are_3d(run_masks):
    for m in run_masks:
        assert len(m.shape) == 3


def test_higher_threshold_produces_smaller_mask(run_masks):
    """Strictly higher threshold should produce a strictly smaller mask.

    We use threshold=0.0 (union) vs threshold=1.0 (intersection) because
    on this 2-run dataset, threshold=0.5 produces an identical mask to
    threshold=1.0 (nilearn's intersect_masks uses strict `>`, so with N=2
    a proportion of 0.5 is NOT > 0.5). The README documents this.
    """
    mask_union = combine_masks(run_masks, threshold=0.0)
    mask_intersection = combine_masks(run_masks, threshold=1.0)
    d_union = mask_union.get_fdata() > 0
    d_intersection = mask_intersection.get_fdata() > 0
    assert np.all(d_union[d_intersection]), "intersection should be subset of union"
    assert d_union.sum() > d_intersection.sum(), "union should be strictly larger than intersection"


def test_threshold_05_equals_threshold_10_on_two_runs(run_masks):
    """N=2 corner case: nilearn's intersect_masks uses strict `>` comparison,
    so threshold=0.5 on 2 runs is functionally identical to threshold=1.0.
    This is an empirical observation that motivates using threshold=0.0 vs
    threshold=1.0 as the 'permissive vs strict' variants in this demo.
    For real data with N>=3 runs, threshold=0.5 would lie strictly between
    union and intersection.
    """
    assert len(run_masks) == 2
    mask_05 = combine_masks(run_masks, threshold=0.5)
    mask_10 = combine_masks(run_masks, threshold=1.0)
    assert np.array_equal(mask_05.get_fdata(), mask_10.get_fdata())
