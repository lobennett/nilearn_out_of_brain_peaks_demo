import numpy as np
import pytest
from nilearn_oob.brain_mask import is_in_brain, load_reference_mask


@pytest.fixture(scope="module")
def ref_mask():
    return load_reference_mask()


def test_origin_is_in_brain(ref_mask):
    assert is_in_brain((0, 0, 0), ref_mask) is True


def test_far_posterior_coord_is_out_of_brain(ref_mask):
    # Y = -108 is well past the back of the MNI152 brain
    assert is_in_brain((0, -108, 0), ref_mask) is False


def test_far_lateral_coord_is_out_of_brain(ref_mask):
    assert is_in_brain((-72, -15, -6), ref_mask) is False


def test_vectorized_returns_array(ref_mask):
    coords = np.array([[0, 0, 0], [0, -108, 0], [-72, -15, -6]])
    out = is_in_brain(coords, ref_mask)
    assert np.array_equal(out, [True, False, False])
