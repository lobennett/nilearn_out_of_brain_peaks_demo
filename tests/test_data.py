# tests/test_data.py
import nibabel as nib
import pandas as pd
import pytest
from nilearn_oob.data import load_dataset


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


def test_returns_two_runs(dataset):
    assert len(dataset.bold_imgs) == 2
    assert len(dataset.event_dfs) == 2


def test_bold_imgs_are_4d_niftis(dataset):
    for img in dataset.bold_imgs:
        assert isinstance(img, nib.Nifti1Image)
        assert len(img.shape) == 4, f"expected 4D, got shape {img.shape}"


def test_event_dfs_are_dataframes_with_required_columns(dataset):
    for df in dataset.event_dfs:
        assert isinstance(df, pd.DataFrame)
        for col in ("onset", "duration", "trial_type"):
            assert col in df.columns
        assert set(df["trial_type"].unique()) <= {"faces", "scrambled"}
        assert len(df) > 0


def test_tr_is_positive(dataset):
    assert dataset.t_r > 0


def test_runs_share_affine(dataset):
    """Both runs must be in the same MNI grid; otherwise intersect_masks fails downstream."""
    import numpy as np
    a0 = dataset.bold_imgs[0].affine
    a1 = dataset.bold_imgs[1].affine
    assert np.allclose(a0, a1), f"affines differ: {a0} vs {a1}"
