import nibabel as nib
import pytest
from nilearn_oob.data import load_dataset
from nilearn_oob.glm import fit_variant


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


def test_variant_A_fit_returns_tmap(dataset):
    t_map = fit_variant(dataset, mask_img=None)
    assert isinstance(t_map, nib.Nifti1Image)
    assert len(t_map.shape) == 3


def test_variant_with_mask_fit_returns_tmap(dataset):
    # use a tiny dummy mask: compute_epi_mask on one run's 4D image
    from nilearn.masking import compute_epi_mask
    mask = compute_epi_mask(dataset.bold_imgs[0])
    t_map = fit_variant(dataset, mask_img=mask)
    assert isinstance(t_map, nib.Nifti1Image)
