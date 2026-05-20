"""Per-run brain-mask computation and cross-run mask combination."""
from nilearn.masking import compute_epi_mask, intersect_masks


def build_run_masks(bold_imgs):
    """Return one EPI-derived brain mask per 4-D BOLD image."""
    return [compute_epi_mask(b) for b in bold_imgs]


def combine_masks(run_masks, threshold):
    """Combine per-run masks with ``intersect_masks(threshold=...)``.

    threshold=1.0 -> strict intersection (voxel must be in every run mask).
    threshold=0.5 -> nilearn default (voxel must be in >=50% of run masks).
    """
    return intersect_masks(run_masks, threshold=threshold)
