"""Orchestrate the three masking variants end-to-end.

Variant A: no mask anywhere (Will's setup)
Variant B: intersect_masks(threshold=0.0) -- UNION of run masks (permissive)
Variant C: intersect_masks(threshold=1.0) -- strict intersection (recommended)

We use threshold=0.0 vs 1.0 (not 0.5 vs 1.0) because nilearn's
``intersect_masks`` uses strict ``>`` for the threshold check, which makes
threshold=0.5 functionally identical to threshold=1.0 when there are only
2 input masks. With N>=3 masks (the typical real-world case), threshold=0.5
lies between union and intersection. See tests/test_masks.py for an
empirical demonstration on this dataset, and the README for discussion.

The reference mask used to classify peaks as in/out of brain is the strict
intersection of run masks -- a voxel is "in brain" iff it is in every run's
EPI mask. This is a defensible operational definition for this subject-space
demo dataset, and means variant C's peaks are all in-brain by construction
while variants A and B can produce peaks outside this reference.
"""
from .data import load_dataset
from .glm import fit_variant
from .masks import build_run_masks, combine_masks
from .peaks import extract_peaks


def run_three_variants(inject_synthetic_outlier=True):
    """Fit all three variants and return a dict ``{label: {tmap, peaks, mask}}``.

    ``inject_synthetic_outlier`` adds a single out-of-brain hot spot to each t-map
    so the demo's variant-A bug is visible. The README explicitly flags this.
    """
    dataset = load_dataset()
    run_masks = build_run_masks(dataset.bold_imgs)
    mask_union = combine_masks(run_masks, threshold=0.0)
    mask_intersection = combine_masks(run_masks, threshold=1.0)
    ref_mask = mask_intersection  # strict intersection is the "in-brain" reference

    variants = [
        ("A", None),
        ("B", mask_union),
        ("C", mask_intersection),
    ]
    out = {}
    for label, mask in variants:
        tmap = fit_variant(dataset, mask_img=mask,
                           inject_synthetic_outlier=inject_synthetic_outlier)
        peaks = extract_peaks(tmap, mask_img=mask, ref_mask=ref_mask)
        out[label] = {"tmap": tmap, "peaks": peaks, "mask": mask}
    return out
