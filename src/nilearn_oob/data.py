# src/nilearn_oob/data.py
"""Load the SPM multimodal-fMRI demo dataset as two runs of the same subject.

``fetch_spm_multimodal_fmri()`` returns each run's BOLD as a *list* of 390
per-volume ``.img`` paths. We concatenate them into single 4-D Nifti1Image
objects per run, matching nilearn's own SPM-multimodal tutorial.

The dataset ships with onset times in MATLAB ``.mat`` files where onsets are
expressed in scans (not seconds). We parse them once and return BIDS-style
event DataFrames with onset in seconds.

Documented TR for this dataset: 2.0 s. Trial duration is 0 s (impulse events).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from nilearn.datasets import fetch_spm_multimodal_fmri
from nilearn.image import concat_imgs, resample_to_img
from scipy.io import loadmat

T_R = 2.0


@dataclass(frozen=True)
class Dataset:
    bold_imgs: tuple
    event_dfs: tuple
    t_r: float


def _onsets_mat_to_df(mat_path, t_r):
    """Convert one SPM-multimodal onsets ``.mat`` file to a BIDS events DataFrame."""
    timing = loadmat(mat_path)
    faces_onsets = timing["onsets"][0][0].ravel() * t_r
    scrambled_onsets = timing["onsets"][0][1].ravel() * t_r
    onsets = np.concatenate([faces_onsets, scrambled_onsets])
    trial_types = ["faces"] * len(faces_onsets) + ["scrambled"] * len(scrambled_onsets)
    return pd.DataFrame({
        "onset": onsets,
        "duration": np.zeros(len(onsets)),
        "trial_type": trial_types,
    })


def load_dataset() -> Dataset:
    """Fetch SPM multimodal-fMRI (faces task, two runs, one subject).

    ``concat_imgs(auto_resample=True)`` only aligns volumes *within* each run.
    The two runs end up with affines that differ in translation by ~0.4 mm,
    which breaks ``intersect_masks`` across runs. We resample run 2 onto run
    1's affine so the demo's mask-intersection step works.
    """
    raw = fetch_spm_multimodal_fmri()
    run1 = concat_imgs(raw.func1, auto_resample=True)
    run2 = concat_imgs(raw.func2, auto_resample=True)
    run2 = resample_to_img(run2, run1)
    return Dataset(
        bold_imgs=(run1, run2),
        event_dfs=(
            _onsets_mat_to_df(raw.trials_ses1, T_R),
            _onsets_mat_to_df(raw.trials_ses2, T_R),
        ),
        t_r=T_R,
    )
