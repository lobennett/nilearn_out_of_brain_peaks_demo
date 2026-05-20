#!/usr/bin/env python3
"""End-to-end reproduction: fit 3 variants, save peak CSVs and 3 figures.

Usage: python reproduce.py [--out-dir OUT_DIR]
"""
import argparse
import os
import sys

from nilearn_oob.figures import (
    plot_masks,
    plot_peak_scatter,
    plot_smoking_gun,
    plot_will_style,
)
from nilearn_oob.pipeline import run_three_variants


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=".", help="Where to write CSVs + figures/")
    args = parser.parse_args(argv)

    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("Fitting three variants (this is slow — three GLM fits)...")
    variants = run_three_variants()

    for label, r in variants.items():
        csv_path = os.path.join(args.out_dir, f"peaks_{label}.csv")
        r["peaks"].to_csv(csv_path, index=False)
        n_out = (~r["peaks"]["in_brain"]).sum()
        print(f"  Variant {label}: {len(r['peaks'])} peaks, {n_out} out of brain -> {csv_path}")

    print("\nWriting figures...")
    plot_will_style(variants, os.path.join(fig_dir, "will_style_AvBvC.png"))
    plot_smoking_gun(variants, os.path.join(fig_dir, "smoking_gun_AvBvC.png"))
    plot_peak_scatter(variants, os.path.join(fig_dir, "peak_locations_AvBvC.png"))
    plot_masks(variants, os.path.join(fig_dir, "masks_BvC.png"))
    print(f"Wrote figures to {fig_dir}/")


if __name__ == "__main__":
    sys.exit(main())
