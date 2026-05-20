import os
import tempfile

from nilearn_oob.figures import (
    plot_masks,
    plot_peak_scatter,
    plot_smoking_gun,
    plot_will_style,
)


def test_plot_will_style_writes_png(variants):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.png")
        plot_will_style(variants, path)
        assert os.path.exists(path) and os.path.getsize(path) > 0


def test_plot_smoking_gun_writes_png(variants):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.png")
        plot_smoking_gun(variants, path)
        assert os.path.exists(path) and os.path.getsize(path) > 0


def test_plot_peak_scatter_writes_png(variants):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.png")
        plot_peak_scatter(variants, path)
        assert os.path.exists(path) and os.path.getsize(path) > 0


def test_plot_masks_writes_png(variants):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.png")
        plot_masks(variants, path)
        assert os.path.exists(path) and os.path.getsize(path) > 0
