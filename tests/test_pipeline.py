import pytest
from nilearn_oob.pipeline import run_three_variants


@pytest.fixture(scope="module")
def results():
    return run_three_variants()


def test_three_variants_returned(results):
    assert set(results.keys()) == {"A", "B", "C"}


def test_each_variant_has_required_keys(results):
    for v in ("A", "B", "C"):
        r = results[v]
        for key in ("tmap", "peaks", "mask"):
            assert key in r, f"{v} missing {key}"


def test_variant_A_has_no_mask(results):
    assert results["A"]["mask"] is None
