import pytest
from nilearn_oob.pipeline import run_three_variants


@pytest.fixture(scope="session")
def variants():
    """Run all three GLM variants once per session and share the result."""
    return run_three_variants()
