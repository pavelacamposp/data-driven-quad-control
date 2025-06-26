import os

import pytest


@pytest.fixture
def test_comparison_params_path() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "config/comparison/test_controller_comparison_config.yaml",
    )
