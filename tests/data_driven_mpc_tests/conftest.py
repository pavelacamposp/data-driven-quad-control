import os

import pytest

from tests.mocks import (
    MockDroneSystemModel,
    MockDroneTrackingController,
)


@pytest.fixture
def mock_system_model() -> MockDroneSystemModel:
    return MockDroneSystemModel()


@pytest.fixture
def mock_tracking_controller() -> MockDroneTrackingController:
    return MockDroneTrackingController()


@pytest.fixture
def test_controller_params_path() -> str:
    return os.path.join(
        os.path.dirname(__file__), "config", "test_controller_params.yaml"
    )


@pytest.fixture
def test_grid_search_params_path() -> str:
    return os.path.join(
        os.path.dirname(__file__), "config", "test_grid_search_params.yaml"
    )
