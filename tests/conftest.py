from typing import Any

import genesis as gs
import pytest

from .mocks import (
    MockHoverEnv,
)

test_gs_initialized = False


@pytest.fixture(scope="session", autouse=True)
def initialize_genesis() -> None:
    global test_gs_initialized
    if not test_gs_initialized:
        gs.init(backend=gs.cpu, logging_level="error")
        test_gs_initialized = True


@pytest.fixture
def dummy_env_cfg() -> dict[str, Any]:
    return {
        "dt": 0.01,
        "decimation": 4,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        "actuator_noise_std": 0.0,
        "termination_if_roll_greater_than": 180,  # degree
        "termination_if_pitch_greater_than": 180,
        "termination_if_close_to_ground": 0.1,
        "termination_if_x_greater_than": 3.0,
        "termination_if_y_greater_than": 3.0,
        "termination_if_z_greater_than": 2.0,
        "termination_if_ang_vel_greater_than": 12,
        "termination_if_lin_vel_greater_than": 20,
        "base_init_pos": [0.0, 0.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 15.0,
        "at_target_threshold": 0.1,
        "min_hover_time_s": 0.01,
        "resampling_time_s": 3.0,
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 100,
    }


@pytest.fixture
def dummy_obs_cfg() -> dict[str, Any]:
    return {
        "obs_scales": {
            "rel_pos": 1.0,
            "lin_vel": 1.0,
            "ang_vel": 1.0,
        },
        "obs_noise_std": 0.0,
    }


@pytest.fixture
def dummy_reward_cfg() -> dict[str, Any]:
    return {
        "yaw_lambda": -1.0,
        "reward_scales": {
            "target": 1.0,
            "closeness": 1.0,
            "hover_time": 1.0,
            "smooth": 1.0,
            "yaw": 1.0,
            "angular": 1.0,
            "crash": 1.0,
        },
    }


@pytest.fixture
def dummy_command_cfg() -> dict[str, Any]:
    return {
        "num_commands": 3,
        "pos_x_range": (-1.0, 1.0),
        "pos_y_range": (-1.0, 1.0),
        "pos_z_range": (0.5, 2.0),
    }


@pytest.fixture
def mock_env() -> MockHoverEnv:
    return MockHoverEnv()
