from typing import Any

import pytest
import torch
from torch.testing import assert_close

from data_driven_quad_control.envs.hover_env import HoverEnv
from data_driven_quad_control.envs.hover_env_config import (
    EnvActionType,
    EnvDroneParams,
)
from data_driven_quad_control.utilities.math_utils import linear_interpolate


@pytest.mark.integration
@pytest.mark.drone_ctbr_controller_integration
def test_drone_ctbr_controller(
    dummy_env_cfg: dict[str, Any],
    dummy_obs_cfg: dict[str, Any],
    dummy_reward_cfg: dict[str, Any],
    dummy_command_cfg: dict[str, Any],
) -> None:
    # Note: Genesis initialized in `tests/conftest.py`

    # Initialize environment
    num_envs = 2
    env = HoverEnv(
        num_envs=num_envs,
        env_cfg=dummy_env_cfg,
        obs_cfg=dummy_obs_cfg,
        reward_cfg=dummy_reward_cfg,
        command_cfg=dummy_command_cfg,
        show_viewer=False,
        device="cpu",
        action_type=EnvActionType.CTBR,
    )

    # Reset environment
    env.reset()

    run_hover_test(env)
    run_body_rate_test(env)


def run_hover_test(env: HoverEnv, tol: float = 0.1, steps: int = 10) -> None:
    """Test if drone hovers stably with gravity compensation thrust."""
    # Hover setpoint with only gravity compensation
    hover_setpoint = torch.tensor(
        [[EnvDroneParams.WEIGHT, 0.0, 0.0, 0.0]],
        dtype=torch.float,
        device=env.device,
    ).expand(env.num_envs, -1)

    # Normalize setpoint to [-1, 1] range to calculate the env action
    env_action = linear_interpolate(
        x=hover_setpoint,
        x_min=env.action_bounds[:, 0],
        x_max=env.action_bounds[:, 1],
        y_min=-1,
        y_max=1,
    )

    # Step environment
    with torch.no_grad():
        for _ in range(steps):
            env.step(env_action)

    # Check that the drone is hovering statically
    assert torch.all(torch.abs(env.base_ang_vel) < tol), "Hover control fails"


def run_body_rate_test(
    env: HoverEnv, tol: float = 1e-3, steps: int = 20
) -> None:
    """Test if drone correctly tracks a roll/pitch/yaw rate setpoint."""
    # Body rates setpoint with only gravity compensation
    test_setpoint = torch.tensor(
        [[EnvDroneParams.WEIGHT, 0.01, -0.01, 1.0]],
        dtype=torch.float,
        device=env.device,
    ).expand(env.num_envs, -1)

    # Normalize setpoint to [-1, 1] range to calculate the env action
    env_action = linear_interpolate(
        x=test_setpoint,
        x_min=env.action_bounds[:, 0],
        x_max=env.action_bounds[:, 1],
        y_min=-1,
        y_max=1,
    )

    # Step environment
    with torch.no_grad():
        for _ in range(steps):
            env.step(env_action)

    # Check that the drone angular velocity error is small
    assert_close(
        env.base_ang_vel,
        test_setpoint[:, 1:],
        rtol=0.0,
        atol=tol,
        msg="Angular velocity tracking fails",
    )
