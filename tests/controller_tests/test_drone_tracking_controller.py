import signal
from typing import Any

import pytest
import torch

from data_driven_quad_control.envs.hover_env import HoverEnv
from data_driven_quad_control.envs.hover_env_config import (
    EnvActionType,
)
from data_driven_quad_control.utilities.drone_tracking_controller import (
    create_drone_tracking_controller,
    hover_at_target,
)


def signal_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("hover_at_target took too long")


@pytest.mark.integration
@pytest.mark.drone_track_controller_integration
def test_drone_tracking_controller(
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

    # Define position and yaw target tensors
    target_pos = torch.tensor(
        [
            [0.0, 1.0, 1.5],
            [-1.0, -0.0, 0.2],
        ],
        device=env.device,
        dtype=torch.float,
    )
    target_yaw = torch.tensor([0.0, 1.0], device=env.device, dtype=torch.float)

    # Create drone tracking controller
    tracking_controller = create_drone_tracking_controller(env=env)

    # Configure a signal alarm to interrupt if hover execution takes
    # too long to prevent the test from hanging indefinitely
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(5)

    # Command drones to hover at target positions and yaws
    try:
        hover_at_target(
            env=env,
            tracking_controller=tracking_controller,
            target_pos=target_pos,
            target_yaw=target_yaw,
            min_at_target_steps=10,
            error_threshold=5e-2,
            ctbr_controller=None,
        )
    finally:
        signal.alarm(0)

    # Test passes if no exception is raised by `hover_at_target`, which
    # means that each drone was stabilized successfully at their target
    assert True
