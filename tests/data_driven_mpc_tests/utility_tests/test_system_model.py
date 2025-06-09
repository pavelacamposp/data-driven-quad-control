from typing import Any

import numpy as np
import pytest
import torch

from data_driven_quad_control.data_driven_mpc.utilities.drone_system_model import (  # noqa: E501
    create_system_model,
)
from data_driven_quad_control.envs.hover_env_config import EnvActionType
from data_driven_quad_control.utilities.drone_environment import create_env


@pytest.mark.integration
@pytest.mark.dd_mpc_system_model_integration
def test_system_model(
    dummy_env_cfg: dict[str, Any],
    dummy_obs_cfg: dict[str, Any],
    dummy_reward_cfg: dict[str, Any],
    dummy_command_cfg: dict[str, Any],
) -> None:
    # Note: Genesis initialized in `tests/conftest.py`

    # Create environment
    num_envs = 2
    env = create_env(
        num_envs=num_envs,
        env_cfg=dummy_env_cfg,
        obs_cfg=dummy_obs_cfg,
        reward_cfg=dummy_reward_cfg,
        command_cfg=dummy_command_cfg,
        show_viewer=False,
        device="cpu",
        action_type=EnvActionType.CTBR_FIXED_YAW,
    )
    env.reset()

    # Initialize drone system model
    base_env_idx = 0  # System model uses a specified env idx
    system_model = create_system_model(env=env, env_idx=base_env_idx)

    # Step environment via system_model
    num_steps = 5
    m = system_model.m  # System inputs
    p = system_model.p  # System outputs
    U = np.zeros((num_steps, m))
    Y = np.zeros((num_steps, p))
    W = system_model.eps_max * np.random.uniform(-1.0, 1.0, (num_steps, p))
    with torch.no_grad():
        for k in range(num_steps):
            # Simulate drone
            Y[k, :] = system_model.simulate_step(u=U[k, :], w=W[k, :])

    # Verify that the system output contains expected values
    assert not np.allclose(Y, 0.0), "All outputs are zero"
    assert np.all(np.isfinite(Y)), "Outputs contain NaNs or Infs"
