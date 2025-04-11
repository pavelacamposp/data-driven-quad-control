from typing import Any

import numpy as np
import torch

from src.utilities.drone_environment import (
    create_env,
    get_current_env_state,
    restore_env_from_state,
    update_env_target_pos,
)
from src.utilities.drone_system_model import create_system_model


def test_hover_env_utilities(
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
    )

    # Reset environment
    obs, _ = env.reset()
    assert obs.shape == (num_envs, dummy_obs_cfg["num_obs"])

    # Initialize drone system model
    base_env_idx = 0  # System model uses a specified env idx
    system_model = create_system_model(env=env, env_idx=base_env_idx)

    # Update drone env target
    hover_target = (0.0, 0.0, 1.5)
    hover_target_tensor = torch.tensor(
        [hover_target], device=env.device, dtype=torch.float
    )
    update_env_target_pos(
        env=env, env_idx=base_env_idx, target_pos=hover_target_tensor
    )

    # Get current env state
    initial_base_env_state = get_current_env_state(
        env=env, env_idx=base_env_idx
    )

    # Step environment via system_model
    num_steps = 5
    m = system_model.m  # System inputs
    p = system_model.p  # System outputs
    U = np.zeros((num_steps, m))
    Y = np.zeros((num_steps, p))
    W = system_model.eps_max * np.random.uniform(-1.0, 1.0, (num_steps, p))
    for k in range(num_steps):
        # Simulate drone
        Y[k, :] = system_model.simulate_step(u=U[k, :], w=W[k, :])

    # Restore env state
    restore_env_from_state(
        env=env, env_idx=base_env_idx, saved_state=initial_base_env_state
    )

    # Get restored env state
    restored_base_env_state = get_current_env_state(
        env=env, env_idx=base_env_idx
    )

    assert compare_dicts_str_tensor(
        initial_base_env_state, restored_base_env_state
    ), "Env states differ after restore."


def compare_dicts_str_tensor(
    dict_1: dict[str, torch.Tensor], dict_2: dict[str, torch.Tensor]
) -> bool:
    if dict_1.keys() != dict_2.keys():
        return False

    for key in dict_1:
        value_1, value_2 = dict_1[key], dict_2[key]

        # Check if tensor values are equal
        if isinstance(value_1, torch.Tensor) and isinstance(
            value_2, torch.Tensor
        ):
            if not torch.equal(value_1, value_2):
                return False

        else:
            return False  # Unexpected type or mismatched types

    return True
