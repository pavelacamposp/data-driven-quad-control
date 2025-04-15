from typing import Any

import torch

from data_driven_quad_control.envs.config.hover_env_config import (
    EnvActionType,
    EnvState,
)
from data_driven_quad_control.envs.hover_env import HoverEnv


def create_env(
    num_envs: int,
    env_cfg: Any,
    obs_cfg: Any,
    reward_cfg: Any,
    command_cfg: Any,
    show_viewer: bool = False,
    device: torch.device | str = "cuda",
    action_type: EnvActionType = EnvActionType.CTBR_FIXED_YAW,
) -> HoverEnv:
    env = HoverEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
        device=device,
        action_type=action_type,
        auto_target_updates=False,  # Disable automatic target position updates
    )

    return env


def get_current_env_state(env: HoverEnv, env_idx: int) -> EnvState:
    # Convert env_idx int into a tensor list of env indices
    envs_idx_tensor = get_tensor_from_env_idx(env=env, env_idx=env_idx)

    return env.get_current_state(envs_idx=envs_idx_tensor)


def restore_env_from_state(
    env: HoverEnv, env_idx: int, saved_state: EnvState
) -> None:
    envs_idx_tensor = get_tensor_from_env_idx(env=env, env_idx=env_idx)

    # Set env state to `saved_state`
    env.restore_from_state(envs_idx=envs_idx_tensor, saved_state=saved_state)


def update_env_target_pos(
    env: HoverEnv, env_idx: int, target_pos: torch.Tensor
) -> None:
    envs_idx_tensor = get_tensor_from_env_idx(env=env, env_idx=env_idx)

    # Set env target position to `target_pos`
    env.update_target_pos(envs_idx=envs_idx_tensor, target_pos=target_pos)


def get_tensor_from_env_idx(env: HoverEnv, env_idx: int) -> torch.Tensor:
    # Convert env_idx int into a tensor list of env indices
    envs_idx_tensor = torch.tensor(
        [env_idx], dtype=torch.long, device=env.device
    )

    return envs_idx_tensor
