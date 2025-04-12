from typing import Any

import torch

from data_driven_quad_control.envs.hover_env import HoverEnv

CfgDict = dict[str, Any]


def get_cfgs() -> tuple[CfgDict, CfgDict, CfgDict, CfgDict]:
    env_cfg = {
        # simulation
        "dt": 0.01,  # sim freq = 100 Hz
        "decimation": 4,  # ctrl freq = 1 / (0.01 * 4) = 25 Hz
        # actions
        "num_actions": 4,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        # termination
        "termination_if_roll_greater_than": 180,  # degree
        "termination_if_pitch_greater_than": 180,
        "termination_if_close_to_ground": 0.1,
        "termination_if_x_greater_than": 3.0,
        "termination_if_y_greater_than": 3.0,
        "termination_if_z_greater_than": 2.0,
        # drone initial pose
        "base_init_pos": [0.0, 0.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        # episode config
        "episode_length_s": 15.0,
        "at_target_threshold": 0.1,
        "resampling_time_s": 3.0,
        # visualization
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 100,  # 1 / dt = 100 Hz
    }
    obs_cfg = {
        "num_obs": 17,
        "obs_scales": {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
        },
    }
    reward_cfg = {
        "yaw_lambda": -10.0,
        "reward_scales": {
            "target": 10.0,
            "smooth": -1e-4,
            "yaw": 0.01,
            "angular": -2e-4,
            "crash": -10.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "pos_x_range": [-1.0, 1.0],
        "pos_y_range": [-1.0, 1.0],
        "pos_z_range": [1.0, 1.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def create_env(
    num_envs: int,
    env_cfg: Any,
    obs_cfg: Any,
    reward_cfg: Any,
    command_cfg: Any,
    show_viewer: bool = False,
    device: torch.device | str = "cuda",
) -> HoverEnv:
    env = HoverEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
        device=device,
        auto_target_updates=False,  # Disable automatic target position updates
    )

    return env


def get_current_env_state(
    env: HoverEnv, env_idx: int
) -> dict[str, torch.Tensor]:
    # Convert env_idx int into a tensor list of env indices
    envs_idx_tensor = get_tensor_from_env_idx(env=env, env_idx=env_idx)

    return env.get_current_state(envs_idx=envs_idx_tensor)


def restore_env_from_state(
    env: HoverEnv, env_idx: int, saved_state: dict[str, torch.Tensor]
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
