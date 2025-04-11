from typing import Any

import torch

from src.envs.hover_env import HoverEnv


def test_hover_env_loop(
    dummy_env_cfg: dict[str, Any],
    dummy_obs_cfg: dict[str, Any],
    dummy_reward_cfg: dict[str, Any],
    dummy_command_cfg: dict[str, Any],
) -> None:
    # Note: Genesis initialized in `tests/conftest.py`

    # Initialize environment
    num_envs = 2
    env = HoverEnv(
        num_envs=2,
        env_cfg=dummy_env_cfg,
        obs_cfg=dummy_obs_cfg,
        reward_cfg=dummy_reward_cfg,
        command_cfg=dummy_command_cfg,
        show_viewer=False,
        device="cpu",
    )

    # Reset environment
    obs, _ = env.reset()
    assert obs.shape == (num_envs, dummy_obs_cfg["num_obs"])

    # Step environment
    num_steps = 5
    for _ in range(num_steps):
        dummy_actions = torch.zeros(
            (num_envs, dummy_env_cfg["num_actions"]),
            dtype=torch.float,
            device=env.device,
        )
        obs, _, reward, done, info = env.step(dummy_actions)

        assert obs.shape == (num_envs, dummy_obs_cfg["num_obs"])
        assert reward.shape == (num_envs,)
        assert done.shape == (num_envs,)
        assert isinstance(info, dict)

        # Check that rewards do not contain NaNs
        assert not torch.isnan(reward).any(), "Reward contains NaNs"

    # Sanity check reward keys
    for k in dummy_reward_cfg["reward_scales"].keys():
        assert f"rew_{k}" in env.extras["episode"]
