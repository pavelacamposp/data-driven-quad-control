from typing import Any

import pytest
import torch

from data_driven_quad_control.envs.hover_env import HoverEnv
from data_driven_quad_control.envs.hover_env_config import EnvActionType


@pytest.mark.integration
@pytest.mark.drone_env_integration
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
        num_envs=num_envs,
        env_cfg=dummy_env_cfg,
        obs_cfg=dummy_obs_cfg,
        reward_cfg=dummy_reward_cfg,
        command_cfg=dummy_command_cfg,
        show_viewer=False,
        device="cpu",
        action_type=EnvActionType.CTBR_FIXED_YAW,
    )

    # Reset environment
    obs, _ = env.reset()
    assert obs.shape == (num_envs, env.num_obs)

    # Step environment
    num_steps = 5
    with torch.no_grad():
        for _ in range(num_steps):
            dummy_actions = torch.zeros(
                (num_envs, env.num_actions),
                dtype=torch.float,
                device=env.device,
            )
            obs, reward, done, info = env.step(dummy_actions)

            assert obs.shape == (num_envs, env.num_obs)
            assert reward.shape == (num_envs,)
            assert done.shape == (num_envs,)
            assert isinstance(info, dict)

            # Check that rewards do not contain NaNs
            assert not torch.isnan(reward).any(), "Reward contains NaNs"

    # Sanity check reward keys
    for k in dummy_reward_cfg["reward_scales"].keys():
        assert f"rew_{k}" in env.extras["episode"]
