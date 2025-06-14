from typing import Any
from unittest.mock import Mock, patch

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


@pytest.mark.parametrize("add_obs_noise", [True, False])
def test_env_compute_observations(
    add_obs_noise: bool, mock_env: HoverEnv
) -> None:
    # Notes:
    # - `mock_env._add_noise` adds the `obs_noise_std` value to tensors
    #    as a constant "noise" to simplify testing.
    # - `mock_env.obs_scales` are all set to 1.0. This removes observation
    #   scaling and simplifies clipping, as mocked observations are in
    #   the [-1, 1] range.

    # Mock observation noise as a fixed constant
    if add_obs_noise:
        mock_env.obs_noise_std = 0.1
        mocked_noise = 0.1
    else:
        mocked_noise = 0.0

    # Construct expected observation
    pos = mock_env.rel_pos + mocked_noise
    quat = mock_env.base_quat + mocked_noise
    quat = quat / quat.norm(dim=1, keepdim=True)
    lin_vel = mock_env.base_lin_vel + mocked_noise
    ang_vel = mock_env.base_ang_vel + mocked_noise
    last_actions = mock_env.last_actions

    expected_obs = torch.cat(
        [pos, quat, lin_vel, ang_vel, last_actions], dim=-1
    )

    # Test `HoverEnv.compute_observations`
    obs = HoverEnv.compute_observations(mock_env)

    # Verify that the computed observation matches the expected tensor
    torch.testing.assert_close(obs, expected_obs)


@pytest.mark.parametrize("add_noise", [True, False])
@patch("torch.randn_like")
def test_env_get_pos_quat(
    mock_randn: Mock, add_noise: bool, mock_env: HoverEnv
) -> None:
    # Patch `torch.randn_like()` to `torch.ones_like`
    mock_randn.side_effect = lambda x: torch.ones_like(x)

    # Mock observation noise as a fixed constant
    mock_env.obs_noise_std = 0.1 if add_noise else 0.0

    # Construct expected observations
    if add_noise:
        expected_pos = (
            mock_env.base_pos
            + mock_env.obs_noise_std / mock_env.obs_scales["rel_pos"]
        )
    else:
        expected_pos = mock_env.base_pos

    pos = HoverEnv.get_pos(mock_env, add_noise)

    # Verify that the computed position matches the expected tensor
    torch.testing.assert_close(pos, expected_pos)


@pytest.mark.parametrize("add_noise", [True, False])
def test_env_get_quat(add_noise: bool, mock_env: HoverEnv) -> None:
    # Mock observation noise as a fixed constant
    mock_env.obs_noise_std = 0.1 if add_noise else 0.0

    # Construct expected observations
    if add_noise:
        expected_quat = mock_env.base_quat + mock_env.obs_noise_std
        expected_quat = expected_quat / expected_quat.norm(dim=1, keepdim=True)
    else:
        expected_quat = mock_env.base_quat

    quat = HoverEnv.get_quat(mock_env, add_noise)

    # Verify computed quaternion matches the expected tensor
    torch.testing.assert_close(quat, expected_quat)
