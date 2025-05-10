from typing import Any

import numpy as np
import torch
from direct_data_driven_mpc.utilities.controller.controller_creation import (
    create_nonlinear_data_driven_mpc_controller,
)
from direct_data_driven_mpc.utilities.controller.controller_params import (
    get_nonlinear_data_driven_mpc_controller_params,
)
from direct_data_driven_mpc.utilities.controller.data_driven_mpc_sim import (
    simulate_nonlinear_data_driven_mpc_control_loop,
)

from data_driven_quad_control.data_driven_mpc.utilities.drone_initial_data_collection import (  # noqa: E501
    collect_initial_input_output_data,
    get_init_hover_pos,
)
from data_driven_quad_control.data_driven_mpc.utilities.drone_system_model import (  # noqa: E501
    create_system_model,
)
from data_driven_quad_control.envs.config.hover_env_config import (
    EnvActionType,
)
from data_driven_quad_control.utilities.drone_environment import (
    create_env,
)
from data_driven_quad_control.utilities.drone_tracking_controller import (
    create_drone_tracking_controller,
    hover_at_target,
)


def test_dd_mpc_controller_eval(
    dummy_env_cfg: dict[str, Any],
    dummy_obs_cfg: dict[str, Any],
    dummy_reward_cfg: dict[str, Any],
    dummy_command_cfg: dict[str, Any],
    test_controller_params_path: str,
) -> None:
    # Note: Genesis initialized in `tests/conftest.py`

    # Initialize environment
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

    # Reset environment
    env.reset()

    # Initialize drone system model
    base_env_idx = 0  # Index of the drone instance used for evaluating
    # the Data-Driven MPC controller
    system_model = create_system_model(env=env, env_idx=base_env_idx)

    m = env.num_actions  # Number of inputs
    p = 3  # Number of outputs
    dd_mpc_config = get_nonlinear_data_driven_mpc_controller_params(
        config_file=test_controller_params_path,
        controller_key_value="controller_key",
        m=m,
        p=p,
    )

    # Create a controller to stabilize the drone at a specific
    # position for initial input-output data collection
    stabilizing_controller = create_drone_tracking_controller(env=env)

    # Load initial hover target position from configuration file
    target_pos = get_init_hover_pos(
        config_path=test_controller_params_path,
        controller_key_value="controller_key",
        env=env,
    )
    target_yaw = torch.tensor([0.0], device=env.device, dtype=torch.float)

    # Command drone to hover at target
    target_pos = target_pos.expand(env.num_envs, -1)
    target_yaw = target_yaw.expand(env.num_envs)
    hover_at_target(
        env=env,
        tracking_controller=stabilizing_controller,
        target_pos=target_pos,
        target_yaw=target_yaw,
        ctbr_controller=None,
    )

    # Collect initial input-output measurement with a
    # generated persistently exciting input
    np_random = np.random.default_rng(0)
    u_N, y_N = collect_initial_input_output_data(
        env=env,
        base_env_idx=base_env_idx,
        stabilizing_controller=stabilizing_controller,
        target_pos=target_pos,
        target_yaw=target_yaw,
        input_bounds=dd_mpc_config["U"],
        u_range=dd_mpc_config["u_range"],
        N=dd_mpc_config["N"],
        m=system_model.m,
        p=system_model.p,
        eps_max=system_model.eps_max,
        np_random=np_random,
        drone_system_model=system_model,
    )

    # Create nonlinear data-driven MPC controller
    nonlinear_dd_mpc_controller = create_nonlinear_data_driven_mpc_controller(
        controller_config=dd_mpc_config, u=u_N, y=y_N
    )

    # Simulate Data-Driven MPC control system
    n_steps = 1
    u_sys, y_sys = simulate_nonlinear_data_driven_mpc_control_loop(
        system_model=system_model,
        data_driven_mpc_controller=nonlinear_dd_mpc_controller,
        n_steps=n_steps,
        np_random=np_random,
        verbose=0,
    )

    # Close environment
    env.close()

    # Verify control input and system output shapes
    assert u_sys.shape == (n_steps, system_model.m)
    assert y_sys.shape == (n_steps, system_model.p)

    # Verify that the system output contains only finite values
    assert np.all(np.isfinite(y_sys)), "System output contains NaNs or Infs"
