from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

from data_driven_quad_control.comparison.utilities.comparison_config_loader import (  # noqa: E501
    load_controller_comparison_params,
)
from data_driven_quad_control.comparison.utilities.controller_comparison_config import (  # noqa: E501
    RLControllerInitData,
    TrackingControllerInitData,
)
from data_driven_quad_control.comparison.utilities.dd_mpc_initial_data_collection import (  # noqa: E501
    get_data_driven_mpc_controller_init_data,
)
from data_driven_quad_control.comparison.utilities.parallel_controller_sim import (  # noqa: E501
    parallel_controller_simulation,
)
from data_driven_quad_control.controllers.tracking.tracking_controller_config import (  # noqa: E501
    TrackingCtrlDroneState,
)
from data_driven_quad_control.envs.hover_env_config import (
    EnvActionType,
)
from data_driven_quad_control.learning.config.hover_ppo_config import (
    get_train_cfg,
)
from data_driven_quad_control.utilities.control_data_plotting import (
    ControlTrajectory,
)
from data_driven_quad_control.utilities.drone_environment import create_env
from data_driven_quad_control.utilities.drone_tracking_controller import (
    create_drone_tracking_controller,
)

MP_GET_START_METHOD_PATCH_PATH = (
    "data_driven_quad_control.comparison.utilities.parallel_controller_sim."
    "mp.get_start_method"
)


@pytest.mark.integration
@pytest.mark.comparison_integration
@patch(MP_GET_START_METHOD_PATCH_PATH)
def test_controller_comparison(
    mock_mpc_get_start_method: Mock,
    dummy_env_cfg: dict[str, Any],
    dummy_obs_cfg: dict[str, Any],
    dummy_reward_cfg: dict[str, Any],
    dummy_command_cfg: dict[str, Any],
    test_comparison_params_path: str,
) -> None:
    # Mock return value of `mp.get_start_method` to ensure it returns "spawn"
    # without actually setting it, since it affects external tests
    #
    # Note:
    # This works for this integration test because the simulation is
    # immediately terminated after the first step (the target position is the
    # same as the initial drone position). In the actual comparison script,
    # not setting the start method to "spawn" will raise:
    #   "RuntimeError: Cannot re-initialize CUDA in forked subprocess."
    mock_mpc_get_start_method.return_value = "spawn"

    # Note: Genesis initialized in `tests/conftest.py`

    # Extend episode length
    dummy_env_cfg["episode_length_s"] = 100

    # Initialize environment
    num_envs = 3  # Number of controllers
    drone_colors = [(1.0, 0.0, 0.0, 1.0)] * 3
    env = create_env(
        num_envs=num_envs,
        env_cfg=dummy_env_cfg,
        obs_cfg=dummy_obs_cfg,
        reward_cfg=dummy_reward_cfg,
        command_cfg=dummy_command_cfg,
        show_viewer=False,
        device="cpu",
        action_type=EnvActionType.CTBR_FIXED_YAW,
        drone_colors=drone_colors,
    )

    # Reset environment
    obs, _ = env.reset()

    # Load controller configuration data
    controller_comparison_params = load_controller_comparison_params(
        config_path=test_comparison_params_path,
        env_device=env.device,
    )

    # --- Construct controller initialization data ---
    # Tracking controller initialization data
    tracking_env_idx = 0
    tracking_controller_config = (
        controller_comparison_params.tracking_controller_config
    )
    init_drone_tracking_ctrl_state = TrackingCtrlDroneState(
        X=env.get_pos()[tracking_env_idx].unsqueeze(0),
        Q=env.get_quat()[tracking_env_idx].unsqueeze(0),
    )
    tracking_controller_init_data = TrackingControllerInitData(
        controller_config=tracking_controller_config,
        controller_dt=env.step_dt,
        initial_state=init_drone_tracking_ctrl_state,
    )

    # RL controller (PPO policy) initialization data:
    rl_env_idx = 1
    train_cfg = get_train_cfg("trained_ppo_policy", 0)
    rl_controller_init_data = RLControllerInitData(
        train_cfg=train_cfg,
        model_path=controller_comparison_params.ppo_model_path,
        initial_observation=obs[rl_env_idx],
    )

    # Nonlinear data-driven MPC controller initialization data
    dd_mpc_env_idx = 2
    dd_mpc_controller_config = (
        controller_comparison_params.dd_mpc_controller_config
    )
    init_hover_pos = controller_comparison_params.init_hover_pos

    # Create a stabilizing controller for initial data collection
    stabilizing_controller = create_drone_tracking_controller(env=env)

    # Collect initial data for the data-driven MPC controller, while
    # stabilizing the drones corresponding to the other controllers
    dd_mpc_controller_init_data = get_data_driven_mpc_controller_init_data(
        env=env,
        dd_mpc_env_idx=dd_mpc_env_idx,
        dd_mpc_controller_config=dd_mpc_controller_config,
        init_hover_pos=init_hover_pos,
        stabilizing_controller=stabilizing_controller,
        np_random=np.random.default_rng(0),
    )

    # Run controller comparison simulation
    control_trajectory_data = parallel_controller_simulation(
        env=env,
        tracking_env_idx=tracking_env_idx,
        tracking_controller_init_data=tracking_controller_init_data,
        rl_env_idx=rl_env_idx,
        rl_controller_init_data=rl_controller_init_data,
        dd_mpc_env_idx=dd_mpc_env_idx,
        dd_mpc_controller_init_data=dd_mpc_controller_init_data,
        eval_setpoints=controller_comparison_params.eval_setpoints,
        min_at_target_steps=1,
        error_threshold=5e-2,
    )

    # Close environment
    env.close()

    # Verify that the trajectory data matches the expected structure
    assert isinstance(control_trajectory_data, ControlTrajectory)

    # Verify that the input and output trajectory
    # lists have the expected number of elements
    assert len(control_trajectory_data.control_inputs) == num_envs
    assert len(control_trajectory_data.system_outputs) == num_envs

    # Verify that the control trajectory data arrays have the expected shape
    input_array_shape = control_trajectory_data.control_inputs[0].shape
    output_array_shape = control_trajectory_data.system_outputs[0].shape
    setpoint_array_shape = control_trajectory_data.system_setpoint.shape

    for i in range(num_envs):
        input_array = control_trajectory_data.control_inputs[i]
        output_array = control_trajectory_data.system_outputs[i]
        setpoint_array = control_trajectory_data.system_setpoint

        assert input_array.shape == input_array_shape
        assert output_array.shape == output_array_shape
        assert setpoint_array.shape == setpoint_array_shape

        assert input_array.shape[1] == num_envs
        assert output_array.shape[1] == num_envs
        assert setpoint_array.shape[1] == num_envs
