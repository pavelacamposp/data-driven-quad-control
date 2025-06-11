from typing import Any
from unittest.mock import Mock, patch

import torch

from data_driven_quad_control.comparison.utilities.parallel_controller_sim import (  # noqa: E501
    parallel_controller_simulation,
)
from data_driven_quad_control.envs.hover_env import HoverEnv
from data_driven_quad_control.utilities.control_data_plotting import (
    ControlTrajectory,
)

PARELLEL_CONTROLLER_SIM_PATH = (
    "data_driven_quad_control.comparison.utilities.parallel_controller_sim."
)

MP_GET_START_METHOD_PATCH_PATH = (
    PARELLEL_CONTROLLER_SIM_PATH + "mp.get_start_method"
)

TRACKING_WORKER_PATCH_PATH = (
    PARELLEL_CONTROLLER_SIM_PATH + "tracking_controller_worker"
)
RL_WORKER_PATCH_PATH = PARELLEL_CONTROLLER_SIM_PATH + "rl_controller_worker"
DD_MPC_WORKER_PATCH_PATH = (
    PARELLEL_CONTROLLER_SIM_PATH + "dd_mpc_controller_worker"
)

UPDATE_SIM_PROGRESS_PATCH_PATH = (
    PARELLEL_CONTROLLER_SIM_PATH + "update_simulation_progress"
)


@patch(UPDATE_SIM_PROGRESS_PATCH_PATH)
@patch(DD_MPC_WORKER_PATCH_PATH)
@patch(RL_WORKER_PATCH_PATH)
@patch(TRACKING_WORKER_PATCH_PATH)
@patch(MP_GET_START_METHOD_PATCH_PATH)
def test_parallel_controller_sim(
    mock_mpc_get_start_method: Mock,
    mock_tracking_worker: Mock,
    mock_rl_worker: Mock,
    mock_dd_mpc_worker: Mock,
    mock_update_sim_progress: Mock,
    mock_env: HoverEnv,
) -> None:
    # Mock return value of `mp.get_start_method` to ensure it returns "spawn"
    # without actually setting it, since it affects external tests
    mock_mpc_get_start_method.return_value = "spawn"

    # Set the number of environments to three for the mocked env,
    # one per each controller worker
    num_envs = 3
    mock_env.num_envs = num_envs
    mock_env.obs_buf = torch.zeros((num_envs, mock_env.num_obs))
    mock_env.rew_buf = torch.zeros((num_envs,))
    mock_env.reset_buf = torch.zeros((num_envs,))

    # Define test parameters
    test_eval_setpoints = [torch.zeros((1, 3))]

    # Patch controller workers to mimic expected queue behavior
    def dummy_controller_worker(*args: Any, **kwargs: Any) -> None:
        env_idx = args[0]
        target_signal_queue = args[-3]
        action_queue = args[-2]
        observation_queue = args[-1]

        # Mock controller closed-loop simulation
        # Get target signal
        target_signal_queue.get()

        # Send dummy action and get dummy observation
        action_queue.put((env_idx, torch.zeros((mock_env.num_actions))))
        observation_queue.get()

    # Patch controller workers
    mock_tracking_worker.side_effect = dummy_controller_worker
    mock_rl_worker.side_effect = dummy_controller_worker
    mock_dd_mpc_worker.side_effect = dummy_controller_worker

    # Patch `update_simulation_progress` so it terminates
    # the simulation on the first iteration and to prevent deadlocks
    def dummy_update_simulation_progress(*args: Any, **kwargs: Any) -> None:
        sim_info = kwargs["sim_info"]
        sim_info.target_done = True

    mock_update_sim_progress.side_effect = dummy_update_simulation_progress

    control_trajectory_data = parallel_controller_simulation(
        env=mock_env,
        tracking_env_idx=0,
        tracking_controller_init_data=Mock(),
        rl_env_idx=1,
        rl_controller_init_data=Mock(),
        dd_mpc_env_idx=2,
        dd_mpc_controller_init_data=Mock(),
        eval_setpoints=test_eval_setpoints,
    )

    # Verify the function executed and returned as expected
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
