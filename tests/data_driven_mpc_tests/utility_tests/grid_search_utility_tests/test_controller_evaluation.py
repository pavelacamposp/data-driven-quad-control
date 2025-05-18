from unittest.mock import Mock, patch

import numpy as np
import torch.multiprocessing as mp
from direct_data_driven_mpc.nonlinear_data_driven_mpc_controller import (
    NonlinearDataDrivenMPCController,
)

from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.controller_evaluation import (  # noqa: E501
    CtrlEvalStatus,
    EnvSimInfo,
    evaluate_dd_mpc_controller_combination,
    isolated_controller_evaluation,
    sim_nonlinear_dd_mpc_control_loop_parallel,
)
from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.param_grid_search_config import (  # noqa: E501
    DDMPCCombinationParams,
    DDMPCEvaluationParams,
    DDMPCFixedParams,
    EnvResetSignal,
)
from data_driven_quad_control.envs.hover_env_config import EnvState

CONTROLLER_EVAL_MODULE_PATH = (
    "data_driven_quad_control.data_driven_mpc.utilities.param_grid_search."
    "controller_evaluation"
)

RUN_IN_ISOLATED_PROCESS_PATCH_PATH = (
    CONTROLLER_EVAL_MODULE_PATH + ".run_in_isolated_process"
)

SIM_DD_MPC_CONTROLLER_PATCH_PATH = (
    CONTROLLER_EVAL_MODULE_PATH + ".sim_nonlinear_dd_mpc_control_loop_parallel"
)

CREATE_DD_MPC_CONTROLLER_PATCH_PATH = (
    CONTROLLER_EVAL_MODULE_PATH + ".create_dd_mpc_controller_for_combination"
)


@patch(RUN_IN_ISOLATED_PROCESS_PATCH_PATH)
def test_evaluate_dd_mpc_controller_combination(
    mock_run_in_isolated_process: Mock,
    test_combination_params: DDMPCCombinationParams,
    test_fixed_params: DDMPCFixedParams,
    test_eval_params: DDMPCEvaluationParams,
    test_drone_state: EnvState,
) -> None:
    # Mock RMSE return value from `run_in_isolated_process`
    mock_run_in_isolated_process.return_value = 1.0

    # Retrieve controller parameters from test objects
    N = test_combination_params.N
    m = test_fixed_params.m
    p = test_fixed_params.p
    np_random = np.random.default_rng(0)

    # Create test parameters
    u_N = np_random.uniform(-1.0, 1.0, (N, m))
    y_N = np.ones((N, p))
    dummy_queue: mp.Queue = mp.Queue()
    dummy_progress = mp.Value("i", 0)

    # Evaluate the controller combination with mocked controller logic
    status, result = evaluate_dd_mpc_controller_combination(
        env_idx=0,
        data_entry_idx=0,
        u_N=u_N,
        y_N=y_N,
        initial_drone_state=test_drone_state,
        env_reset_queue=dummy_queue,
        action_queue=dummy_queue,
        observation_queue=dummy_queue,
        combination_params=test_combination_params,
        fixed_params=test_fixed_params,
        eval_params=test_eval_params,
        progress=dummy_progress,
    )

    # Verify that the evaluation was successful and returned the expected RMSE
    assert status == CtrlEvalStatus.SUCCESS
    assert result["average_RMSE"] == 1.0

    # Ensure all combination parameters are included in the result dictionary
    for param in test_combination_params._fields:
        assert result[param] == getattr(test_combination_params, param)


@patch(SIM_DD_MPC_CONTROLLER_PATCH_PATH)
@patch(CREATE_DD_MPC_CONTROLLER_PATCH_PATH)
def test_isolated_controller_evaluation(
    mock_create_controller: Mock,
    mock_sim_controller_loop: Mock,
    mock_dd_mpc_controller: NonlinearDataDrivenMPCController,
    test_combination_params: DDMPCCombinationParams,
    test_eval_params: DDMPCEvaluationParams,
    test_fixed_params: DDMPCFixedParams,
) -> None:
    # Mock controller return value from
    # `create_dd_mpc_controller_for_combination`
    mock_create_controller.return_value = mock_dd_mpc_controller

    # Mock RMSE return value from `sim_nonlinear_dd_mpc_control_loop_parallel`
    mock_sim_controller_loop.return_value = 1.0

    # Retrieve controller parameters from test objects
    N = test_combination_params.N
    m = test_fixed_params.m
    p = test_fixed_params.p
    np_random = np.random.default_rng(0)

    # Create test parameters
    u_N = np_random.uniform(-1.0, 1.0, (N, m))
    y_N = np.ones((N, p))
    y_r = np.ones((p, 1))
    dummy_queue: mp.Queue = mp.Queue()
    env_sim_info = EnvSimInfo()

    # Evaluate controller in isolation with mocked
    # controller creation and simulation logic
    rmse = isolated_controller_evaluation(
        env_idx=0,
        data_entry_idx=0,
        u_N=u_N,
        y_N=y_N,
        y_r=y_r,
        env_reset_queue=dummy_queue,
        action_queue=dummy_queue,
        observation_queue=dummy_queue,
        combination_params=test_combination_params,
        fixed_params=test_fixed_params,
        num_steps=test_eval_params.eval_time_steps,
        initial_distance=1.0,
        max_target_dist_increment=test_eval_params.max_target_dist_increment,
        env_sim_info=env_sim_info,
    )

    # Verify RMSE is the expected return value
    assert rmse == 1.0

    # Verify that mocked functions were correctly called
    assert mock_create_controller.called
    assert mock_sim_controller_loop.called


def test_sim_nonlinear_dd_mpc_control_loop_parallel(
    mock_dd_mpc_controller: NonlinearDataDrivenMPCController,
    test_eval_params: DDMPCEvaluationParams,
    test_fixed_params: DDMPCFixedParams,
) -> None:
    # Create test parameters
    env_idx = 0
    data_entry_idx = 123
    dummy_controller = mock_dd_mpc_controller
    dummy_env_reset_queue: mp.Queue = mp.Queue()
    dummy_action_queue: mp.Queue = mp.Queue()
    dummy_observation_queue: mp.Queue = mp.Queue()
    env_sim_info = EnvSimInfo(reset_state=False, sim_step_progress=0)

    # Pre-fill observation queue with fake observations to prevent deadlocks
    for _ in range(test_eval_params.eval_time_steps):
        dummy_observation_queue.put({env_idx: np.array([0.0, 0.0, 0.0])})

    # Simulation control loop with mocked controller and observation queue data
    rmse = sim_nonlinear_dd_mpc_control_loop_parallel(
        env_idx=env_idx,
        data_entry_idx=data_entry_idx,
        env_reset_queue=dummy_env_reset_queue,
        action_queue=dummy_action_queue,
        observation_queue=dummy_observation_queue,
        env_sim_info=env_sim_info,
        dd_mpc_controller=dummy_controller,
        fixed_params=test_fixed_params,
        num_steps=test_eval_params.eval_time_steps,
        initial_distance=1.0,
        max_target_dist_increment=test_eval_params.max_target_dist_increment,
    )

    # Verify the controller simulation returned a valid RMSE
    assert isinstance(rmse, float)
    assert rmse >= 0

    # Verify reset queue data
    reset_signal: EnvResetSignal = dummy_env_reset_queue.get()
    assert reset_signal.env_idx == env_idx
    assert isinstance(reset_signal.reset, bool)
    assert isinstance(reset_signal.done, bool)
    assert reset_signal.data_entry_idx == data_entry_idx

    # Verify action queue data
    received_env_idx, action = dummy_action_queue.get()
    assert received_env_idx == env_idx
    assert isinstance(action, np.ndarray)
    assert action.shape == (test_fixed_params.m,)
