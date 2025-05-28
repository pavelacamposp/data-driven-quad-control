from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch.multiprocessing as mp

from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.controller_evaluation import (  # noqa: E501
    CtrlEvalStatus,
)
from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.parallel_worker import (  # noqa: E501
    worker_data_driven_mpc,
)
from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.param_grid_search_config import (  # noqa: E501
    DataDrivenCache,
    DDMPCCombinationParams,
    DDMPCEvaluationParams,
    DDMPCFixedParams,
    EnvResetSignal,
)

EVALUATE_DD_MPC_CONTROLLER_PATCH_PATH = (
    "data_driven_quad_control.data_driven_mpc.utilities.param_grid_search."
    "parallel_worker.evaluate_dd_mpc_controller_combination"
)


@pytest.mark.parametrize("eval_succeeded", [True, False])
@patch(EVALUATE_DD_MPC_CONTROLLER_PATCH_PATH)
def test_worker_data_driven_mpc(
    mock_evaluate_controller: Mock,
    eval_succeeded: bool,
    test_combination_params: DDMPCCombinationParams,
    test_data_driven_cache: DataDrivenCache,
    test_fixed_params: DDMPCFixedParams,
    test_eval_params: DDMPCEvaluationParams,
) -> None:
    # Mock controller evaluation from
    # `evaluate_dd_mpc_controller_combination`
    success_return = {"success": True, "average_RMSE": 1.0}
    failure_return = {
        "success": False,
        "n_successful_runs": 0,
        "average_RMSE": float(np.nan),
    }

    def evaluate_controller_side_effect(*args: Any, **kwargs: Any) -> Any:
        progress = kwargs["progress"]

        if eval_succeeded:
            with progress.get_lock():
                progress.value += 1

            return (
                CtrlEvalStatus.SUCCESS,
                success_return,
            )
        else:
            return (
                CtrlEvalStatus.FAILURE,
                failure_return,
            )

    mock_evaluate_controller.side_effect = evaluate_controller_side_effect

    # Create test parameters
    process_id = 0
    dummy_manager = mp.Manager()
    dummy_env_reset_queue: mp.Queue = mp.Queue()
    dummy_queue: mp.Queue = mp.Queue()
    dummy_combination_params_queue: mp.Queue = mp.Queue()
    dummy_successful_results = dummy_manager.list()
    dummy_failed_results = dummy_manager.list()
    dummy_lock = mp.Lock()
    dummy_progress = mp.Value("i", 0)

    # Add one combination parameter to the combination parameter queue
    dummy_combination_params_queue.put(test_combination_params)

    # Run worker
    worker_data_driven_mpc(
        process_id=process_id,
        env_reset_queue=dummy_env_reset_queue,
        action_queue=dummy_queue,
        observation_queue=dummy_queue,
        combination_params_queue=dummy_combination_params_queue,
        successful_results=dummy_successful_results,
        failed_result=dummy_failed_results,
        lock=dummy_lock,
        progress=dummy_progress,
        data_driven_cache=test_data_driven_cache,
        fixed_params=test_fixed_params,
        eval_params=test_eval_params,
    )

    # Verify success results list data
    if eval_succeeded:
        assert len(dummy_successful_results) == 1
        assert len(dummy_failed_results) == 0
        assert dummy_successful_results[0] == success_return
    else:
        assert len(dummy_successful_results) == 0
        assert len(dummy_failed_results) == 1
        np.testing.assert_equal(dummy_failed_results[0], failure_return)

    # Verify that the progress value increased by the number of evaluation runs
    num_eval_runs = (
        len(test_eval_params.eval_setpoints)
        * test_eval_params.num_collections_per_N
    )
    assert dummy_progress.value == num_eval_runs

    # Verify process sent a task completion signal through the env reset queue
    reset_signal = dummy_env_reset_queue.get()
    assert reset_signal == EnvResetSignal(process_id, False, True, None), (
        "Worker sent wrong reset signal"
    )


@patch(EVALUATE_DD_MPC_CONTROLLER_PATCH_PATH)
def test_worker_data_driven_mpc_exits_on_empty_queue(
    mock_evaluate_controller: Mock,
    test_data_driven_cache: DataDrivenCache,
    test_fixed_params: DDMPCFixedParams,
    test_eval_params: DDMPCEvaluationParams,
) -> None:
    # Create test parameters
    process_id = 0
    dummy_manager = mp.Manager()
    dummy_env_reset_queue: mp.Queue = mp.Queue()
    dummy_queue: mp.Queue = mp.Queue()
    dummy_combination_params_queue: mp.Queue = mp.Queue()  # Empty
    dummy_successful_results = dummy_manager.list()
    dummy_failed_result = dummy_manager.list()
    dummy_lock = mp.Lock()
    dummy_progress = mp.Value("i", 0)

    # Run worker
    worker_data_driven_mpc(
        process_id=process_id,
        env_reset_queue=dummy_env_reset_queue,
        action_queue=dummy_queue,
        observation_queue=dummy_queue,
        combination_params_queue=dummy_combination_params_queue,
        successful_results=dummy_successful_results,
        failed_result=dummy_failed_result,
        lock=dummy_lock,
        progress=dummy_progress,
        data_driven_cache=test_data_driven_cache,
        fixed_params=test_fixed_params,
        eval_params=test_eval_params,
    )

    # Check that evaluate function was never called
    mock_evaluate_controller.assert_not_called()

    # Verify that result lists are empty
    assert len(dummy_successful_results) == 0
    assert len(dummy_failed_result) == 0

    # Verify that the progress value did not increase
    assert dummy_progress.value == 0, (
        "Worker should not increment the progress value"
    )

    # Verify that a reset signal was correctly sent
    reset_signal = dummy_env_reset_queue.get()
    assert reset_signal == EnvResetSignal(process_id, False, True, None), (
        "Worker sent wrong reset signal on empty queue"
    )
