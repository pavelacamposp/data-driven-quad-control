from typing import Any
from unittest.mock import Mock, patch

import numpy as np

from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.controller_evaluation import (  # noqa: E501
    CtrlEvalStatus,
)
from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.parallel_grid_search import (  # noqa: E501
    parallel_grid_search,
)
from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.param_grid_search_config import (  # noqa: E501
    DataDrivenCache,
    DDMPCCombinationParams,
    DDMPCEvaluationParams,
    DDMPCFixedParams,
    EnvResetSignal,
)
from data_driven_quad_control.envs.hover_env import HoverEnv
from data_driven_quad_control.envs.hover_env_config import EnvState

WORKER_DD_MPC_PATCH_PATH = (
    "data_driven_quad_control.data_driven_mpc.utilities.param_grid_search."
    "parallel_grid_search.worker_data_driven_mpc"
)


@patch(WORKER_DD_MPC_PATCH_PATH)
def test_parallel_grid_search(
    mock_worker: Mock,
    mock_env: HoverEnv,
    test_combination_params: DDMPCCombinationParams,
    test_fixed_params: DDMPCFixedParams,
    test_eval_params: DDMPCEvaluationParams,
    test_data_driven_cache: DataDrivenCache,
    test_drone_state_cache: dict[int, EnvState],
) -> None:
    # Patch worker to mimic expected queue behavior
    data_entry_idx = 0

    def dummy_worker(*args: Any, **kwargs: Any) -> None:
        process_id = args[0]
        env_reset_queue = args[1]
        action_queue = args[2]
        observation_queue = args[3]
        successful_results = args[5]
        failed_result = args[6]

        # Mock controller closed-loop simulation
        # Send reset signal to reset the drone environment
        # for the `process_id` env index
        env_reset_queue.put(
            EnvResetSignal(process_id, True, False, data_entry_idx)
        )

        # Send dummy action and get dummy observation
        # (mock controller simulation)
        action_queue.put((process_id, np.zeros((mock_env.num_actions))))
        observation_queue.get()

        # Send reset signal to continue env stepping
        env_reset_queue.put(
            EnvResetSignal(process_id, False, False, data_entry_idx)
        )

        # Send dummy action and get dummy observation
        # (mock controller simulation)
        action_queue.put((process_id, np.zeros((mock_env.num_actions))))
        observation_queue.get()

        # Add results to successful and failed result lists
        successful_results.append({"success": True})
        failed_result.append({"success": False})

        # Send reset signal signaling worker task completion
        env_reset_queue.put(EnvResetSignal(process_id, False, True, None))

    mock_worker.side_effect = dummy_worker

    results = parallel_grid_search(
        env=mock_env,
        parameter_combinations=[test_combination_params],
        fixed_params=test_fixed_params,
        eval_params=test_eval_params,
        data_driven_cache=test_data_driven_cache,
        drone_state_cache=test_drone_state_cache,
        num_processes=1,
    )

    # Assert that `CtrlEvalStatus` keys are present in the results
    assert CtrlEvalStatus.SUCCESS in results
    assert CtrlEvalStatus.FAILURE in results

    # Assert that each result category contains a list of results
    assert isinstance(results[CtrlEvalStatus.SUCCESS], list)
    assert isinstance(results[CtrlEvalStatus.FAILURE], list)

    # Verify that the mocked results are correctly returned
    assert results[CtrlEvalStatus.SUCCESS][0] == {"success": True}
    assert results[CtrlEvalStatus.FAILURE][0] == {"success": False}
