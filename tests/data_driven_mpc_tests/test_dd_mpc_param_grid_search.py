import os
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.grid_search_param_loader import (  # noqa: E501
    load_dd_mpc_grid_search_params,
)
from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.initial_data_collection_cache import (  # noqa: E501
    cache_initial_data_and_states,
)
from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.parallel_grid_search import (  # noqa: E501
    parallel_grid_search,
)
from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.param_grid_search_config import (  # noqa: E501
    CtrlEvalStatus,
    DDMPCCombinationParams,
)
from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.results_writer import (  # noqa: E501
    write_results_to_file,
)
from data_driven_quad_control.envs.hover_env_config import (
    EnvActionType,
)
from data_driven_quad_control.utilities.drone_environment import (
    create_env,
    get_current_env_state,
)
from data_driven_quad_control.utilities.drone_tracking_controller import (
    create_drone_tracking_controller,
    hover_at_target,
)


def test_dd_mpc_param_grid_search(
    dummy_env_cfg: dict[str, Any],
    dummy_obs_cfg: dict[str, Any],
    dummy_reward_cfg: dict[str, Any],
    dummy_command_cfg: dict[str, Any],
    test_grid_search_params_path: str,
    tmp_path: Path,
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

    # Reset env
    env.reset()

    # Load parameters for the data-driven MPC controller parameter grid search
    m = env.num_actions  # Number of inputs
    p = 3  # Number of outputs (drone position)
    init_data_collection_params, fixed_params, eval_params, param_grid = (
        load_dd_mpc_grid_search_params(
            m=m, p=p, config_path=test_grid_search_params_path
        )
    )

    # Initial input-output data collection
    base_env_idx = 0

    # Create stabilization controller for initial input-output data collection
    stabilizing_controller = create_drone_tracking_controller(env=env)

    # Define hover target
    target_pos = torch.tensor(
        init_data_collection_params.init_hover_pos,
        device=env.device,
        dtype=torch.float,
    )
    target_yaw = torch.tensor([0.0], device=env.device, dtype=torch.float)

    # Command drones to hover at target
    target_pos = target_pos.expand(env.num_envs, -1)
    target_yaw = target_yaw.expand(env.num_envs)
    hover_at_target(
        env=env,
        tracking_controller=stabilizing_controller,
        target_pos=target_pos,
        target_yaw=target_yaw,
        ctbr_controller=None,
    )

    # Save hovering drone state
    init_hovering_state = get_current_env_state(env=env, env_idx=base_env_idx)

    # Create data-driven cache:
    # Collect initial input-output data and save drone env states
    np_random = np.random.default_rng(0)
    data_driven_cache, drone_state_cache = cache_initial_data_and_states(
        env=env,
        base_env_idx=base_env_idx,
        stabilizing_controller=stabilizing_controller,
        target_pos=target_pos,
        target_yaw=target_yaw,
        init_hovering_state=init_hovering_state,
        init_data_collection_params=init_data_collection_params,
        param_grid=param_grid,
        fixed_params=fixed_params,
        np_random=np_random,
    )

    # Perform grid search in parallel
    parameter_combinations = [
        DDMPCCombinationParams(*combination)
        for combination in product(*param_grid)
    ]

    num_processes = 2
    with torch.no_grad():
        results = parallel_grid_search(
            env=env,
            parameter_combinations=parameter_combinations,
            fixed_params=fixed_params,
            eval_params=eval_params,
            data_driven_cache=data_driven_cache,
            drone_state_cache=drone_state_cache,
            num_processes=num_processes,
        )

    # Write results to a file
    test_output_dir = str(tmp_path)
    elapsed_time = 123.123
    output_file = write_results_to_file(
        output_dir=test_output_dir,
        elapsed_time=elapsed_time,
        init_data_collection_params=init_data_collection_params,
        fixed_params=fixed_params,
        eval_params=eval_params,
        param_grid=param_grid,
        results=results,
    )

    # Assert output file exists (report file was written)
    assert os.path.isfile(output_file)

    # Read contents from output file
    with open(output_file) as f:
        output = f.read()

    # Verify file structure
    assert "Grid search complete in 0h 2m 3.12s." in output
    assert "Initial data collection parameters:" in output
    assert "Fixed parameters:" in output
    assert "Evaluation parameters:" in output
    assert "Grid Search conducted over the following parameters:" in output
    assert "Successful Results (2/2):" in output
    assert "Failed Results (0/2):" in output

    # Check expected parameter values
    assert "n_n_mpc_step: False" in output
    assert "lamb_sigma_s: [1000.0]" in output

    # Check expected result values
    assert "N=40, n=2, L=5, lamb_alpha=10.0" in output
    assert "average_RMSE=" in output
    assert "No results for any parameter combination." in output

    # Verify successful results were sorted by RMSE in ascending order
    first_result_line = output.split("Successful Results")[1].splitlines()[1]
    min_result_rmse = min(
        result["average_RMSE"] for result in results[CtrlEvalStatus.SUCCESS]
    )
    assert f"average_RMSE={min_result_rmse}" in first_result_line

    # Sanity checks
    assert output.count("average_RMSE=") == 2
    assert output.count("N=") == 2
