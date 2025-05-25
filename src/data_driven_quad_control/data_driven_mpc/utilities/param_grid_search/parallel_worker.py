"""
Nonlinear data-driven MPC parameter grid search - parallel worker

This module implements the parallel worker process used for evaluating
controller parameter combinations during a nonlinear data-driven MPC parameter
grid search.

Each worker independently evaluates multiple unique controller parameter
combinations from the grid and stores the results in shared lists based on
their outcome.

Workers terminate when no tasks remain or upon exception, sending a termination
signal to the main process to report task completion.
"""

import logging
from multiprocessing.managers import ListProxy
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Lock
from queue import Empty

import numpy as np
import torch.multiprocessing as mp

from .controller_evaluation import evaluate_dd_mpc_controller_combination
from .param_grid_search_config import (
    CtrlEvalStatus,
    DataDrivenCache,
    DDMPCCombinationParams,
    DDMPCEvaluationParams,
    DDMPCFixedParams,
    EnvResetSignal,
)

# Retrieve main process logger
logger = logging.getLogger(__name__)


def worker_data_driven_mpc(
    process_id: int,
    env_reset_queue: mp.Queue,
    action_queue: mp.Queue,
    observation_queue: mp.Queue,
    combination_params_queue: mp.Queue,
    successful_results: ListProxy,
    failed_result: ListProxy,
    lock: Lock,
    progress: Synchronized,
    data_driven_cache: DataDrivenCache,
    fixed_params: DDMPCFixedParams,
    eval_params: DDMPCEvaluationParams,
) -> None:
    """
    Parallel worker process that independently evaluates nonlinear data-driven
    MPC controller parameter combinations during a parallel grid search.

    This worker continuously retrieves controller parameter combinations from
    a shared queue, evaluates them in simulation via synchronous communication
    with the main process, and stores the results in shared result lists based
    on the evaluation outcome.

    The worker task terminates when the parameter combination queue is empty or
    an exception occurs. In either case, the worker sends a termination signal
    to the main process via the environment reset queue to report task
    completion.

    Args:
        process_id (int): The ID of this process, assigned by the main process.
        env_reset_queue (mp.Queue): The reset queue used for sending
            environment reset commands to the main process.
        action_queue (mp.Queue): The action queue used for sending control
            actions to the main process for environment stepping.
        observation_queue (mp.Queue): The observation queue used for receiving
            environment observations from the main process.
        combination_params_queue (mp.Queue): The queue that contains the
            controller parameter combinations to evaluate.
        successful_results (ListProxy): A shared list used for storing result
            dictionaries corresponding to successful controller parameter
            combinations.
        failed_result (ListProxy): A shared list used for storing result
            dictionaries corresponding to failed controller parameter
            combinations.
        lock (Lock): A lock used for synchronizing access to result lists.
        progress (Synchronized): The shared grid search progress tracker.
        data_driven_cache (DataDrivenCache): A `DataDrivenCache` object
            containing initial input-output data and their associated drone
            states for a series of initial input-output trajectory length (`N`)
            values. Each drone state must be allocated on the CPU for use in
            multiprocessing queues.
        fixed_params (DDMPCFixedParams): The nonlinear data-driven MPC
            parameters that remain fixed across the grid search.
        eval_params (DDMPCEvaluationParams): The parameters that define the
            evaluation procedure for each controller parameter combination in
            the grid search.
    """
    logger.info(f"[Worker] Process {process_id} started")

    # Loop until there are no more combinations to test
    while True:
        logger.info(f"[Worker] Entered process {process_id}")

        try:
            # Get the next Nonlinear Data-Driven MPC
            # parameter combination with a timeout
            combination_params: DDMPCCombinationParams = (
                combination_params_queue.get(timeout=1)
            )
            logger.info(
                f"[Worker] Process {process_id} evaluating params: "
                f"{combination_params._asdict()}."
            )

            # Get `N` from combination params
            N = combination_params.N

            # Get available entry indices for this trajectory length
            init_data_entry_indices = data_driven_cache.N_to_entry_indices[N]
            num_collections_per_N = len(init_data_entry_indices)

            logger.info(
                f"[Worker] Process {process_id} evaluating for "
                f"{num_collections_per_N} collected data entries"
            )

            # Evaluate the controller for every collected initial data entry
            # for this trajectory length `N`
            combination_succeeded = True
            average_rmse_from_runs = []
            total_n_successful_runs = 0
            num_setpoints_per_run = len(eval_params.eval_setpoints)

            for i, entry_idx in enumerate(init_data_entry_indices):
                logger.info(
                    f"[Worker] Process {process_id} evaluating using initial "
                    f"data {i + 1}/{num_collections_per_N} (entry index = "
                    f"{entry_idx})"
                )

                # Get initial input-output measurements
                u_N = data_driven_cache.u_N[entry_idx]
                y_N = data_driven_cache.y_N[entry_idx]

                # Get initial drone state
                initial_drone_state = data_driven_cache.drone_state[entry_idx]

                # Evaluate the combination
                logger.info(
                    f"[Worker] Entered evaluation in process {process_id}"
                )

                status, result = evaluate_dd_mpc_controller_combination(
                    env_idx=process_id,
                    data_entry_idx=entry_idx,
                    u_N=u_N,
                    y_N=y_N,
                    initial_drone_state=initial_drone_state,
                    env_reset_queue=env_reset_queue,
                    action_queue=action_queue,
                    observation_queue=observation_queue,
                    combination_params=combination_params,
                    fixed_params=fixed_params,
                    eval_params=eval_params,
                )

                # Store the average RMSE from the current evaluation run
                average_rmse_from_runs.append(result["average_RMSE"])

                # Mark the combination as failed if any run fails
                # and stop the evaluation
                if status == CtrlEvalStatus.FAILURE:
                    combination_succeeded = False
                    total_n_successful_runs += result["n_successful_runs"]
                    break
                else:
                    total_n_successful_runs += num_setpoints_per_run

            # Calculate the overall average RMSE across all runs
            eval_average_rmse = (
                np.nanmean(average_rmse_from_runs)
                if len(average_rmse_from_runs) > 0
                else np.nan
            )

            # Overwrite the average RMSE in the evaluation result
            result["average_RMSE"] = float(eval_average_rmse)

            # Store results
            with lock:
                if combination_succeeded:
                    successful_results.append(result)
                    logger.info(
                        f"[Worker] Process {process_id} succeeded with "
                        f"params: {combination_params._asdict()}."
                    )
                else:
                    # Overwrite the number of successful runs in
                    # the evaluation result
                    result["n_successful_runs"] = total_n_successful_runs

                    failed_result.append(result)
                    logger.error(
                        f"[Worker] Process {process_id} failed: {result}."
                    )

        except Empty:
            logger.info(f"[Worker] Process {process_id} found no more tasks.")
            break

        except Exception as e:
            logger.exception(f"[Worker] Exception in process {process_id}")

            with lock:
                failed_result.append(
                    {
                        "params": combination_params,
                        "error": str(e),
                    }
                )

        # Update global progress bar
        with progress.get_lock():
            progress.value += 1

    # Signal task completion to main process
    env_reset_queue.put(
        EnvResetSignal(
            env_idx=process_id, reset=False, done=True, data_entry_idx=None
        )
    )

    logger.info(f"[Worker] ----- Process {process_id} finished -----")
