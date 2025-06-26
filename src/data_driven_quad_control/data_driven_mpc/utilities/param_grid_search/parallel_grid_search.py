"""
Run the main process of a nonlinear data-driven MPC parameter grid search.

This module implements the main process responsible for managing parallel
grid searches over nonlinear data-driven MPC controller parameter combinations
using the `HoverEnv` vectorized drone environment and `torch.multiprocessing`.

The main process spawns multiple parallel worker processes to evaluate
controller parameter combinations independently. It also manages the stepping
of the vectorized drone environment and the drone environment state resets
required between evaluation runs.

Communication with workers occurs via multiprocessing queues in synchronized
loops, preventing deadlocks.
"""

import logging
from typing import Any

import psutil
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from data_driven_quad_control.envs.hover_env import HoverEnv
from data_driven_quad_control.envs.hover_env_config import EnvState
from data_driven_quad_control.utilities.drone_environment import (
    restore_env_from_state,
)
from data_driven_quad_control.utilities.math_utils import linear_interpolate

from .parallel_worker import (
    worker_data_driven_mpc,
)
from .param_grid_search_config import (
    CtrlEvalStatus,
    DataDrivenCache,
    DDMPCCombinationParams,
    DDMPCEvaluationParams,
    DDMPCFixedParams,
    EnvResetSignal,
)
from .resource_usage_logging import (
    log_system_resources,
)

# Retrieve main process logger
logger = logging.getLogger(__name__)


def parallel_grid_search(
    env: HoverEnv,
    num_processes: int,
    parameter_combinations: list[DDMPCCombinationParams],
    fixed_params: DDMPCFixedParams,
    eval_params: DDMPCEvaluationParams,
    data_driven_cache: DataDrivenCache,
    drone_state_cache: dict[int, EnvState],
) -> dict[CtrlEvalStatus, list[dict[str, Any]]]:
    """
    Execute a parallel grid search over nonlinear data-driven MPC controller
    parameter combinations using a vectorized drone environment.

    This function implements the main process of the parallel grid search. It
    spawns multiple parallel worker processes to independently evaluate
    controller parameter combinations from the grid.

    It also manages the stepping of the vectorized environment, communicating
    synchronously with the workers via multiprocessing queues to receive
    control actions and environment reset signals, and to send observations.
    This architecture is necessary because the vectorized environment requires
    batched actions at each step, simulating all drones simultaneously.

    Drone environment states are reset using cached states to ensure that the
    data-driven MPC controllers created during evaluation use consistent
    initial input-output data.

    Args:
        env (HoverEnv): The vectorized drone environment.
        num_processes (int): The number of parallel worker processes to spawn.
        parameter_combinations (list[DDMPCCombinationParams]): A list of
            nonlinear data-driven MPC parameter combinations to evaluate.
        fixed_params (DDMPCFixedParams): The nonlinear data-driven MPC
            parameters that remain fixed across the grid search.
        eval_params (DDMPCEvaluationParams): The parameters that define the
            evaluation procedure for each controller parameter combination in
            the grid search.
        data_driven_cache (DataDrivenCache): A `DataDrivenCache` object
            containing initial input-output data and their associated drone
            states for a series of initial input-output trajectory length (`N`)
            values. Each drone state must be allocated on the CPU for use in
            multiprocessing queues.
        drone_state_cache (dict[int, EnvState]): A dictionary that maps initial
            input-output trajectory length (`N`) values to drone states for use
            in environment resets. Each drone state must be allocated on the
            environment's device.

    Returns:
        dict[CtrlEvalStatus, list[dict[str, Any]]]: A dictionary mapping
            evaluation statuses (`CtrlEvalStatus.SUCCESS`,
            `CtrlEvalStatus.FAILURE`) to lists of result dictionaries, each
            containing evaluation metrics and context.
    """
    logger.info(
        f"Starting parallel grid search with {num_processes} processes."
    )
    # Log system resources before starting multiprocessing
    log_system_resources(indent_level=1)

    # Retrieve environment parameters
    env_action_bounds = env.action_bounds  # Env action bounds used for
    # control action normalization

    # Create a queue for parameter combinations
    combination_params_queue: mp.Queue = mp.Queue()
    for params in parameter_combinations:
        combination_params_queue.put(params)

    # Create multiprocessing queues for synchronous
    # communication with the vectorized environment
    env_reset_queue: mp.Queue = mp.Queue()
    action_queue: mp.Queue = mp.Queue()
    observation_queue: mp.Queue = mp.Queue()

    # Shared memory structures
    manager = mp.Manager()
    successful_results = manager.list()
    failed_result = manager.list()
    lock = mp.Lock()  # Lock for synchronizing access
    global_progress = mp.Value("i", 0)  # Shared progress tracker

    # Create and start worker processes
    processes: list[mp.Process] = []
    for process_id in range(num_processes):
        logger.info(f"[MAIN] Created {process_id + 1} processes")

        p = mp.Process(
            target=worker_data_driven_mpc,
            args=(
                process_id,
                env_reset_queue,
                action_queue,
                observation_queue,
                combination_params_queue,
                successful_results,
                failed_result,
                lock,
                global_progress,
                data_driven_cache,
                fixed_params,
                eval_params,
            ),
        )
        processes.append(p)
        p.start()

    # Step environment in the main process
    logger.info("[MAIN] Started env stepping")

    total_combinations = len(parameter_combinations)
    total_eval_runs = (
        total_combinations
        * len(eval_params.eval_setpoints)
        * eval_params.num_collections_per_N
    )
    done_processes = 0  # Track how many processes have finished
    action_buffer = torch.zeros(
        (env.num_envs, env.num_actions), device=env.device, dtype=torch.float
    )

    with tqdm(total=total_eval_runs) as global_pbar:
        while any(p.is_alive() for p in processes):
            # Update global progress bar
            with lock:
                update_global_progress_bar(
                    global_progres_bar=global_pbar,
                    global_progress_value=global_progress.value,
                    n_successful_results=len(successful_results),
                    total_combinations=total_combinations,
                )

            # Update number of active processes
            active_processes = num_processes - done_processes

            logger.info(
                f"[MAIN] Done: {done_processes}  Active: {active_processes}"
            )

            # Handle environment reset
            logger.info("[MAIN] Reached env reset point")

            for _ in range(active_processes):
                reset_signal: EnvResetSignal = env_reset_queue.get()

                # Handle done environment
                if reset_signal.done:
                    done_processes += 1  # Increment finished process count
                    active_processes -= 1  # Decrement active process count

                    logger.info(
                        f"[MAIN] Process {reset_signal.env_idx} finished"
                    )

                    continue

                # Restore env_idx drone state to the state immediately after
                # the initial input-output measurement for parameter N
                if reset_signal.reset:
                    data_entry_idx = reset_signal.data_entry_idx
                    assert data_entry_idx is not None
                    initial_drone_state = drone_state_cache[data_entry_idx]
                    restore_env_from_state(
                        env=env,
                        env_idx=reset_signal.env_idx,
                        saved_state=initial_drone_state,
                    )

                    logger.info(
                        "[MAIN] Restored initial state for Process "
                        f"{reset_signal.env_idx}"
                    )

            logger.info("[MAIN] Reached env step point")

            # Get actions from action queue for each process
            action_buffer.zero_()
            for _ in range(active_processes):
                env_idx, action = action_queue.get()

                logger.info(
                    f"[MAIN] Action received for Process {env_idx}: {action}"
                )

                # Convert action array to tensor
                action = torch.tensor(action, device=env.device).unsqueeze(0)

                # Store action in action buffer at its corresponding env idx
                action_buffer[env_idx] = action

            # Calculate env action by scaling actions to a [-1, 1] range
            action_buffer = linear_interpolate(
                x=action_buffer,
                x_min=env_action_bounds[:, 0],
                x_max=env_action_bounds[:, 1],
                y_min=-1,
                y_max=1,
            )

            # Step environment only if there are active processes
            if active_processes > 0:
                # Step using batched actions
                env.step(action_buffer)

                # Get environment observations
                # Note: We only "observe" the non-normalized drone's
                # base position, which is the system output for the
                # Data-Driven MPC control system
                observation = env.get_pos().cpu().numpy()

                logger.info("[MAIN] Reached env observation sending point")

                # Send observations to workers
                for _ in range(active_processes):
                    observation_queue.put(observation)

                logger.info(f"[MAIN] Sent {active_processes} observations")

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Close global progress bar
    global_pbar.close()

    logger.info("[MAIN] Parallel Grid Search finished")
    log_system_resources()  # Log system usage after processing

    return {
        CtrlEvalStatus.SUCCESS: list(successful_results),
        CtrlEvalStatus.FAILURE: list(failed_result),
    }


def get_available_memory_percent() -> float:
    total_memory = psutil.virtual_memory().total / (1024 * 1024)
    available_memory = psutil.virtual_memory().available / (1024 * 1024)
    available_memory_percent = (available_memory / total_memory) * 100

    return available_memory_percent


def update_global_progress_bar(
    global_progres_bar: tqdm,
    global_progress_value: int,
    n_successful_results: int,
    total_combinations: int,
) -> None:
    global_progres_bar.n = global_progress_value
    available_mem_percent = get_available_memory_percent()
    global_progres_bar.desc = (
        f"Progress: {n_successful_results}/{total_combinations} "
        "successful combinations | Available RAM: "
        f"{available_mem_percent:2.2f}%"
    )

    global_progres_bar.refresh()
