"""
Evaluate nonlinear data-driven MPC controllers in a parameter grid search.

This module provides functionality for evaluating nonlinear data-driven MPC
controllers based on their position tracking performance.

Each controller is instantiated using a specific set of fixed and combination
parameters from a controller parameter grid search and evaluated through
closed-loop simulation.
"""

import logging
import math
from multiprocessing.sharedctypes import Synchronized
from typing import Any

import numpy as np
import torch.multiprocessing as mp
from direct_data_driven_mpc.nonlinear_data_driven_mpc_controller import (
    NonlinearDataDrivenMPCController,
)

from data_driven_quad_control.envs.hover_env_config import EnvState

from .controller_creation import create_dd_mpc_controller_for_combination
from .isolated_execution import run_in_isolated_process
from .param_grid_search_config import (
    CtrlEvalStatus,
    DDMPCCombinationParams,
    DDMPCEvaluationParams,
    DDMPCFixedParams,
    EnvResetSignal,
    EnvSimInfo,
)
from .resource_usage_logging import (
    log_system_resources,
)

# Retrieve main process logger
logger = logging.getLogger(__name__)


def evaluate_dd_mpc_controller_combination(
    env_idx: int,
    data_entry_idx: int,
    u_N: np.ndarray,
    y_N: np.ndarray,
    initial_drone_state: EnvState,
    env_reset_queue: mp.Queue,
    action_queue: mp.Queue,
    observation_queue: mp.Queue,
    combination_params: DDMPCCombinationParams,
    fixed_params: DDMPCFixedParams,
    eval_params: DDMPCEvaluationParams,
    progress: Synchronized,
) -> tuple[CtrlEvalStatus, dict[str, Any]]:
    """
    Evaluate a data-driven MPC parameter combination based on the position
    tracking performance of its corresponding nonlinear data-driven MPC
    controller.

    This function creates a nonlinear data-driven MPC controller from the
    specified parameters and communicates with the main process (responsible
    for stepping the drone simulation environment) to simulate closed-loop
    control using the instantiated controller.

    The controller is evaluated based on its ability to guide a drone toward a
    series of target positions. A parameter combination is considered
    successful if its corresponding controller commands the drone to move
    toward each setpoint. If the drone's distance to its target increases by
    more than a threshold relative to the initial distance, the evaluation is
    terminated early and the combination is deemed a failure.

    The evaluation is executed in an isolated subprocess to ensure correct
    resource cleanup and to prevent memory leaks caused by CVXPY objects
    (created during controller creation) persisting across runs.

    Args:
        env_idx (int): The environment instance (drone) index selected for
            evaluation.
        data_entry_idx (int): The index of the collected initial input-output
            data entry in the data-driven cache.
        u_N (np.ndarray): An array of shape `(N, m)` representing a
            persistently exciting input sequence used for output data
            collection. `N` is the trajectory length and `m` is the number of
            control inputs.
        y_N (np.ndarray): An array of shape `(N, p)` representing the collected
            system output in response to `u_N`. `N` is the trajectory length
            and `p` is the number of drone system outputs.
        initial_drone_state (EnvState): The initial drone state obtained
            immediately after the initial input-output data collection. Used
            for resetting the `env_idx` environment after each evaluation run
            (evaluation for each target position).
        env_reset_queue (mp.Queue): The reset queue used for sending
            environment reset commands to the main process.
        action_queue (mp.Queue): The action queue used for sending control
            actions to the main process for environment stepping.
        observation_queue (mp.Queue): The observation queue used for receiving
            environment observations from the main process.
        combination_params (DDMPCCombinationParams): A combination of nonlinear
            data-driven MPC parameters selected from the grid search.
        fixed_params (DDMPCFixedParams): The nonlinear data-driven MPC
            parameters that remain fixed across the grid search.
        eval_params (DDMPCEvaluationParams): The parameters that define the
            evaluation procedure for each controller parameter combination in
            the grid search.
        progress (Synchronized): The shared grid search progress tracker.

    Returns:
        tuple[CtrlEvalStatus, dict[str, Any]]: A tuple containing:
            - The controller evaluation status (`CtrlEvalStatus.SUCCESS` or
              `CtrlEvalStatus.FAILURE`).
            - A dictionary with evaluation metrics and context, including
              the controller evaluation average RMSE, number of successful
              runs, and failure reason (if any).
    """
    logger.info(f"[Process {env_idx}] Entered evaluation")
    log_system_resources(one_line=True)

    # Evaluate controller parameter combination with different setpoints
    eval_setpoints = eval_params.eval_setpoints
    required_successful_runs = len(eval_setpoints)
    has_failed = False

    # Define list to store RMSE values for each run
    rmse_values: list[float] = []

    # Initialize object to track the reset status and the closed-loop
    # simulation progress for the `env_idx` environment
    env_sim_info = EnvSimInfo()

    for run_index in range(required_successful_runs):
        # Reset env sim info data to start a new run
        env_sim_info.reset_state = True
        env_sim_info.sim_step_progress = 0

        # Get drone's current position
        initial_drone_pos_tensor = initial_drone_state.base_pos
        initial_drone_pos = initial_drone_pos_tensor.cpu().numpy()
        initial_drone_pos = np.array(initial_drone_pos).reshape(-1, 1)

        # Update setpoint `y_r`
        y_r = eval_setpoints[run_index]

        # Get initial distance to the setpoint position
        initial_distance = float(np.linalg.norm(initial_drone_pos - y_r))

        try:
            # Isolation: Create Nonlinear Data-Driven MPC controller for the
            # current combination and evaluate it using the current env
            logger.info(f"[Process {env_idx}] Running evaluation in isolation")

            rmse = run_in_isolated_process(
                target_func=isolated_controller_evaluation,
                env_idx=env_idx,
                data_entry_idx=data_entry_idx,
                u_N=u_N,
                y_N=y_N,
                y_r=y_r,
                env_reset_queue=env_reset_queue,
                action_queue=action_queue,
                observation_queue=observation_queue,
                combination_params=combination_params,
                fixed_params=fixed_params,
                num_steps=eval_params.eval_time_steps,
                initial_distance=initial_distance,
                max_target_dist_increment=eval_params.max_target_dist_increment,
                env_sim_info=env_sim_info,
            )

            rmse_values.append(rmse)

            # Increment the grid search progress by 1
            # if the evaluation completed successfully
            with progress.get_lock():
                progress.value += 1

        except Exception as e:
            logger.exception(f"[Process {env_idx}] Exception in evaluation")

            has_failed = True

            # Send dummy values to and retrieve data from queues if the
            # environment simulation was interrupted to prevent deadlocks and
            # ensure synchronized communication with the main process
            m = fixed_params.m
            if env_sim_info.sim_step_progress == 0:
                env_reset_queue.put(
                    EnvResetSignal(
                        env_idx=env_idx,
                        reset=False,
                        done=False,
                        data_entry_idx=data_entry_idx,
                    )
                )
                action_queue.put((env_idx, np.zeros(m)))
                observation_queue.get()
            elif env_sim_info.sim_step_progress == 1:
                action_queue.put((env_idx, np.zeros(m)))
                observation_queue.get()

            # Calculate average RMSE of successful tests, if any
            average_RMSE = (
                np.mean(rmse_values) if len(rmse_values) > 0 else np.nan
            )

            # Store results unpacking combination parameters
            failed_result = {
                "n_successful_runs": run_index,
                **combination_params._asdict(),
                "average_RMSE": float(average_RMSE),
                "failure_reason": str(e),
            }

            return (CtrlEvalStatus.FAILURE, failed_result)

        finally:
            logger.info(f"[Process {env_idx}] Evaluation completed")
            log_system_resources(one_line=True)

    if not has_failed:
        # Store results unpacking combination parameters
        result = {
            **combination_params._asdict(),
            "average_RMSE": float(np.mean(rmse_values)),
        }

        return (CtrlEvalStatus.SUCCESS, result)

    # Fallback return to prevent mypy return error
    return (
        CtrlEvalStatus.FAILURE,
        {"error": "Unknown error in controller evaluation."},
    )


def isolated_controller_evaluation(
    env_idx: int,
    data_entry_idx: int,
    u_N: np.ndarray,
    y_N: np.ndarray,
    y_r: np.ndarray,
    env_reset_queue: mp.Queue,
    action_queue: mp.Queue,
    observation_queue: mp.Queue,
    num_steps: int,
    combination_params: DDMPCCombinationParams,
    fixed_params: DDMPCFixedParams,
    initial_distance: float,
    max_target_dist_increment: float,
    env_sim_info: EnvSimInfo,
) -> float:
    # Create Nonlinear Data-Driven MPC controller
    # for the current combination
    logger.info(f"[Process {env_idx}] Isolated: Creating controller")
    log_system_resources(indent_level=1, one_line=True)

    dd_mpc_controller = create_dd_mpc_controller_for_combination(
        u_N=u_N,
        y_N=y_N,
        y_r=y_r,
        combination_params=combination_params,
        fixed_params=fixed_params,
    )

    # Evaluate Nonlinear Data-Driven MPC controller
    logger.info(
        f"[Process {env_idx}] Isolated: Evaluating controller in simulation"
    )

    rmse = sim_nonlinear_dd_mpc_control_loop_parallel(
        env_idx=env_idx,
        data_entry_idx=data_entry_idx,
        env_reset_queue=env_reset_queue,
        action_queue=action_queue,
        observation_queue=observation_queue,
        env_sim_info=env_sim_info,
        dd_mpc_controller=dd_mpc_controller,
        fixed_params=fixed_params,
        num_steps=num_steps,
        initial_distance=initial_distance,
        max_target_dist_increment=max_target_dist_increment,
    )

    return rmse


def sim_nonlinear_dd_mpc_control_loop_parallel(
    env_idx: int,
    data_entry_idx: int,
    env_reset_queue: mp.Queue,
    action_queue: mp.Queue,
    observation_queue: mp.Queue,
    env_sim_info: EnvSimInfo,
    dd_mpc_controller: NonlinearDataDrivenMPCController,
    fixed_params: DDMPCFixedParams,
    num_steps: int,
    initial_distance: float,
    max_target_dist_increment: float | None = None,
) -> float:
    """
    Simulate a closed-loop control using a nonlinear data-driven MPC controller
    in a vectorized environment.

    This function runs a closed-loop simulation of a nonlinear data-driven MPC
    controller for a specified number of time steps. It communicates with the
    main process, which manages the simulation stepping of a vectorized drone
    environment, via multiprocessing queues, sending control actions and
    receiving observations at each simulation step.

    The function evaluates the controller's ability to command a drone toward a
    target position. If the drone's distance to its target increases by more
    than a threshold (`max_target_dist_increment`) relative to the initial
    distance, the evaluation is terminated early.

    Args:
        env_idx (int): The environment instance (drone) index selected for
            evaluation.
        data_entry_idx (int): The index of the collected initial input-output
            data entry in the data-driven cache.
        env_reset_queue (mp.Queue): The reset queue used for sending
            environment reset commands to the main process.
        action_queue (mp.Queue): The action queue used for sending control
            actions to the main process for environment stepping.
        observation_queue (mp.Queue): The observation queue used for receiving
            environment observations from the main process.
        env_sim_info (EnvSimInfo): A dictionary tracking the reset status and
            step progress for each `env_idx` environment.
        dd_mpc_controller (NonlinearDataDrivenMPCController): The nonlinear
            data-driven MPC controller instance used for simulation.
        fixed_params (DDMPCFixedParams): The nonlinear data-driven MPC
            parameters that remain fixed across the grid search.
        num_steps (int): The total number of simulation steps.
        initial_distance (float): The initial distance between the drone and
            the target position.
        max_target_dist_increment (float | None): The maximum allowed increment
            in distance to the target relative to the initial distance. If
            exceeded, the evaluation run is terminated early. If `None`, this
            early termination check will be disabled. Defaults to `None`.

    Returns:
        float: The root mean square error (RMSE) of the drone's position
            tracking performance.
    """
    n = dd_mpc_controller.n
    y_r = dd_mpc_controller.y_r

    # Retrieve fixed parameters
    m = fixed_params.m
    p = fixed_params.p
    n_mpc_step = n if fixed_params.n_n_mpc_step else 1

    # Apply the Nonlinear Data-Driven MPC controller
    u_sys = np.zeros((num_steps, m))
    y_sys = np.zeros((num_steps, p))
    for t in range(0, num_steps, n_mpc_step):
        # Reset sim step progress
        env_sim_info.sim_step_progress = 0

        # Update and solve the Data-Driven MPC problem
        dd_mpc_controller.update_and_solve_data_driven_mpc()

        # Controller closed loop
        for k in range(t, min(t + n_mpc_step, num_steps)):
            # Reset sim step progress
            env_sim_info.sim_step_progress = 0

            # Send env reset signal back to the main process
            env_reset_queue.put(
                EnvResetSignal(
                    env_idx=env_idx,
                    reset=env_sim_info.reset_state,
                    done=False,
                    data_entry_idx=data_entry_idx,
                )
            )
            # Set env reset status to False
            env_sim_info.reset_state = False
            # Advance step progress
            env_sim_info.sim_step_progress += 1

            logger.info(
                f"[Process {env_idx}] Reset state ("
                f"{env_sim_info.reset_state}) sent to queue"
            )

            # Update control input
            n_step = k - t
            optimal_u_step_n = (
                dd_mpc_controller.get_optimal_control_input_at_step(
                    n_step=n_step
                )
            )
            u_sys[k, :] = optimal_u_step_n

            # Send action (control input) back to the main process
            action_queue.put((env_idx, u_sys[k, :]))
            env_sim_info.sim_step_progress += 1  # Advance sim step progress

            logger.info(
                f"[Process {env_idx}] Action sent to queue: {u_sys[k, :]}"
            )

            # Get observations from vectorized environment
            env_observations = observation_queue.get()

            # Retrieve system output from observations
            # for the current env
            y_sys[k, :] = env_observations[env_idx]

            logger.info(
                f"[Process {env_idx}] Observation received from queue: "
                f"{y_sys[k, :]}"
            )

            # Update input-output measurements online
            du_current = dd_mpc_controller.get_du_value_at_step(n_step=n_step)
            dd_mpc_controller.store_input_output_measurement(
                u_current=u_sys[k, :],
                y_current=y_sys[k, :],
                du_current=du_current,
            )

            if max_target_dist_increment is not None:
                # Stop simulation by raising a ValueError if the distance from
                # the drone to its setpoint position has increased by more than
                # `max_target_dist_increment` compared to the initial distance
                current_distance = np.linalg.norm(
                    y_sys[k, :].reshape(-1, 1) - y_r
                )

                if (
                    current_distance - initial_distance
                    > max_target_dist_increment
                ):
                    logger.warning(
                        f"[Process {env_idx}] Drone moved too far from its "
                        "target. Raising ValueError to terminate evaluation."
                    )

                    raise ValueError(
                        "Drone moved away from its target by more than "
                        f"{max_target_dist_increment} relative to its initial "
                        "distance."
                    )

    # Calculate target position tracking RMSE
    rmse = math.sqrt(np.mean(np.sum((y_sys - y_r.T) ** 2, axis=1)))

    return rmse
