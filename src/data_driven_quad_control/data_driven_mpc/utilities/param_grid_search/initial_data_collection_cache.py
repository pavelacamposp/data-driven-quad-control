"""
Collect initial data and cache states for a data-driven MPC grid search.

This module provides functionality for collecting initial input-output
trajectory data required for initializing data-driven MPC controllers, as well
as drone environment states used for resetting drone environments in nonlinear
data-driven MPC parameter grid searches.
"""

import numpy as np
import torch
from numpy.random import Generator

from data_driven_quad_control.controllers.tracking.tracking_controller import (
    DroneTrackingController,
)
from data_driven_quad_control.envs.hover_env import HoverEnv
from data_driven_quad_control.envs.hover_env_config import EnvState
from data_driven_quad_control.utilities.drone_environment import (
    get_current_env_state,
    restore_env_from_state,
)

from ..drone_initial_data_collection import collect_initial_input_output_data
from .param_grid_search_config import (
    DataDrivenCache,
    DDMPCEvaluationParams,
    DDMPCFixedParams,
    DDMPCInitialDataCollectionParams,
    DDMPCParameterGrid,
)


def cache_initial_data_and_states(
    env: HoverEnv,
    base_env_idx: int,
    stabilizing_controller: DroneTrackingController,
    target_pos: torch.Tensor,
    target_yaw: torch.Tensor,
    init_hovering_state: EnvState,
    init_data_collection_params: DDMPCInitialDataCollectionParams,
    fixed_params: DDMPCFixedParams,
    eval_params: DDMPCEvaluationParams,
    param_grid: DDMPCParameterGrid,
    np_random: Generator,
    verbose: int = 0,
) -> tuple[DataDrivenCache, dict[int, EnvState]]:
    """
    Cache initial input-output data and drone states for each `N` value in a
    nonlinear data-driven MPC parameter grid.

    This function collects initial input-output measurement trajectories
    and their corresponding drone environment states obtained after collection
    for a set of trajectory lengths (`N`) defined in the parameter grid. For
    each `N`, the drone is stabilized at a target position and yaw using a
    stabilizing controller while a persistently exciting input is applied. The
    input-output data and resulting drone states are cached for reuse in
    controller creation and environment resets in the data-driven MPC parameter
    grid search.

    Data collection is repeated `eval_params.num_collections_per_N` times per
    `N` to cache different initial data entries.

    Args:
        env (HoverEnv): The vectorized drone environment.
        base_env_idx (int): The index of the drone environment instance used
            for data collection.
        stabilizing_controller (DroneTrackingController): A drone tracking
            controller used for stabilizing the drone during data collection.
        target_pos (torch.Tensor): The target position for stabilization, with
            shape (`num_envs`, 3).
        target_yaw (torch.Tensor): The target yaw for stabilization, with shape
            (`num_envs`,).
        init_hovering_state (EnvState): The drone environment state saved after
            stabilizing the drone at the target position and yaw (`target_pos`
            and `target_yaw`). Used to reset the drone environment before data
            collection.
        init_data_collection_params (DDMPCInitialDataCollectionParams): The
            parameters for collecting initial input-output data.
        fixed_params (DDMPCFixedParams): The nonlinear data-driven MPC
            parameters that remain fixed across the grid search.
        eval_params (DDMPCEvaluationParams): The parameters that define the
            evaluation procedure for each controller parameter combination in
            the grid search.
        param_grid (DDMPCParameterGrid): The parameter grid defining the list
            of trajectory lengths (`N`) for which data should be collected.
        np_random (Generator): A Numpy random number generator used for
            generating persistently exciting inputs.
        verbose (int): The verbosity level: 0 = no output, 1 = minimal output,
            2 = detailed output. Defaults to 0.

    Returns:
        tuple[DataDrivenCache, dict[int, EnvState]]:
            - A `DataDrivenCache` object containing input-output data and
              their associated drone states (allocated on the CPU for use in
              multiprocessing queues) for each `N` value. Entries are indexed
              by a flat `entry_index`, and the `N_to_entry_indices` field maps
              each trajectory length (`N`) to the corresponding list of entry
              indices.
            - A dictionary mapping generation indices to their corresponding
              `EnvState` (allocated on the environment's device), for use in
              environment resets.
    """
    # Initialize state-related variables
    N_to_entry_indices: dict[int, list[int]] = {}
    N_cache: dict[int, int] = {}
    u_N_cache: dict[int, np.ndarray] = {}
    y_N_cache: dict[int, np.ndarray] = {}
    drone_state_cache: dict[int, EnvState] = {}
    drone_state_cache_cpu: dict[int, EnvState] = {}

    N_values = param_grid.N
    num_collections_per_N = eval_params.num_collections_per_N
    total_N_values = len(N_values)
    for N_value_index, N in enumerate(N_values):
        if verbose:
            print(
                f"  [{N_value_index + 1}/{total_N_values}] Collecting "
                f"initial input-output data and saving drone states (N = {N})"
            )

        N_to_entry_indices[N] = []
        for gen_index in range(num_collections_per_N):
            if verbose:
                print(
                    f"    - Data collection entry {gen_index + 1} of "
                    f"{num_collections_per_N}"
                )

            # Reset drone state to initial hovering state
            if verbose > 1:
                print(
                    "        Resetting drone state to initial hovering state"
                )

            restore_env_from_state(
                env=env, env_idx=base_env_idx, saved_state=init_hovering_state
            )

            # Reset stabilizing controller
            stabilizing_controller.reset()

            # Collect initial input-output measurement with a generated
            # persistently exciting input while stabilizing the drone at a
            # fixed target position using a stabilizing controller
            if verbose > 1:
                print("        Collecting initial input-output data")

            u_N, y_N = collect_initial_input_output_data(
                env=env,
                base_env_idx=base_env_idx,
                stabilizing_controller=stabilizing_controller,
                target_pos=target_pos,
                target_yaw=target_yaw,
                input_bounds=fixed_params.U,
                u_range=init_data_collection_params.u_range,
                N=N,
                m=fixed_params.m,
                p=fixed_params.p,
                np_random=np_random,
            )

            # Calculate data entry index
            entry_index = N_value_index * num_collections_per_N + gen_index

            # Store input-output data
            N_to_entry_indices[N].append(entry_index)
            N_cache[entry_index] = N
            u_N_cache[entry_index] = u_N
            y_N_cache[entry_index] = y_N

            # Get current env state tensor dictionary
            drone_state_dict_gpu = get_current_env_state(
                env=env, env_idx=base_env_idx
            )
            # Move state dictionary from GPU to CPU so it
            # can be used in multiprocessing
            drone_state_dict_cpu = drone_state_dict_gpu.to_cpu()

            # Store drone states
            if verbose > 1:
                print("        Saving drone state")

            drone_state_cache[entry_index] = drone_state_dict_gpu
            drone_state_cache_cpu[entry_index] = drone_state_dict_cpu

    data_driven_cache = DataDrivenCache(
        N_to_entry_indices=N_to_entry_indices,
        N=N_cache,
        u_N=u_N_cache,
        y_N=y_N_cache,
        drone_state=drone_state_cache_cpu,
    )

    return data_driven_cache, drone_state_cache
