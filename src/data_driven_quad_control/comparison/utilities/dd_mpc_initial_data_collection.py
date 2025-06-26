"""
Construct initialization data for data-driven MPC controllers.

This module provides functionality to construct the initialization data for a
nonlinear data-driven MPC controller. The initialization data includes the
controller configuration parameters and an initial input-output data trajectory
collected in simulation.

During data collection, all drones in simulation are stabilized at a fixed
target position, while the drone corresponding to the data-driven MPC
controller is used for data collection.
"""

import torch
from direct_data_driven_mpc.utilities.controller.controller_params import (
    NonlinearDataDrivenMPCParams,
)
from numpy.random import Generator

from data_driven_quad_control.controllers.tracking.tracking_controller import (
    DroneTrackingController,
)
from data_driven_quad_control.data_driven_mpc.utilities.drone_initial_data_collection import (  # noqa: E501
    collect_initial_input_output_data,
)
from data_driven_quad_control.envs.hover_env import HoverEnv
from data_driven_quad_control.utilities.drone_tracking_controller import (
    hover_at_target,
)

from .controller_comparison_config import DDMPCControllerInitData


def get_data_driven_mpc_controller_init_data(
    env: HoverEnv,
    dd_mpc_env_idx: int,
    dd_mpc_controller_config: NonlinearDataDrivenMPCParams,
    init_hover_pos: torch.Tensor,
    stabilizing_controller: DroneTrackingController,
    np_random: Generator,
    verbose: int = 0,
) -> DDMPCControllerInitData:
    """
    Construct the initialization data for a nonlinear data-driven MPC
    controller, consisting of the controller configuration parameters and an
    initial input-output data trajectory collected in simulation.

    This function collects initial input-output data for a data-driven
    controller using an individual drone (`dd_mpc_env_idx`) from a vectorized
    drone environment, while stabilizing the remaining ones using a stabilizing
    controller. The collected data, along with the controller configuration, is
    returned as the controller initialization data.

    Args:
        env (HoverEnv): The vectorized drone environment.
        dd_mpc_env_idx (int): The index of the drone used for data collection.
        dd_mpc_controller_config (NonlinearDataDrivenMPCParams): The nonlinear
            data-driven MPC controller configuration parameters.
        init_hover_pos (torch.Tensor): The initial hover position at which all
            drones are stabilized during data collection.
        stabilizing_controller (DroneTrackingController): A drone tracking
            controller used for stabilization.
        np_random (Generator): A Numpy random number generator used for
            generating persistently exciting inputs.
        verbose (int): The verbosity level: 0 = no output, 1 = minimal output,
            2 = detailed output.

    Returns:
        DDMPCControllerInitData: The nonlinear data-driven MPC controller
            initialization data containing the controller configuration
            parameters and the collected initial input-output data.
    """
    m = 3  # Number of inputs (considering CTBR_FIXED_YAW env actions)
    p = 3  # Number of outputs (drone position)

    # Command drones to hover at initial target
    # for initial input-output data collection
    if verbose:
        print(
            "  Stabilizing drones at the initial position "
            f"{init_hover_pos.tolist()}"
        )

    target_pos = init_hover_pos.expand(env.num_envs, -1)
    target_yaw = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.float
    )
    hover_at_target(
        env=env,
        tracking_controller=stabilizing_controller,
        target_pos=target_pos,
        target_yaw=target_yaw,
        ctbr_controller=None,
    )

    # Collect initial input-output measurement with a
    # generated persistently exciting input
    if verbose:
        print(
            "  Collecting initial input-output data for data-driven MPC "
            "controller"
        )

    u_N, y_N = collect_initial_input_output_data(
        env=env,
        base_env_idx=dd_mpc_env_idx,
        stabilizing_controller=stabilizing_controller,
        target_pos=target_pos,
        target_yaw=target_yaw,
        input_bounds=dd_mpc_controller_config["U"],
        u_range=dd_mpc_controller_config["u_range"],
        N=dd_mpc_controller_config["N"],
        m=m,
        p=p,
        np_random=np_random,
    )

    return DDMPCControllerInitData(dd_mpc_controller_config, u_N, y_N)
