"""
Initial input-output data collection for data-driven MPC controllers

This module provides functionality to collect initial input-output data using
a generated persistently exciting input, which is required for creating
nonlinear data-driven MPC controllers. It is designed to work with the
`HoverEnv` environment in Genesis.
"""

import numpy as np
import torch
import yaml
from direct_data_driven_mpc.utilities.controller.controller_params import (
    NonlinearDataDrivenMPCParams,
)
from direct_data_driven_mpc.utilities.models.nonlinear_model import (
    NonlinearSystem,
)
from numpy.random import Generator

from data_driven_quad_control.controllers.tracking.tracking_controller import (
    DroneTrackingController,
)
from data_driven_quad_control.controllers.tracking.tracking_controller_config import (  # noqa: E501
    TrackingCtrlDroneState,
)
from data_driven_quad_control.envs.config.hover_env_config import EnvActionType
from data_driven_quad_control.envs.hover_env import HoverEnv
from data_driven_quad_control.utilities.math_utils import (
    linear_interpolate,
    yaw_to_quaternion,
)


def collect_initial_input_output_data(
    env: HoverEnv,
    drone_system_model: NonlinearSystem,
    base_env_idx: int,
    dd_mpc_config: NonlinearDataDrivenMPCParams,
    stabilizing_controller: DroneTrackingController,
    target_pos: torch.Tensor,
    target_yaw: torch.Tensor,
    np_random: Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect initial input-output data from a single drone in a vectorized
    environment for use in data-driven MPC controller creation.

    During collection, the drone is stabilized at a specified target position
    and yaw using a stabilizing controller. A persistently exciting input,
    generated based on the data-driven MPC controller parameters, is added as
    a perturbation to the commands of the stabilizing controller. This ensures
    effective output data collection while maintaining drone stability.

    As a result, the actual input used for collecting output data is the sum of
    the stabilizing controller commands and the persistently exciting input.

    Args:
        env (HoverEnv): The vectorized drone environment.
        drone_system_model (NonlinearSystem): The drone nonlinear system. It is
            a wrapper around the environment to interface with the
            `direct-data-driven-mpc` package.
        base_env_idx (int): The environment instance (drone) index used for
            data collection.
        dd_mpc_config (NonlinearDataDrivenMPCParams): A dictionary containing
            parameters for configuring a nonlinear data-driven MPC controller.
        stabilizing_controller (DroneTrackingController): A drone tracking
            controller used for stabilizing the drone during data collection.
        target_pos (torch.Tensor): The target position for stabilization, with
            shape (`num_envs`, 3).
        target_yaw (torch.Tensor): The target yaw for stabilization, with shape
            (`num_envs`).
        np_random (Generator): A Numpy random number generator for generating
            the persistently exciting input and random noise for the system's
            output.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two arrays: a
            persistently exciting input and the system's output response.
            The input array has shape `(N, m)` and the output array has shape
            `(N, p)`, where `N` is the trajectory length, `m` is the number of
            control inputs, and `p` is the number of drone system outputs.
    """
    # Retrieve model parameters
    m = drone_system_model.m  # Number of control inputs
    p = drone_system_model.p  # Number of system outputs
    eps_max = drone_system_model.eps_max  # Upper bound of the system
    # measurement noise

    # Retrieve Data-Driven MPC controller parameters
    input_bounds = dd_mpc_config["U"]  # Predicted input bounds
    u_range = dd_mpc_config["u_range"]  # Persistently exciting input range
    N = dd_mpc_config["N"]  # Initial input-output trajectory length

    # Calculate target quaternion from target yaw
    target_quat = yaw_to_quaternion(target_yaw)

    # Define target drone state
    target_state = TrackingCtrlDroneState(X=target_pos, Q=target_quat)

    # Initialize current drone state
    current_state = TrackingCtrlDroneState(X=env.base_pos, Q=env.base_quat)

    # Initialize control command buffer for vectorized env
    actions_buffer = torch.zeros(
        (env.num_envs, env.num_actions), dtype=torch.float, device=env.device
    )

    # Retrieve env action bounds from env
    env_action_bounds = env.action_bounds

    # Generate persistently exciting input
    u_N = np.hstack(
        [
            np_random.uniform(u_range[i, 0], u_range[i, 1], (N, 1))
            for i in range(m)
        ]
    )

    # Generate bounded uniformly distributed additive measurement noise
    w_N = eps_max * np_random.uniform(-1.0, 1.0, (N, p))

    # Simulate the system using the commands of a stabilizing controller to
    # stabilize the drone at a target position, while adding input deviations
    # from the generated persistently exciting input
    y_N = np.zeros((N, p))
    with torch.no_grad():
        for k in range(N):
            # Update current drone state
            current_state.X = env.base_pos
            current_state.Q = env.base_quat

            # Compute command from stabilizing controller
            ctrl_ctbr_cmd = stabilizing_controller.compute(
                state_setpoint=target_state, state_measurement=current_state
            )

            # Drop yaw setpoint when using CTBR_FIXED_YAW env action type
            if env.action_type == EnvActionType.CTBR_FIXED_YAW:
                ctrl_ctbr_cmd = ctrl_ctbr_cmd[:, :-1]

            # Add persistently exciting input deviations to the
            # stabilizing input for input-output data generation
            controller_action_base = ctrl_ctbr_cmd[base_env_idx]
            u_N[k, :] += controller_action_base.squeeze(0).cpu().numpy()

            # Clip control input to control input bounds
            u_N[k, :] = u_N[k, :].clip(input_bounds[:, 0], input_bounds[:, 1])

            # Calculate env action by scaling control input to a [-1, 1] range
            control_action = torch.tensor(
                u_N[k, :], dtype=torch.float, device=env.device
            ).unsqueeze(0)

            env_action = linear_interpolate(
                x=control_action,
                x_min=env_action_bounds[:, 0],
                x_max=env_action_bounds[:, 1],
                y_min=-1,
                y_max=1,
            )
            env_action = torch.clamp(env_action, -1, 1)

            # Update base env action
            actions_buffer[base_env_idx] = env_action

            # Step environment
            y_N[k, :] = drone_system_model.simulate_step(
                u=u_N[k, :], w=w_N[k, :]
            )

    return u_N, y_N


def get_init_hover_pos(
    config_path: str, controller_key_value: str, env: HoverEnv
) -> torch.Tensor:
    """
    Retrieve the initial hover position from a data-driven MPC controller YAML
    configuration file as a tensor.

    Args:
        config_path (str): The path to the YAML configuration file containing
            data-driven MPC parameters.
        controller_key_value (str): The key to access the controller parameters
            in the config file.
        env (HoverEnv): The drone environment.

    Returns:
        torch.Tensor: The initial hover position with shape (3,).
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    hover_pos_list = config[controller_key_value]["init_hover_pos"]
    init_hover_pos = torch.tensor(
        hover_pos_list, device=env.device, dtype=torch.float
    )

    return init_hover_pos
