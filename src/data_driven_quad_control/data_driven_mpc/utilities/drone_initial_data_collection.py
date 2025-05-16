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
from data_driven_quad_control.envs.hover_env import HoverEnv
from data_driven_quad_control.envs.hover_env_config import EnvActionType
from data_driven_quad_control.utilities.math_utils import (
    linear_interpolate,
    yaw_to_quaternion,
)


def collect_initial_input_output_data(
    env: HoverEnv,
    base_env_idx: int,
    stabilizing_controller: DroneTrackingController,
    target_pos: torch.Tensor,
    target_yaw: torch.Tensor,
    input_bounds: np.ndarray,
    u_range: np.ndarray,
    N: int,
    m: int,
    p: int,
    np_random: Generator,
    eps_max: float = 0.0,
    drone_system_model: NonlinearSystem | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect initial input-output data from a single drone in a vectorized
    environment for use in data-driven MPC controller creation.

    During collection, the drone is stabilized at a specified target position
    and yaw using a stabilizing controller. A persistently exciting input,
    generated based on the data-driven MPC controller parameters, is added as
    a perturbation to the commands from the stabilizing controller. This
    ensures effective output data collection while maintaining drone stability.

    As a result, the actual input used for collecting output data is the sum of
    the stabilizing controller commands and the persistently exciting input.

    Args:
        env (HoverEnv): The vectorized drone environment.
        base_env_idx (int): The environment instance (drone) index used for
            data collection.
        input_bounds (np.ndarray): The bounds for the predicted input in a
            nonlinear data-driven MPC controller, with shape (`m`, 2).
        u_range (np.ndarray): The range of the persistently exciting input,
            with shape (`m`, 2).
        N (int): The length of the input-output trajectory.
        m (int): The number of control inputs.
        p (int): The number of drone system outputs.
        stabilizing_controller (DroneTrackingController): A drone tracking
            controller used for stabilizing the drone during data collection.
        target_pos (torch.Tensor): The target position for stabilization, with
            shape (`num_envs`, 3).
        target_yaw (torch.Tensor): The target yaw for stabilization, with shape
            (`num_envs`).
        np_random (Generator): A Numpy random number generator for generating
            the persistently exciting input and random noise for the system's
            output.
        eps_max (float): The upper bound of the system measurement noise. Used
            when the `drone_system_model` is provided. Defaults to 0.0.
        drone_system_model (NonlinearSystem | None): Optional drone nonlinear
            system. It is a wrapper around the environment to interface with
            the `direct-data-driven-mpc` package. If not provided, the drone
            environment is stepped directly.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two arrays: a
            persistently exciting input and the system's output response.
            The input array has shape `(N, m)` and the output array has shape
            `(N, p)`, where `N` is the trajectory length, `m` is the number of
            control inputs, and `p` is the number of drone system outputs.
    """
    # Determine whether a drone system model will be used or
    # the drone environment will be stepped directly
    use_system_model = drone_system_model is not None

    if drone_system_model is not None:
        assert m == drone_system_model.m, (
            "Mismatch between provided `m` and `drone_system_model.m`"
        )
        assert p == drone_system_model.p, (
            "Mismatch between provided `p` and `drone_system_model.p`"
        )

    # Define target drone state
    target_quat = yaw_to_quaternion(target_yaw)
    target_state = TrackingCtrlDroneState(X=target_pos, Q=target_quat)

    # Initialize current drone state
    current_state = TrackingCtrlDroneState(X=env.base_pos, Q=env.base_quat)

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

    # Simulate the system using the commands from a stabilizing controller to
    # stabilize the drone at a target position, while adding the generated
    # persistently exciting input as a perturbation
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

            # Add persistently exciting input as a perturbation to the
            # stabilizing input for the the `base_env_idx` env action
            # Note:
            # The input used for collecting output data is the sum
            # of the stabilizing controller commands and the persistently
            # exciting input.
            controller_action_base = ctrl_ctbr_cmd[base_env_idx]
            u_N[k, :] += controller_action_base.squeeze(0).cpu().numpy()

            # Clip control input to control input bounds
            u_N[k, :] = u_N[k, :].clip(input_bounds[:, 0], input_bounds[:, 1])

            # Update controller commands for the `base_env_idx` env
            # with the generated persistently exciting input
            ctrl_ctbr_cmd[base_env_idx] = torch.from_numpy(u_N[k : k + 1]).to(
                env.device
            )

            # Calculate env action by scaling the CTBR controller
            # commands to a [-1, 1] range
            env_action = linear_interpolate(
                x=ctrl_ctbr_cmd,
                x_min=env_action_bounds[:, 0],
                x_max=env_action_bounds[:, 1],
                y_min=-1,
                y_max=1,
            )
            env_action = torch.clamp(env_action, -1, 1)

            # Step environment
            if use_system_model:
                # Prevent mypy [union-attr] error
                assert drone_system_model is not None

                y_N[k, :] = drone_system_model.simulate_step(
                    u=u_N[k, :], w=w_N[k, :]
                )
            else:
                env.step(env_action)

                # Get system output (drone position)
                y_N[k, :] = env.base_pos[base_env_idx].cpu().numpy()

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
