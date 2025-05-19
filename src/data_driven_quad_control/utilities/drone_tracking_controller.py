import os

import torch

from data_driven_quad_control.controllers.ctbr.ctbr_controller import (
    DroneCTBRController,
)
from data_driven_quad_control.controllers.tracking.tracking_controller import (
    DroneTrackingController,
)
from data_driven_quad_control.controllers.tracking.tracking_controller_config import (  # noqa: E501
    TrackingCtrlDroneState,
)
from data_driven_quad_control.envs.hover_env import HoverEnv
from data_driven_quad_control.envs.hover_env_config import (
    EnvActionType,
    EnvDroneParams,
)
from data_driven_quad_control.utilities.config_utils import load_yaml_config
from data_driven_quad_control.utilities.math_utils import (
    linear_interpolate,
    yaw_to_quaternion,
)

# Config file path for drone tracking controller parameters
TRACKING_CONTROLLER_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../configs/controllers/tracking/tracking_controller_params.yaml",
)


def create_drone_tracking_controller(env: HoverEnv) -> DroneTrackingController:
    # Load tracking controller config from YAML file
    tracking_controller_config = load_yaml_config(
        TRACKING_CONTROLLER_CONFIG_PATH
    )

    # Create drone tracking controller
    controller = DroneTrackingController(
        drone_mass=EnvDroneParams.MASS,
        controller_config=tracking_controller_config,
        dt=env.step_dt,
        num_envs=env.num_envs,
        device=env.device,
    )

    return controller


def hover_at_target(
    env: HoverEnv,
    tracking_controller: DroneTrackingController,
    target_pos: torch.Tensor,
    target_yaw: torch.Tensor,
    min_at_target_steps: float | None = 10,
    error_threshold: float = 5e-2,
    ctbr_controller: DroneCTBRController | None = None,
) -> None:
    """
    Command drones in a vectorized environment to hover at a target position
    using a tracking controller. This function returns once all the drones are
    stabilized.

    A drone is considered stabilized at the target once it remains within its
    vicinity (i.e., the position error is less than `error_threshold`) for a
    given number of consecutive steps (`min_at_target_steps`).

    This function supports both CTBR (`CTBR` and `CTBR_FIXED_YAW`) and direct
    rotor RPM (`ROTOR_RPMS`) action types.

    Note:
        An external CTBR controller (`DroneCTBRController`) must be provided
        when using the `ROTOR_RPMS` environment action type.

    Args:
        env (HoverEnv): The vectorized drone environment containing `num_envs`
            parallel environments.
        tracking_controller (DroneTrackingController): The drone tracking
            controller. Outputs CTBR commands (total thrust and body rates).
        target_pos (torch.Tensor): The target position tensor of shape
            (`num_envs`, 3).
        target_yaw (torch.Tensor): The target yaw tensor of shape (`num_envs`).
        min_at_target_steps (float | None): The minimum number of consecutive
            steps the drone must remain near the target to be considered
            stabilized. If `None`, the drone will hover indefinitely.
        error_threshold (float): The maximum allowable position error to
            consider the drone "at its target".
        ctbr_controller (DroneCTBRController | None): The external drone CTBR
            controller used to follow CTBR commands from the tracking
            controller and output rotor RPMs. Required if the environment
            action type is `ROTOR_RPMS`.
    """
    # Determine whether an external CTBR controller is required
    # Note:
    # An internal CTBR controller is included within the drone
    # environment for env action types CTBR and CTBR_FIXED_YAW
    use_ctbr = env.action_type == EnvActionType.ROTOR_RPMS

    if use_ctbr and ctbr_controller is None:
        raise ValueError(
            "A CTBR controller must be provided when `ROTOR_RPMS` "
            "is used as the env action type."
        )

    # Calculate target quaternion from target yaw
    target_quat = yaw_to_quaternion(target_yaw)

    # Define target drone state
    target_state = TrackingCtrlDroneState(X=target_pos, Q=target_quat)

    # Get env action bounds
    env_action_bounds = env.action_bounds

    # Initialize current drone state
    current_state = TrackingCtrlDroneState(X=env.base_pos, Q=env.base_quat)

    at_target_steps = 0
    with torch.no_grad():
        while (
            min_at_target_steps is None
            or at_target_steps < min_at_target_steps
        ):
            # Update current drone state
            current_state.X = env.base_pos
            current_state.Q = env.base_quat

            # Compute CTBR action from tracking controller
            ctrl_ctbr_action = tracking_controller.compute(
                state_setpoint=target_state, state_measurement=current_state
            )

            # Drop yaw setpoint when using CTBR_FIXED_YAW env action type
            if env.action_type == EnvActionType.CTBR_FIXED_YAW:
                ctrl_ctbr_action = ctrl_ctbr_action[:, :-1]

            # Compute rotor RPMs from CTBR action using the external CTBR
            # controller if the env action type is ROTOR_RPMS
            if use_ctbr:
                # Prevent mypy union-attr error
                assert ctbr_controller is not None

                ctrl_rpm_action = ctbr_controller.compute(
                    rate_measurements=env.base_ang_vel,
                    rate_setpoints=ctrl_ctbr_action[:, 1:],
                    thrust_setpoints=ctrl_ctbr_action[:, 0],
                )
                ctrl_action = ctrl_rpm_action
            else:
                ctrl_action = ctrl_ctbr_action

            # Calculate env action by normalizing the
            # control action to a [-1, 1] range
            env_action = linear_interpolate(
                x=ctrl_action,
                x_min=env_action_bounds[:, 0],
                x_max=env_action_bounds[:, 1],
                y_min=-1,
                y_max=1,
            )

            # Step simulation
            env.step(env_action)

            # Update duration of drone hovering close to its target
            pos_error = (
                torch.abs(target_state.X - current_state.X).max().item()
            )
            if pos_error < error_threshold:
                at_target_steps += 1
            else:
                at_target_steps = 0
