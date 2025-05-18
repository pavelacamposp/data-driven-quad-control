"""
Worker process for a tracking controller.

This module defines a parallel worker function that initializes and runs a
tracking controller in closed loop to control the position of a drone in a
vectorized environment.

The worker communicates with the main process via multiprocessing queues.
"""

import torch
import torch.multiprocessing as mp

from data_driven_quad_control.controllers.tracking.tracking_controller import (
    DroneTrackingController,
)
from data_driven_quad_control.controllers.tracking.tracking_controller_config import (  # noqa: E501
    TrackingCtrlDroneState,
)
from data_driven_quad_control.envs.hover_env_config import (
    EnvDroneParams,
)

from ..controller_comparison_config import (
    EnvTargetSignal,
    TrackingControllerInitData,
)


def tracking_controller_worker(
    env_idx: int,
    env_device: torch.device,
    tracking_controller_init_data: TrackingControllerInitData,
    target_signal_queue: mp.Queue,
    action_queue: mp.Queue,
    tracking_obs_queue: mp.Queue,
) -> None:
    """
    Parallel worker for a tracking controller.

    This function initializes a tracking controller from the provided
    initialization data and runs it in closed loop to control the position
    of a drone in simulation.

    The worker communicates with the main process via multiprocessing queues
    to perform the following tasks:
    - Receive target position updates and simulation termination signals.
    - Receive drone position and quaternion observations.
    - Send control actions.

    Args:
        env_idx (int): The index of the drone controlled by the tracking
            controller.
        env_device (torch.device): The drone environment device.
        tracking_controller_init_data (TrackingControllerInitData): The
            tracking controller initialization data.
        target_signal_queue (mp.Queue): A queue used for receiving
            `EnvTargetSignal` messages from the main process. Each message
            includes the current target position, a flag indicating whether
            it's a new target (used to trigger controller target updates), and
            a done signal indicating whether the simulation will be terminated.
        action_queue (mp.Queue): A queue used for sending control actions to
            the main process for environment stepping.
        tracking_obs_queue (mp.Queue): A queue used for receiving environment
            observations (`TrackingCtrlDroneState`) from the main process,
            containing the current drone position and orientation (as a
            quaternion).
    """
    # Create drone tracking controller
    tracking_controller = DroneTrackingController(
        drone_mass=EnvDroneParams.MASS,
        controller_config=tracking_controller_init_data.controller_config,
        dt=tracking_controller_init_data.controller_dt,
        num_envs=1,
        device=env_device,
    )

    # Initialize target drone state
    # controller_device = tracking_controller.device
    target_state = TrackingCtrlDroneState(
        X=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=env_device),
        Q=torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float, device=env_device
        ),
    )

    # Initialize current drone state
    current_state = tracking_controller_init_data.initial_state

    while True:
        # Receive target signal from the main process
        target_signal: EnvTargetSignal = target_signal_queue.get()

        # Update target state position if it changes
        if target_signal.is_new_target:
            target_pos = target_signal.target_pos
            target_state.X = target_pos

        # Compute CTBR action from tracking controller
        ctrl_action = tracking_controller.compute(
            state_setpoint=target_state, state_measurement=current_state
        )

        # Drop yaw setpoint since the env uses CTBR_FIXED_YAW actions
        ctrl_action = ctrl_action[:, :-1]

        # Send action (control input) to the main process
        action_queue.put((env_idx, ctrl_action))

        # Get observations from vectorized environment
        env_obs: TrackingCtrlDroneState = tracking_obs_queue.get()

        # Update current drone state
        current_state.X = env_obs.X
        current_state.Q = env_obs.Q

        # Stop simulation if main process signals termination
        if target_signal.done:
            break
