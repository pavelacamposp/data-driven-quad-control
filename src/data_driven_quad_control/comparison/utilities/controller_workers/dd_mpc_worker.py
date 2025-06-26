"""
Worker process for a nonlinear data-driven MPC controller.

This module defines a parallel worker function that initializes and runs a
nonlinear data-driven MPC controller in closed loop to control the position of
a drone in a vectorized environment.

The worker communicates with the main process via multiprocessing queues.
"""

import torch.multiprocessing as mp
from direct_data_driven_mpc.utilities.controller.controller_creation import (
    create_nonlinear_data_driven_mpc_controller,
)

from ..controller_comparison_config import (
    DDMPCControllerInitData,
    EnvTargetSignal,
)


def dd_mpc_controller_worker(
    env_idx: int,
    dd_mpc_controller_init_data: DDMPCControllerInitData,
    target_signal_queue: mp.Queue,
    action_queue: mp.Queue,
    dd_mpc_obs_queue: mp.Queue,
) -> None:
    """
    Parallel worker for a nonlinear data-driven MPC (DD-MPC) controller.

    This function initializes a DD-MPC controller from the provided
    initialization data and runs it in closed loop to control the position
    of a drone in simulation.

    The worker communicates with the main process via multiprocessing queues
    to perform the following tasks:
    - Receive target position updates and simulation termination signals.
    - Receive drone position observations.
    - Send control actions.

    Args:
        env_idx (int): The index of the drone controlled by the DD-MPC
            controller.
        dd_mpc_controller_init_data (DDMPCControllerInitData): The DD-MPC
            controller initialization data.
        target_signal_queue (mp.Queue): A queue used for receiving
            `EnvTargetSignal` messages from the main process. Each message
            includes the current target position, a flag indicating whether
            it's a new target (used to trigger controller target updates), and
            a done signal indicating whether the simulation will be terminated.
        action_queue (mp.Queue): A queue used for sending control actions to
            the main process for environment stepping.
        dd_mpc_obs_queue (mp.Queue): A queue used for receiving environment
            observations (drone positions as Numpy arrays) from the main
            process.
    """
    # Create nonlinear data-driven MPC controller
    dd_mpc_controller = create_nonlinear_data_driven_mpc_controller(
        controller_config=dd_mpc_controller_init_data.controller_config,
        u=dd_mpc_controller_init_data.u_N,
        y=dd_mpc_controller_init_data.y_N,
    )

    # Retrieve controller parameters
    n_mpc_step = dd_mpc_controller.n_mpc_step

    # Run the Nonlinear Data-Driven MPC controller in closed loop
    while True:
        # Receive target signal from the main process
        target_signal: EnvTargetSignal = target_signal_queue.get()

        # Update control setpoint if the target position changes
        if target_signal.is_new_target:
            target_pos_tensor = target_signal.target_pos
            target_pos = target_pos_tensor.cpu().numpy().reshape(-1, 1)

            dd_mpc_controller.set_output_setpoint(y_r=target_pos)

        # Update and solve the Data-Driven MPC problem
        dd_mpc_controller.update_and_solve_data_driven_mpc()

        # Controller closed loop for `n_mpc_step` steps
        for n_step in range(n_mpc_step):
            # Update control input
            optimal_u_step_n = (
                dd_mpc_controller.get_optimal_control_input_at_step(
                    n_step=n_step
                )
            )
            u_k = optimal_u_step_n

            # Send action (control input) to the main process
            action_queue.put((env_idx, u_k))

            # Get observations from vectorized environment
            drone_pos = dd_mpc_obs_queue.get()

            # Retrieve system output from observations for the current env
            y_k = drone_pos

            # Update input-output measurements online
            du_current = dd_mpc_controller.get_du_value_at_step(n_step=n_step)
            dd_mpc_controller.store_input_output_measurement(
                u_current=u_k,
                y_current=y_k,
                du_current=du_current,
            )

        # Stop simulation if main process signals termination
        if target_signal.done:
            return
