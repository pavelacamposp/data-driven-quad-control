"""
Run the main process of a data-driven controller comparison.

This module implements the main process responsible for managing the creation
of parallel controller processes (workers) and the vectorized environment
stepping, which requires bidirectional communication with the controller
workers through multiprocessing queues.
"""

import torch
import torch.multiprocessing as mp

from data_driven_quad_control.controllers.tracking.tracking_controller_config import (  # noqa: E501
    TrackingCtrlDroneState,
)
from data_driven_quad_control.envs.hover_env import HoverEnv
from data_driven_quad_control.utilities.drone_environment import (
    update_env_target_pos,
)
from data_driven_quad_control.utilities.math_utils import linear_interpolate

from .controller_comparison_config import EnvTargetSignal, SimInfo
from .controller_workers.dd_mpc_worker import (
    DDMPCControllerInitData,
    dd_mpc_controller_worker,
)
from .controller_workers.rl_worker import (
    RLControllerInitData,
    rl_controller_worker,
)
from .controller_workers.tracking_worker import (
    TrackingControllerInitData,
    tracking_controller_worker,
)


def parallel_controller_simulation(
    env: HoverEnv,
    tracking_env_idx: int,
    tracking_controller_init_data: TrackingControllerInitData,
    dd_mpc_env_idx: int,
    dd_mpc_controller_init_data: DDMPCControllerInitData,
    rl_env_idx: int,
    rl_controller_init_data: RLControllerInitData,
    eval_setpoints: list[torch.Tensor],
    min_at_target_steps: int = 10,
    error_threshold: float = 5e-2,
    record: bool = False,
    video_fps: int = 60,
    verbose: int = 0,
) -> None:
    """
    Run a controller performance comparison in a vectorized drone environment
    using controllers instantiated and run in parallel, independent processes.

    This function manages the execution of a controller comparison simulation
    in a vectorized environment (`HoverEnv`) to compare the performance of
    three controllers:

        - A baseline PID-based tracking controller
        - A Reinforcement Learning (RL) controller (trained PPO policy)
        - A nonlinear data-driven MPC controller (DD-MPC)

    Each controller is instantiated and run in a parallel worker process to
    independently control its assigned drone.

    The main process manages the stepping of the vectorized environment,
    communicating synchronously with the workers via multiprocessing queues.

    The drone controllers are evaluated over a sequence of target setpoints,
    which are updated simultaneously when all drones stabilize at them
    for a minimum number of steps.

    Args:
        env (HoverEnv): The vectorized drone environment.
        tracking_env_idx (int): The index of the drone controlled by the
            tracking controller.
        tracking_controller_init_data (TrackingControllerInitData): The
            tracking controller initialization data.
        dd_mpc_env_idx (int): The index of the drone controlled by the DD-MPC
            controller.
        dd_mpc_controller_init_data (DDMPCControllerInitData): The DD-MPC
            controller initialization data.
        rl_env_idx (int): The index of the drone controlled by the RL
            controller.
        rl_controller_init_data (RLControllerInitData): The RL controller
            initialization data.
        eval_setpoints (list[torch.Tensor]): The target setpoints for
            evaluation.
        min_at_target_steps (int): The minimum number of consecutive steps
            drones must remain near the target to be considered stabilized.
            Once stabilized, the target setpoint is updated to the next on the
            list. Defaults to 10.
        error_threshold (float): The maximum allowable position error to
            consider drones "at their target". Defaults to 5e-2.
        record (bool): If `True`, enables recording the simulation. Defaults to
            `False`.
        video_fps (int): The FPS value for the simulation recording. Unused if
            simulation recording is disabled. Defaults to 60.
        verbose (int): The verbosity level: 0 = no output, 1 = minimal output,
            2 = detailed output.
    """
    # Validate that the multiprocessing start method is "spawn",
    # as it is required for working with CUDA tensors
    if mp.get_start_method(allow_none=True) != "spawn":
        raise ValueError("Spawn start method is required for this function.")

    # Retrieve environment parameters
    env_action_bounds = env.action_bounds  # Env action bounds used for
    # control action normalization

    # Create multiprocessing queues for synchronous
    # communication with the vectorized environment
    target_signal_queue: mp.Queue = mp.Queue()
    action_queue: mp.Queue = mp.Queue()
    tracking_obs_queue: mp.Queue = mp.Queue()
    dd_mpc_obs_queue: mp.Queue = mp.Queue()
    rl_obs_queue: mp.Queue = mp.Queue()

    # Create and start controller worker processes
    processes: list[mp.Process] = []

    if verbose:
        print("Creating parallel processes for each controller")

    # Tracking controller
    if verbose:
        print("  Initializing tracking controller")

    tracking_controller_process = mp.Process(
        target=tracking_controller_worker,
        args=(
            tracking_env_idx,
            env.device,
            tracking_controller_init_data,
            target_signal_queue,
            action_queue,
            tracking_obs_queue,
        ),
    )
    processes.append(tracking_controller_process)
    tracking_controller_process.start()

    # RL controller (trained PPO policy)
    if verbose:
        print("  Initializing Reinforcement Learning controller")

    env_specs = {
        "device": env.device,
        "step_dt": env.step_dt,
        "num_envs": env.num_envs,
        "num_obs": env.num_obs,
        "num_actions": env.num_actions,
        "max_episode_length": env.max_episode_length,
    }
    rl_agent_process = mp.Process(
        target=rl_controller_worker,
        args=(
            rl_env_idx,
            env_specs,
            rl_controller_init_data,
            target_signal_queue,
            action_queue,
            rl_obs_queue,
        ),
    )
    processes.append(rl_agent_process)
    rl_agent_process.start()

    # Data-driven MPC controller
    if verbose:
        print("  Initializing nonlinear data-driven MPC controller")

    dd_mpc_controller_process = mp.Process(
        target=dd_mpc_controller_worker,
        args=(
            dd_mpc_env_idx,
            dd_mpc_controller_init_data,
            target_signal_queue,
            action_queue,
            dd_mpc_obs_queue,
        ),
    )
    processes.append(dd_mpc_controller_process)
    dd_mpc_controller_process.start()

    # Step environment in the main process
    num_processes = len(processes)
    action_buffer = torch.zeros(
        (env.num_envs, env.num_actions), device=env.device, dtype=torch.float
    )

    # Create mask for non-RL envs that require action normalization to [-1, 1]
    non_rl_env_mask = torch.ones(action_buffer.shape[0], dtype=torch.bool)
    non_rl_env_mask[rl_env_idx] = False

    # Start controller simulation
    if verbose:
        print("Running data-driven position control simulation")

    num_setpoints = len(eval_setpoints)
    sim_info = SimInfo(num_targets=num_setpoints)

    with torch.no_grad():
        # Start recording if enabled
        if record:
            env.cam.start_recording()

        for target_idx, target_pos in enumerate(eval_setpoints):
            if verbose:
                print(
                    f"  [{target_idx + 1}/{num_setpoints}] Setting target "
                    f"pos to: {target_pos.tolist()}"
                )

            sim_info.at_target_steps = 0
            sim_info.stabilized_at_target = False
            sim_info.current_target_idx = target_idx
            is_new_target = True

            # Update environment target position
            update_env_target_pos(
                env=env,
                env_idx=list(range(env.num_envs)),
                target_pos=target_pos,
            )

            # Manage environment simulation and communication with controllers
            while not sim_info.stabilized_at_target:
                # Update simulation progress
                update_simulation_progress(
                    target_pos=target_pos,
                    drone_pos=env.get_pos(add_noise=False),  # True drone pos
                    min_at_target_steps=min_at_target_steps,
                    error_threshold=error_threshold,
                    sim_info=sim_info,
                    verbose=verbose,
                )

                # Send drone target position to each process
                for _ in range(num_processes):
                    done = not sim_info.in_progress
                    target_signal_queue.put(
                        EnvTargetSignal(
                            target_pos=target_pos,
                            is_new_target=is_new_target,
                            done=done,
                        )
                    )

                is_new_target = False  # Mark target as already seen

                # Get actions from action queue for each process
                for _ in range(num_processes):
                    env_idx, action = action_queue.get()

                    # Convert action to tensor
                    if env_idx == dd_mpc_env_idx:
                        # Convert array to tensor
                        action = torch.tensor(
                            action, device=env.device
                        ).unsqueeze(0)
                    else:
                        # Move action tensor to env device
                        action = action.to(env.device).unsqueeze(0)

                    # Store action in the action buffer
                    # at its corresponding env idx
                    action_buffer[env_idx] = action

                # Calculate env action by scaling actions to
                # a [-1, 1] range, except for the RL agent action,
                # which is already in this range
                action_buffer[non_rl_env_mask] = linear_interpolate(
                    x=action_buffer[non_rl_env_mask],
                    x_min=env_action_bounds[:, 0],
                    x_max=env_action_bounds[:, 1],
                    y_min=-1,
                    y_max=1,
                )

                # Step environment using batched actions
                obs, _, _, _ = env.step(action_buffer)

                if record:
                    # Render camera view if recording
                    env.cam.render()

                # --- Send observations to workers ---
                drone_pos = env.get_pos()
                drone_quat = env.get_quat()

                # Send tracking controller observation
                tracking_obs_queue.put(
                    TrackingCtrlDroneState(
                        drone_pos[tracking_env_idx].unsqueeze(0),
                        drone_quat[tracking_env_idx].unsqueeze(0),
                    )
                )

                # Send RL environment observation
                rl_obs_queue.put(obs[rl_env_idx])

                # Send data-driven MPC controller observation
                dd_mpc_obs = drone_pos[dd_mpc_env_idx].cpu().numpy()
                dd_mpc_obs_queue.put(dd_mpc_obs)

        # Stop recording and save video file
        if record:
            env.cam.stop_recording(
                save_to_filename="drone_eval.mp4",
                fps=video_fps,
            )

    # Wait for all processes to complete
    for p in processes:
        p.join()


def update_simulation_progress(
    target_pos: torch.Tensor,
    drone_pos: torch.Tensor,
    min_at_target_steps: int,
    error_threshold: float,
    sim_info: SimInfo,
    verbose: int,
) -> None:
    # Update the duration of drones hovering close to its target
    pos_error = torch.abs(target_pos - drone_pos).max().item()
    if pos_error < error_threshold:
        sim_info.at_target_steps += 1

        # Mark drones as stabilized if they remained at its target long enough
        if sim_info.at_target_steps == min_at_target_steps:
            sim_info.stabilized_at_target = True

            if verbose:
                print("    Drones successfully stabilized")

            # Mark simulation for termination if the
            # drones stabilized at the last target
            if sim_info.current_target_idx == sim_info.num_targets - 1:
                sim_info.in_progress = False
    else:
        # Reset at target step count if a drone
        # moves out of the target's vicinity
        sim_info.at_target_steps = 0
