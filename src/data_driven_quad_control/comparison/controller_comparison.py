"""
Data-driven Controller Comparison: Reinforcement Learning vs. Data-Driven MPC

This script compares three controllers for drone position control in Genesis
(via the `HoverEnv` environment):
    1. [Red] A baseline PID-based tracking controller
    2. [Green] A Reinforcement Learning (RL) controller (trained PPO policy)
    3. [Goldenrod] A nonlinear data-driven MPC controller (DD-MPC)

Simulation overview:

1. All drones are stabilized at an initial hovering position using a
   stabilizing controller (`DroneTrackingController`).

2. Initialization data is constructed for each controller based on the
   configuration parameters defined in the main YAML config file
   (`controller_comparison_config.yaml`).

3. Since the DD-MPC controller requires an initial input-output trajectory, its
   corresponding drone is then used for initial data collection, while the
   other drones remain stabilized.

4. The controller initialization data is passed to the main simulation process,
   which then instantiates each controller in parallel, independent processes.

5. The comparison simulation starts, handled by the main simulation process,
   which manages the vectorized environment stepping and communication with the
   controller processes via multiprocessing queues.

   Each parallel controller process controls its corresponding drone to follow
   the setpoints defined in the main YAML config file.

6. Upon termination, control trajectory data collected during simulation is
   saved to a file in the `logs/comparison` directory. This data includes
   control inputs, drone positions, and target setpoints for all controllers,
   and can be used for post-evaluation and plotting to compare the position
   control performance of each controller.

The drone environment is configured to use a Collective Thrust and Body Rates
(CTBR) controller internally, with a fixed yaw angular rate of 0 rad/s. This
simplifies the drone control system and sets the number of control inputs to
three:
    - Total thrust [N]
    - Roll angular rate [rad/s]
    - Pitch angular rate [rad/s]
"""

import argparse
import os
import pickle
import warnings

import genesis as gs
import numpy as np
import torch.multiprocessing as mp

from data_driven_quad_control.controllers.tracking.tracking_controller_config import (  # noqa: E501
    TrackingCtrlDroneState,
)
from data_driven_quad_control.envs.hover_env_config import (
    EnvActionType,
    get_cfgs,
)
from data_driven_quad_control.learning.config.hover_ppo_config import (
    get_train_cfg,
)
from data_driven_quad_control.utilities.drone_environment import (
    create_env,
    update_env_target_pos,
)
from data_driven_quad_control.utilities.drone_tracking_controller import (
    create_drone_tracking_controller,
)

from .utilities.comparison_config_loader import (
    load_controller_comparison_params,
)
from .utilities.controller_comparison_config import (
    RLControllerInitData,
    TrackingControllerInitData,
)
from .utilities.dd_mpc_initial_data_collection import (
    get_data_driven_mpc_controller_init_data,
)
from .utilities.parallel_controller_sim import parallel_controller_simulation

# Required for multiprocessing with CUDA tensors
mp.set_start_method("spawn", force=True)

# Suppress all warnings from CVXPY to maintain clean terminal output
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")

# Tracking controller configuration file
DEFAULT_COMPARISON_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../configs/comparison/controller_comparison_config.yaml",
)

# Directory for saving controller comparison data
COMPARISON_LOGS_DIR = "logs/comparison"
os.makedirs(COMPARISON_LOGS_DIR, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nonlinear Data-Driven MPC for Drone Position Control"
    )

    parser.add_argument(
        "--comparison_config_path",
        type=str,
        default=DEFAULT_COMPARISON_CONFIG_PATH,
        help="The path to the YAML configuration file containing the "
        "controller comparison configuration parameters.",
    )
    parser.add_argument(
        "--t_sim",
        type=int,
        default=300,
        help="The simulation length in time steps.",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Disable GUI viewer."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The Random Number Generator seed for reproducibility. Defaults "
        "to `None`.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Enable video recording of the simulation.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="The verbosity level: 0 = no output, 1 = minimal output, 2 = "
        "detailed output.",
    )

    return parser.parse_args()


def main() -> None:
    # Parse arguments
    args = parse_args()
    comparison_config_path = args.comparison_config_path
    headless = args.headless
    seed = args.seed
    record = args.record
    verbose = args.verbose

    if verbose:
        print(
            "--- Data-Driven Controller Comparison: RL and Data-Driven MPC ---"
        )
        print("-" * 65)

    # Create a Random Number Generator for reproducibility
    np_random = np.random.default_rng(seed=seed)

    # Initialize Genesis simulator
    if verbose:
        print("Initializing Genesis simulator")

        if verbose > 1 and seed is not None:
            print(f"  RNG seed: {seed}")

    gs.init(seed=seed, backend=gs.gpu, logging_level="warning")

    # Load environment configuration
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    # Enable observation noise
    obs_cfg["obs_noise_std"] = 1e-4  # Observation noise std (normalized)
    # Note:
    # Observation noise is added to normalized observations. For drone position
    # observations, an std of 1e-4 corresponds to a physical position error std
    # of 1e-4 / obs_cfg["obs_scales"]["rel_pos"] = 1e-4 / (1 / 3) = 0.3 mm
    # after denormalization and rescaling (since x_norm = x * scale).

    # Set up visualization
    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = args.record  # Enable camera for recording
    env_cfg["max_visualize_FPS"] = 100  # Sim visualization FPS

    # Increase episode length and spatial bounds to allow sufficient
    # time and space for the data-driven MPC controller evaluation
    env_cfg["episode_length_s"] = 100
    env_cfg["termination_if_close_to_ground"] = 0.0
    env_cfg["termination_if_x_greater_than"] = 100.0
    env_cfg["termination_if_y_greater_than"] = 100.0
    env_cfg["termination_if_z_greater_than"] = 10.0

    # Create vectorized environment with `CTBR_FIXED_YAW` actions
    if verbose:
        print("Creating drone environment")

    num_envs = 3
    show_viewer = not headless
    drone_colors = [
        (1.0, 0.0, 0.0, 0.75),  # Red: Tracking controller
        (0.0, 1.0, 0.0, 0.75),  # Green: RL-based controller
        (0.855, 0.647, 0.125, 0.75),  # Goldenrod: Data-Driven MPC controller
    ]
    env = create_env(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
        action_type=EnvActionType.CTBR_FIXED_YAW,
        drone_colors=drone_colors,
    )

    # Reset environment
    obs, _ = env.reset()

    # Load controller configuration data
    controller_comparison_params = load_controller_comparison_params(
        config_path=comparison_config_path,
        env_device=env.device,
        verbose=verbose,
    )

    # --- Construct controller initialization data ---
    if verbose:
        print("\nController Initialization Data Construction")
        print("-" * 43)

    # Tracking controller initialization data
    if verbose:
        print("Constructing initialization data for tracking controller")

    tracking_env_idx = 0
    tracking_controller_config = (
        controller_comparison_params.tracking_controller_config
    )
    init_drone_tracking_ctrl_state = TrackingCtrlDroneState(
        X=env.get_pos()[tracking_env_idx].unsqueeze(0),
        Q=env.get_quat()[tracking_env_idx].unsqueeze(0),
    )
    tracking_controller_init_data = TrackingControllerInitData(
        controller_config=tracking_controller_config,
        controller_dt=env.step_dt,
        initial_state=init_drone_tracking_ctrl_state,
    )

    # RL controller (PPO policy) initialization data:
    # PPO agent training configuration, model path, and initial env observation
    if verbose:
        print("Constructing initialization data for RL-based controller")

    rl_env_idx = 1
    train_cfg = get_train_cfg("trained_ppo_policy", 0)
    rl_controller_init_data = RLControllerInitData(
        train_cfg=train_cfg,
        model_path=controller_comparison_params.ppo_model_path,
        initial_observation=obs[rl_env_idx],
    )

    # Nonlinear data-driven MPC controller initialization data
    if verbose:
        print(
            "Constructing initialization data for data-driven MPC controller"
        )

    dd_mpc_env_idx = 2
    dd_mpc_controller_config = (
        controller_comparison_params.dd_mpc_controller_config
    )
    init_hover_pos = controller_comparison_params.init_hover_pos

    # Create a stabilizing controller for initial data collection
    stabilizing_controller = create_drone_tracking_controller(env=env)

    # Update target position for visualization
    update_env_target_pos(
        env=env,
        env_idx=list(range(env.num_envs)),
        target_pos=init_hover_pos,
    )

    # Collect initial data for the data-driven MPC controller, while
    # stabilizing the drones corresponding to the other controllers
    dd_mpc_controller_init_data = get_data_driven_mpc_controller_init_data(
        env=env,
        dd_mpc_env_idx=dd_mpc_env_idx,
        dd_mpc_controller_config=dd_mpc_controller_config,
        init_hover_pos=init_hover_pos,
        stabilizing_controller=stabilizing_controller,
        np_random=np_random,
        verbose=verbose,
    )

    # Simulate control systems for comparison
    if verbose:
        print("\nData-Driven Controller Comparison Simulation")
        print("-" * 44)

    control_trajectory_data = None

    try:
        min_at_target_steps = 10
        error_threshold = 5e-2

        control_trajectory_data = parallel_controller_simulation(
            env=env,
            tracking_env_idx=tracking_env_idx,
            tracking_controller_init_data=tracking_controller_init_data,
            rl_env_idx=rl_env_idx,
            rl_controller_init_data=rl_controller_init_data,
            dd_mpc_env_idx=dd_mpc_env_idx,
            dd_mpc_controller_init_data=dd_mpc_controller_init_data,
            eval_setpoints=controller_comparison_params.eval_setpoints,
            min_at_target_steps=min_at_target_steps,
            error_threshold=error_threshold,
            record=record,
            video_fps=env_cfg["max_visualize_FPS"],
            verbose=verbose,
        )

    except Exception as e:
        if verbose:
            print(f"Controller comparison failed with error: {str(e)}")

    finally:
        # Save control trajectory data if no exception occurred
        control_data_file = os.path.join(
            COMPARISON_LOGS_DIR, "control_trajectory.pkl"
        )
        if control_trajectory_data is not None:
            with open(control_data_file, "wb") as f:
                pickle.dump(control_trajectory_data, f)

            if verbose:
                print(
                    f"Controller trajectory data saved to: {control_data_file}"
                )

        if verbose:
            print("\nController comparison finished.")

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
