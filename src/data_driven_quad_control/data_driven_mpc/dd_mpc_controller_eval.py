"""
Nonlinear Data-Driven MPC for Drone Position Control

This script demonstrates the use of a nonlinear data-driven Model Predictive
Control (MPC) controller for drone position control in Genesis (via the
`HoverEnv` environment).

It first uses a stabilizing controller (`DroneTrackingController`) to stabilize
a drone at a target position for initial input-output data collection. Then, it
initializes a nonlinear data-driven MPC controller with the collected data and
evaluates it in a closed-loop simulation.

The drone environment is configured to use a Collective Thrust and Body Rates
(CTBR) controller internally, with a fixed yaw angular rate of 0 rad/s. This
simplifies the drone control system and sets the number of control inputs to 3:
total thrust [N], roll angular rate [rad/s], pitch angular rate [rad/s].

The nonlinear data-driven MPC controller is available at:
https://github.com/pavelacamposp/direct_data_driven_mpc
"""

import argparse
import os
import warnings

import genesis as gs
import numpy as np
import torch
from direct_data_driven_mpc.utilities.controller.controller_creation import (
    create_nonlinear_data_driven_mpc_controller,
)
from direct_data_driven_mpc.utilities.controller.controller_params import (
    get_nonlinear_data_driven_mpc_controller_params,
)
from direct_data_driven_mpc.utilities.controller.data_driven_mpc_sim import (
    simulate_nonlinear_data_driven_mpc_control_loop,
)

from data_driven_quad_control.envs.config.hover_env_config import (
    EnvActionType,
    get_cfgs,
)
from data_driven_quad_control.utilities.drone_environment import (
    create_env,
    update_env_target_pos,
)
from data_driven_quad_control.utilities.drone_tracking_controller import (
    create_drone_tracking_controller,
    hover_at_target,
)

from .utilities.drone_initial_data_collection import (
    collect_initial_input_output_data,
    get_init_hover_pos,
)
from .utilities.drone_system_model import (
    create_system_model,
)

# Suppress all warnings from CVXPY to maintain clean terminal output
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")

# Directory paths
DD_MPC_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "./config")
)

# Data-Driven MPC controller configuration file
DEFAULT_CONTROLLER_CONFIG_FILE = "dd_mpc_controller_params.yaml"
DEFAULT_CONTROLLER_CONFIG_PATH = os.path.join(
    DD_MPC_CONFIG_DIR, DEFAULT_CONTROLLER_CONFIG_FILE
)
DEFAULT_CONTROLLER_KEY_VALUE = "nonlinear_dd_mpc_approx_1_step"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nonlinear Data-Driven MPC for Drone Position Control"
    )

    parser.add_argument(
        "--controller_config_path",
        type=str,
        default=DEFAULT_CONTROLLER_CONFIG_PATH,
        help="The path to the YAML configuration file containing the "
        "nonlinear data-driven MPC parameters.",
    )
    parser.add_argument(
        "--controller_key_value",
        type=str,
        default=DEFAULT_CONTROLLER_KEY_VALUE,
        help="The key to access the controller parameters in the "
        "configuration file.",
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
    controller_config_path = args.controller_config_path
    controller_key_value = args.controller_key_value
    t_sim = args.t_sim
    headless = args.headless
    seed = args.seed
    verbose = args.verbose

    if verbose:
        print("--- Nonlinear Data-Driven MPC Controller Evaluation ---")
        print("-" * 55)

    # Create a Random Number Generator for reproducibility
    np_random = np.random.default_rng(seed=seed)

    # Initialize Genesis simulator
    if verbose:
        print("Initializing Genesis simulator")

        if verbose > 1 and seed is not None:
            print(f"    RNG seed: {seed}")

    gs.init(seed=seed, backend=gs.gpu, logging_level="warning")

    # Load environment configuration
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    # Set up visualization
    env_cfg["visualize_target"] = True
    env_cfg["max_visualize_FPS"] = 100  # Sim visualization FPS

    # Increase episode length and spatial bounds to allow sufficient
    # time and space for the data-driven MPC controller evaluation
    env_cfg["episode_length_s"] = 100
    env_cfg["termination_if_close_to_ground"] = 0.0
    env_cfg["termination_if_x_greater_than"] = 100.0
    env_cfg["termination_if_y_greater_than"] = 100.0
    env_cfg["termination_if_z_greater_than"] = 10.0

    # Create environment with a single instance, since the
    # Data-Driven MPC controller is not vectorized
    if verbose:
        print("Creating drone environment")

    num_envs = 1
    show_viewer = not headless
    env = create_env(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
        action_type=EnvActionType.CTBR_FIXED_YAW,
    )

    # Reset environment
    env.reset()

    # Initialize drone system model
    base_env_idx = 0  # Index of the drone instance used for evaluating
    # the Data-Driven MPC controller
    system_model = create_system_model(env=env, env_idx=base_env_idx)

    # Load nonlinear data-driven MPC controller parameters from config file
    if verbose:
        print(
            "Loading nonlinear data-driven MPC controller parameters from "
            "configuration file"
        )

    m = env.num_actions  # Number of inputs
    p = 3  # Number of outputs
    dd_mpc_config = get_nonlinear_data_driven_mpc_controller_params(
        config_file=controller_config_path,
        controller_key_value=controller_key_value,
        m=m,
        p=p,
        verbose=verbose,
    )

    # Initial input-output data collection
    if verbose:
        print("\nInitial Input-Output Data Collection")
        print("-" * 36)

    # Create a controller to stabilize the drone at a specific
    # position for initial input-output data collection
    stabilizing_controller = create_drone_tracking_controller(env=env)

    # Load initial hover target position from configuration file
    target_pos = get_init_hover_pos(
        config_path=controller_config_path,
        controller_key_value=controller_key_value,
        env=env,
    )
    target_yaw = torch.tensor([0.0], device=env.device, dtype=torch.float)

    # Update target position for visualization
    update_env_target_pos(
        env=env, env_idx=list(range(env.num_envs)), target_pos=target_pos
    )

    # Command drone to hover at target
    if verbose:
        print(f"Hovering drone at target {target_pos.tolist()}")

    target_pos = target_pos.expand(env.num_envs, -1)
    target_yaw = target_yaw.expand(env.num_envs)
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
        print("Collecting initial input-output data")

    u_N, y_N = collect_initial_input_output_data(
        env=env,
        base_env_idx=base_env_idx,
        stabilizing_controller=stabilizing_controller,
        target_pos=target_pos,
        target_yaw=target_yaw,
        input_bounds=dd_mpc_config["U"],
        u_range=dd_mpc_config["u_range"],
        N=dd_mpc_config["N"],
        m=system_model.m,
        p=system_model.p,
        eps_max=system_model.eps_max,
        np_random=np_random,
        drone_system_model=system_model,
    )

    # Evaluate data-driven MPC controller in simulation
    if verbose:
        print("\nNonlinear Data-Driven MPC Controller Evaluation")
        print("-" * 47)

    # Update target position for visualization
    y_r = np.array(dd_mpc_config["y_r"], dtype=float).reshape(1, -1)
    y_r_tensor = torch.tensor(y_r, device=env.device, dtype=torch.float)
    update_env_target_pos(
        env=env, env_idx=list(range(env.num_envs)), target_pos=y_r_tensor
    )

    if verbose:
        print(f"Setting drone target to {y_r.flatten().tolist()}")

    # Create nonlinear data-driven MPC controller
    if verbose:
        print("Initializing nonlinear data-driven MPC controller")

    nonlinear_dd_mpc_controller = create_nonlinear_data_driven_mpc_controller(
        controller_config=dd_mpc_config, u=u_N, y=y_N
    )

    # Simulate Data-Driven MPC control system
    if verbose:
        print("Running data-driven MPC position control simulation")

    try:
        simulate_nonlinear_data_driven_mpc_control_loop(
            system_model=system_model,
            data_driven_mpc_controller=nonlinear_dd_mpc_controller,
            n_steps=t_sim,
            np_random=np_random,
            verbose=verbose,
        )

    except Exception as e:
        if verbose:
            print(f"Controller evaluation failed with error: {str(e)}")

    finally:
        if verbose:
            print("Controller evaluation finished.")

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
