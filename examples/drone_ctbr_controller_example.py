"""
Drone Control using a Collective Thrust and Body Rates (CTBR) controller

This script demonstrates the control of drones in a vectorized Genesis
environment (`HoverEnv`) using low-level CTBR controllers
(`DroneCTBRController`). The environment actions (CTBR actions) consist of
the drone's total thrust and its roll, pitch, and yaw body rates.
"""

import argparse

import genesis as gs
import torch

from data_driven_quad_control.envs.hover_env import HoverEnv
from data_driven_quad_control.envs.hover_env_config import (
    EnvActionBounds,
    EnvActionType,
    EnvDroneParams,
    get_cfgs,
)
from data_driven_quad_control.utilities.math_utils import linear_interpolate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drone CTBR controller example in Genesis"
    )

    parser.add_argument(
        "--num_envs", type=int, default=1, help="The number of parallel envs."
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=300,
        help="The number of simulation steps.",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Disable GUI viewer."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_envs = args.num_envs
    num_steps = args.num_steps
    headless = args.headless

    print("--- Drone CTBR Controller Example ---")
    print("-" * 37)

    # Initialize Genesis simulator
    print("Initializing Genesis simulator")

    gs.init(backend=gs.gpu, logging_level="warning")

    # Load environment configuration
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    # Set up visualization
    env_cfg["visualize_target"] = True
    env_cfg["max_visualize_FPS"] = 100  # Sim visualization FPS

    # Create environment
    print(
        "Creating drone environment with internal CTBR controller "
        "(CTBR actions)"
    )

    show_viewer = not headless
    env = HoverEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
        action_type=EnvActionType.CTBR,
    )

    # Reset environment
    env.reset()

    # Retrieve CTBR controller setpoint bounds from env action bounds
    ctbr_setpoint_bounds = env.action_bounds

    # Define a sequence of CTBR setpoints to demonstrate different behaviors
    # Note: Each setpoint vector consists of:
    #   [Total thrust [N],
    #    Roll rate [rad/s],
    #    Pitch rate [rad/s],
    #    Yaw rate [rad/s]]
    ctbr_setpoint_list = [
        # Positive Yaw rate
        [EnvDroneParams.WEIGHT, 0.0, 0.0, 5.0],
        # Ascend with negative Yaw rate
        [EnvDroneParams.WEIGHT + 0.01, 0.0, 0.0, -5.0],
        # Descend with max Yaw rate
        [
            EnvDroneParams.WEIGHT - 0.01,
            0.0,
            0.0,
            EnvActionBounds.MAX_ANG_VELS[2],
        ],
        # Hover
        [EnvDroneParams.WEIGHT, 0.0, 0.0, 0.0],
        # Positive Roll rate
        [EnvDroneParams.WEIGHT, 0.01, 0.0, 0.0],
        # Negative Roll rate
        [EnvDroneParams.WEIGHT, -0.01, 0.0, 0.0],
        # Positive Pitch rate
        [EnvDroneParams.WEIGHT, 0.0, 0.01, 0.0],
        # Negative Pitch rate
        [EnvDroneParams.WEIGHT, 0.0, -0.01, 0.0],
    ]

    ctbr_setpoint_tensor_list = [
        torch.tensor(
            ctbr_setpoint,
            dtype=torch.float,
            device=env.device,
        ).expand(num_envs, -1)
        for ctbr_setpoint in ctbr_setpoint_list
    ]

    # Clamp setpoint between environment action bounds to prevent
    # user defined setpoints from exceeding action bounds
    ctbr_setpoint_tensor_list = [
        torch.clamp(
            ctbr_setpoint_tensor,
            min=ctbr_setpoint_bounds[:, 0],
            max=ctbr_setpoint_bounds[:, 1],
        )
        for ctbr_setpoint_tensor in ctbr_setpoint_tensor_list
    ]

    # Calculate env actions from setpoints
    ctbr_env_action_list = []
    for ctbr_setpoint_tensor in ctbr_setpoint_tensor_list:
        # Normalize CTBR controller setpoints to a [-1, 1] range
        # to get the action expected by the environment
        ctbr_env_action = linear_interpolate(
            x=ctbr_setpoint_tensor,
            x_min=ctbr_setpoint_bounds[:, 0],
            x_max=ctbr_setpoint_bounds[:, 1],
            y_min=-1,
            y_max=1,
        )

        ctbr_env_action_list.append(ctbr_env_action)

    # Simulate drone environment
    print("Drone CTBR control simulation")

    num_setpoints = len(ctbr_setpoint_list)
    steps_per_command = num_steps // len(ctbr_env_action_list)
    prev_action_idx = -1
    with torch.no_grad():
        for step in range(num_steps):
            # Calculate action idx for ctbr action selection
            action_idx = step // steps_per_command
            # Avoid out-of-bounds error
            action_idx = min(action_idx, len(ctbr_env_action_list) - 1)

            ctbr_env_action = ctbr_env_action_list[action_idx]

            if action_idx != prev_action_idx:
                if prev_action_idx != -1:
                    print()
                print(
                    f"  [{action_idx + 1}/{num_setpoints}] CTBR setpoint: "
                    f"{ctbr_setpoint_list[action_idx]}\n"
                )
                prev_action_idx = action_idx

            # Step simulation
            env.step(ctbr_env_action)

            # Print control error for the env of idx 0
            ctbr_setpoint = ctbr_setpoint_tensor_list[action_idx]
            ctbr_thrust_setpoint = ctbr_setpoint[0, 0]
            ctbr_measurement = torch.hstack(
                [ctbr_thrust_setpoint, env.base_ang_vel[0, :]]
            )
            # Note: The total thrust is directly applied in the drone env,
            # so we assume it immediately matches its setpoint

            print_formatted_control_data(
                setpoint=ctbr_setpoint[0, :], measurement=ctbr_measurement
            )

    print("\nControl simulation finished.")


def print_formatted_control_data(
    setpoint: torch.Tensor, measurement: torch.Tensor
) -> None:
    setpoint_array = setpoint.cpu().numpy()
    measurement_array = measurement.cpu().numpy()

    # Calculate error
    error_array = setpoint_array - measurement_array

    # Format arrays for printing
    formatted_measurement = ", ".join(
        [f"{measurement:>7.5f}" for measurement in measurement_array]
    )
    formatted_error = ", ".join([f"{error:>10.3e}" for error in error_array])

    print(f"\033[F\033[K        Measurement: [{formatted_measurement}]")
    print(
        f"\033[K        Error: [{formatted_error}]",
        end="",
        flush=True,
    )


if __name__ == "__main__":
    main()
