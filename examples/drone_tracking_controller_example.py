"""
Drone Control using a Tracking Controller

This script demonstrates the control of drones in a vectorized Genesis
environment (`HoverEnv`) using tracking controllers
(`DroneTrackingController`) for position and yaw tracking.
"""

import argparse
import math

import genesis as gs
import torch

from data_driven_quad_control.controllers.ctbr.ctbr_controller import (
    DroneCTBRController,
)
from data_driven_quad_control.envs.hover_env import HoverEnv
from data_driven_quad_control.envs.hover_env_config import (
    EnvActionType,
    EnvCTBRControllerConfig,
    EnvDroneParams,
    get_cfgs,
)
from data_driven_quad_control.utilities.drone_environment import (
    update_env_target_pos,
)
from data_driven_quad_control.utilities.drone_tracking_controller import (
    create_drone_tracking_controller,
    hover_at_target,
)

# Mapping for env action types
ENV_ACTION_TYPES_MAP = {
    "rpms": EnvActionType.ROTOR_RPMS,
    "ctbr": EnvActionType.CTBR,
    "ctbr_fixed_yaw": EnvActionType.CTBR_FIXED_YAW,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drone CTBR controller example in Genesis"
    )

    parser.add_argument(
        "--num_envs", type=int, default=1, help="The number of parallel envs."
    )
    parser.add_argument(
        "--action_type",
        type=str,
        choices=list(ENV_ACTION_TYPES_MAP.keys()),
        default="rpms",
        help="The environment action type.",
    )
    parser.add_argument(
        "--target_pos_noise",
        action="store_true",
        help="Enable the addition of noise to target positions.",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Disable GUI viewer."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_envs = args.num_envs
    action_type = args.action_type
    target_pos_noise = args.target_pos_noise
    headless = args.headless

    print("--- Drone Tracking Controller Example ---")
    print("-" * 41)

    # Initialize Genesis simulator
    print("Initializing Genesis simulator")

    gs.init(backend=gs.gpu, logging_level="error")

    # Load environment configuration
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    # Set up visualization
    env_cfg["visualize_target"] = True
    env_cfg["max_visualize_FPS"] = 100  # Sim visualization FPS

    # Create environment
    action_type = ENV_ACTION_TYPES_MAP[action_type]

    print(f"Creating drone environment with {action_type.name} actions")

    show_viewer = not headless
    env = HoverEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
        auto_target_updates=False,  # Disable automatic target position updates
        action_type=action_type,
    )

    # Reset environment
    env.reset()

    # Define a sequence of target positions (x, y, z) and yaw angles (radians)
    target_pos_list = [
        [0.0, 0.0, 1.5],
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.5],
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 1.5],
    ]
    target_yaw_list = [0.0, 1.5 * math.pi, -0.5 * math.pi, 0.0, 2 * math.pi]

    # Convert targets to tensors
    target_pos_tensor_list = torch.tensor(
        target_pos_list, device=env.device, dtype=torch.float
    )
    target_yaw_tensor_list = torch.tensor(
        target_yaw_list, device=env.device, dtype=torch.float
    )

    # Create drone tracking controller
    print("Creating drone tracking controller")

    tracking_controller = create_drone_tracking_controller(env=env)

    # Create drone CTBR controller if env action type is ROTOR_RPMS
    # Note:
    #   - The tracking controller outputs CTBR actions and not RPMs directly.
    #   - For CTBR and CTBR_FIXED_YAW env action types, an internal CTBR
    #     controller is included within the drone environment.
    ctbr_controller = None
    if env.action_type is EnvActionType.ROTOR_RPMS:
        print("Creating an external CTBR controller")

        drone_params = EnvDroneParams.get()
        controller_config = EnvCTBRControllerConfig.get()
        ctbr_controller = DroneCTBRController(
            drone_params=drone_params,
            controller_config=controller_config,
            dt=env.step_dt,
            num_envs=num_envs,
            device=env.device,
        )
    else:
        print(
            "CTBR controller not created (already initialized within the env)"
        )

    # Command drones to hover at each target
    print("Drone tracking control simulation")

    noise_scale = 0.5  # Noise scale for target positions
    num_targets = len(target_pos_tensor_list)
    for i in range(num_targets):
        target_pos_tensor = target_pos_tensor_list[i]
        target_yaw_tensor = target_yaw_tensor_list[i]

        print(
            f"  [{i + 1}/{num_targets}] Hovering at target pos: "
            f"{target_pos_tensor.tolist()}, yaw: "
            f"{target_yaw_tensor.item():6.4f}"
        )

        target_pos = target_pos_tensor.repeat(num_envs, 1)
        target_yaw = target_yaw_tensor.expand(num_envs)

        # Add noise to target positions to show independent control behavior
        if target_pos_noise:
            target_pos += 2 * noise_scale * (torch.rand_like(target_pos) - 0.5)
            target_pos[:, 2] = torch.clamp(target_pos[:, 2], min=0.2, max=1.9)
            print("         * Added target position noise")

        # Update environment goal position for visualization
        update_env_target_pos(
            env=env, env_idx=list(range(num_envs)), target_pos=target_pos
        )

        # Command drones to hover at target positions and yaws
        hover_at_target(
            env=env,
            tracking_controller=tracking_controller,
            target_pos=target_pos,
            target_yaw=target_yaw,
            min_at_target_steps=10,
            error_threshold=5e-2,
            ctbr_controller=ctbr_controller,
        )

    print("Control simulation finished.")


if __name__ == "__main__":
    main()
