"""
Script to evaluate a trained PPO agent for drone hovering.

This script loads a trained PPO agent from a model, restores its policy, and
evaluates it in simulation using the `HoverEnv` vectorized environment.
"""

import argparse
import json
import os

import genesis as gs
import torch
from rsl_rl.runners import OnPolicyRunner

from data_driven_quad_control.envs.hover_env import HoverEnv
from data_driven_quad_control.learning.config.hover_ppo_config import (
    ENV_ACTION_TYPES_MAP,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO agent for drone hovering"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help=(
            "The path to the experiment directory containing `cfgs.json` and "
            "`model_*.pt` files."
        ),
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="The number of parallel environments (drones) to create.",
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        default=300,
        help="The index of the model checkpoint to load.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Enable video recording of the simulation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("--- Trained PPO Model Evaluation ---")
    print("-" * 36)

    # Initialize Genesis simulator
    print("Initializing Genesis simulator")

    gs.init(backend=gs.gpu, logging_level="error")

    # Load environment and training configuration
    log_dir = args.log_dir
    cfgs_path = f"{log_dir}/cfgs.json"

    print(f"Loading training configuration from {cfgs_path}")

    with open(cfgs_path, "r") as f:
        cfgs = json.load(f)

    env_cfg = cfgs["env_cfg"]
    obs_cfg = cfgs["obs_cfg"]
    reward_cfg = cfgs["reward_cfg"]
    command_cfg = cfgs["command_cfg"]
    train_cfg = cfgs["train_cfg"]
    action_type_str = cfgs["action_type_str"]

    # Set up visualization
    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = args.record  # Enable camera for recording
    env_cfg["max_visualize_FPS"] = 100  # Sim visualization FPS

    # Customize camera configuration
    camera_config = {
        "res": (640, 480),  # Used for video recording
        "pos": (3.0, 0.0, 3.0),
        "lookat": (0.0, 0.0, 1.5),
        "fov": 40,
    }

    # Create vectorized environment
    action_type = ENV_ACTION_TYPES_MAP[action_type_str]

    print(f"Creating drone environment with {action_type.name} actions")

    env = HoverEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        action_type=action_type,
        camera_config=camera_config,
    )
    obs, _ = env.reset()

    # Load PPO policy from model checkpoint
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=env.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")

    print(f"Loading PPO policy from model checkpoint {resume_path}")

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.device)

    # Evaluate policy in simulation
    print("\nTrained Policy Evaluation")
    print("-" * 25)

    max_sim_step = int(
        env_cfg["episode_length_s"]
        * env_cfg["max_visualize_FPS"]
        / env_cfg["decimation"]  # Take into account decimation
    )
    with torch.no_grad():
        if args.record:
            print("Recording: starting video capture")
            env.start_recording()

        print("Evaluating trained policy")

        for _ in range(max_sim_step):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        if args.record:
            output_file = "drone_eval.mp4"

            print(f"Recording: Stopping and saving recording to {output_file}")

            env.stop_and_save_recording(
                save_to_filename=output_file,
                fps=env_cfg["max_visualize_FPS"],
            )

    print("\nPolicy evaluation finished.")


if __name__ == "__main__":
    main()
