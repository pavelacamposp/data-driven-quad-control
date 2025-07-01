"""
Script to train a PPO agent for drone hovering.

This script creates a vectorized environment (`HoverEnv`) and uses it to train
a PPO agent in parallel.
"""

import argparse
import json
import os
import shutil

import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from data_driven_quad_control.envs.hover_env import HoverEnv
from data_driven_quad_control.envs.hover_env_config import get_cfgs
from data_driven_quad_control.learning.config.hover_ppo_config import (
    ENV_ACTION_TYPES_MAP,
    get_train_cfg,
)
from data_driven_quad_control.learning.config.reward_config import (
    get_reward_cfg_by_action_type,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO agent for hovering a drone"
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default="drone-hovering",
        help="The experiment name.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=8192,
        help="The number of parallel environments (drones) used for training.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=300,
        help="The number of training iterations.",
    )
    parser.add_argument(
        "--action_type",
        type=str,
        choices=list(ENV_ACTION_TYPES_MAP.keys()),
        default="rpms",
        help="The environment action type.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Visualize the simulation during training.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("--- PPO Model Training ---")
    print("-" * 26)

    # Initialize Genesis simulator
    print("Initializing Genesis simulator")

    gs.init(backend=gs.gpu, logging_level="error")

    # Retrieve environment and training configuration
    log_dir = f"logs/{args.exp_name}_{args.action_type}"
    env_cfg, obs_cfg, _, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # Retrieve reward config based on action type
    action_type = ENV_ACTION_TYPES_MAP[args.action_type]
    reward_cfg = get_reward_cfg_by_action_type(action_type)

    # Save configurations to a file
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    cfgs = {
        "env_cfg": env_cfg,
        "obs_cfg": obs_cfg,
        "reward_cfg": reward_cfg,
        "command_cfg": command_cfg,
        "train_cfg": train_cfg,
        "action_type_str": args.action_type,
    }
    cfgs_path = f"{log_dir}/cfgs.json"

    print(f"Saving training configuration to {cfgs_path}")

    with open(cfgs_path, "w") as f:
        json.dump(cfgs, f, indent=2)

    # Set up target visualization
    if args.vis:
        env_cfg["visualize_target"] = True

    # Create vectorized environment
    print(f"Creating drone environment with {action_type.name} actions")

    env = HoverEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
        action_type=action_type,
    )

    # Create `OnPolicyRunner` model
    print("\nPPO Policy Training")
    print("-" * 19)
    print("Creating model")

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=env.device)

    # Start model training
    print("Starting model training")

    runner.learn(
        num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
    )

    print("\nTraining finished successfully.")


if __name__ == "__main__":
    main()
