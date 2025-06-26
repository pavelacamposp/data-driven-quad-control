"""
Worker process for a Reinforcement Learning controller (trained PPO policy).

This module defines a parallel worker function that initializes and runs a
Reinforcement Learning controller (trained PPO policy) in closed loop to
control the position of a drone in a vectorized environment.

The worker communicates with the main process via multiprocessing queues.
"""

import contextlib
import io
from typing import Any

import torch
import torch.multiprocessing as mp
from rsl_rl.runners import OnPolicyRunner

from ..controller_comparison_config import (
    EnvTargetSignal,
    RLControllerInitData,
)


class DummyHoverEnv:
    """
    A dummy `HoverEnv` class used to create `OnPolicyRunner` runners without
    requiring access to the real simulation environment.

    This class is useful when creating `OnPolicyRunner` instances in parallel
    environments, since Taichi objects cannot be directly serialized with
    `pickle`.
    """

    def __init__(self, specs: dict[str, Any]):
        self.unwrapped = self
        self.device = specs["device"]
        self.step_dt = specs["step_dt"]
        self.num_envs = specs["num_envs"]
        self.num_obs = specs["num_obs"]
        self.num_actions = specs["num_actions"]
        self.max_episode_length = specs["max_episode_length"]

        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device
        )
        self.reward_buf = torch.zeros((self.num_envs,), device=self.device)
        self.reset_buf = torch.zeros_like(self.reward_buf)

    def get_observations(self) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.obs_buf, {"observations": {}}

    def step(self, actions: torch.Tensor) -> Any:
        return self.obs_buf, self.reward_buf, self.reset_buf, {}

    def reset(self) -> tuple[torch.Tensor, None]:
        return self.obs_buf, None


def rl_controller_worker(
    env_idx: int,
    env_specs: dict[str, Any],
    rl_controller_init_data: RLControllerInitData,
    target_signal_queue: mp.Queue,
    action_queue: mp.Queue,
    rl_obs_queue: mp.Queue,
) -> None:
    """
    Parallel worker for a Reinforcement Learning (RL) controller (trained PPO
    policy).

    This function initializes the RL controller (loads a policy from a trained
    PPO model) from the provided initialization data and runs it in closed loop
    to control the position of a drone in simulation.

    The worker communicates with the main process via multiprocessing queues
    to perform the following tasks:
    - Receive target position updates and simulation termination signals.
    - Receive drone environment observations.
    - Send control actions.

    Args:
        env_idx (int): The index of the drone controlled by the RL controller.
        env_specs (dict[str, Any]): Specifications required to construct the
            dummy `HoverEnv` environment instance used for loading the PPO
            policy.
        rl_controller_init_data (RLControllerInitData): The RL controller
            initialization data.
        target_signal_queue (mp.Queue): A queue used for receiving
            `EnvTargetSignal` messages from the main process. Each message
            includes the current target position, a flag indicating whether
            it's a new target (used to trigger controller target updates), and
            a done signal indicating whether the simulation will be terminated.
        action_queue (mp.Queue): A queue used for sending control actions to
            the main process for environment stepping.
        rl_obs_queue (mp.Queue): A queue used for receiving environment
            observations from the main process. Each observation is the raw
            observation buffer from the `HoverEnv` environment.
    """
    # Create dummy env with the attributes required by
    # `OnPolicyRunner` to initialize the PPO policy
    env = DummyHoverEnv(env_specs)

    # Load PPO policy from model checkpoint
    # Suppress stdout during runner initialization
    with contextlib.redirect_stdout(io.StringIO()):
        runner = OnPolicyRunner(
            env, rl_controller_init_data.train_cfg, None, device=env.device
        )

    runner.load(rl_controller_init_data.model_path)

    # Get inference policy from PPO runner
    policy = runner.get_inference_policy(device=env.device)

    # Initialize observation
    obs = rl_controller_init_data.initial_observation

    # Evaluate policy in simulation
    while True:
        # Receive target signal from the main process
        target_signal: EnvTargetSignal = target_signal_queue.get()

        # Compute action
        action = policy(obs)

        # Send action to the main process
        action_queue.put((env_idx, action.detach()))

        # Get observations from vectorized environment
        obs = rl_obs_queue.get()

        # Stop simulation if main process signals termination
        if target_signal.done:
            break
