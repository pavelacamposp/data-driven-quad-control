import multiprocessing as mp
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch

from data_driven_quad_control.comparison.utilities.controller_comparison_config import (  # noqa: E501
    EnvTargetSignal,
    RLControllerInitData,
)
from data_driven_quad_control.comparison.utilities.controller_workers.rl_worker import (  # noqa: E501
    rl_controller_worker,
)

RL_RUNNER_PATCH_PATH = (
    "data_driven_quad_control.comparison.utilities.controller_workers."
    "rl_worker.OnPolicyRunner"
)


class MockOnPolicyRunner:
    def __init__(self, policy: Any):
        self._policy = policy

    def load(self, path: str, load_optimizer: bool = True) -> int:
        return 0

    def get_inference_policy(self, device: torch.device) -> Any:
        return self._policy


@pytest.fixture
def mock_on_policy_runner(mock_rl_policy: Any) -> MockOnPolicyRunner:
    return MockOnPolicyRunner(policy=mock_rl_policy)


@patch(RL_RUNNER_PATCH_PATH)
def test_rl_worker(
    mock_runner_class: Mock,
    mock_on_policy_runner: MockOnPolicyRunner,
) -> None:
    # Define test parameters
    dummy_env_specs = {
        "device": torch.device("cpu"),
        "step_dt": 0.1,
        "num_envs": 1,
        "num_obs": 1,
        "num_actions": 1,
        "max_episode_length": 1,
    }

    # Mock `OnPolicyRunner` instance
    mock_runner = mock_on_policy_runner

    # Mock return value of `OnPolicyRunner`
    mock_runner_class.return_value = mock_runner

    # Create dummy queues and initialization data
    dummy_target_signal_queue: mp.Queue = mp.Queue()
    dummy_action_queue: mp.Queue = mp.Queue()
    dummy_obs_queue: mp.Queue = mp.Queue()

    dummy_env_observation = torch.zeros((10, 1))
    dummy_init_data = RLControllerInitData(
        train_cfg=Mock(),
        model_path="dummy_model.pt",
        initial_observation=dummy_env_observation,
    )

    # Send dummy signals to queues
    target_signal = EnvTargetSignal(
        target_pos=torch.zeros((3, 1)), is_new_target=True, done=False
    )
    target_signal_done = EnvTargetSignal(
        target_pos=torch.zeros((3, 1)), is_new_target=False, done=True
    )

    dummy_target_signal_queue.put(target_signal)

    # Send target signal to terminate the worker execution
    dummy_target_signal_queue.put(target_signal_done)

    # Fill obs_queue with dummy observations
    dummy_obs_queue.put(dummy_env_observation)
    dummy_obs_queue.put(dummy_env_observation)

    rl_controller_worker(
        env_idx=0,
        env_specs=dummy_env_specs,
        rl_controller_init_data=dummy_init_data,
        target_signal_queue=dummy_target_signal_queue,
        action_queue=dummy_action_queue,
        rl_obs_queue=dummy_obs_queue,
    )

    # Verify that exactly two control actions were produced by the controller
    try:
        dummy_action_queue.get(timeout=1)
        dummy_action_queue.get(timeout=1)
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")


@pytest.mark.parametrize("done_signal_first", [True, False])
@patch(RL_RUNNER_PATCH_PATH)
def test_rl_worker_exit_on_done(
    mock_runner_class: Mock,
    done_signal_first: bool,
    mock_on_policy_runner: MockOnPolicyRunner,
) -> None:
    # Define test parameters
    dummy_env_specs = {
        "device": torch.device("cpu"),
        "step_dt": 0.1,
        "num_envs": 1,
        "num_obs": 1,
        "num_actions": 1,
        "max_episode_length": 1,
    }

    # Mock `OnPolicyRunner` instance
    mock_runner = mock_on_policy_runner

    # Mock return value of `OnPolicyRunner`
    mock_runner_class.return_value = mock_runner

    # Create dummy queues and initialization data
    dummy_target_signal_queue: mp.Queue = mp.Queue()
    dummy_action_queue: mp.Queue = mp.Queue()
    dummy_obs_queue: mp.Queue = mp.Queue()

    dummy_env_observation = torch.zeros((10, 1))
    dummy_init_data = RLControllerInitData(
        train_cfg=Mock(),
        model_path="dummy_model.pt",
        initial_observation=dummy_env_observation,
    )

    # Send dummy target signal and observation to queues
    dummy_target_signal_queue.put(
        EnvTargetSignal(
            target_pos=torch.zeros((3, 1)),
            is_new_target=False,
            done=done_signal_first,
        )
    )
    dummy_obs_queue.put(dummy_env_observation)

    # If the first signal doesn't terminate the worker execution, send a second
    # one with `done = True` to ensure termination after the second iteration
    if not done_signal_first:
        dummy_target_signal_queue.put(
            EnvTargetSignal(
                target_pos=torch.zeros((3, 1)),
                is_new_target=False,
                done=True,
            )
        )
        dummy_obs_queue.put(dummy_env_observation)

    rl_controller_worker(
        env_idx=0,
        env_specs=dummy_env_specs,
        rl_controller_init_data=dummy_init_data,
        target_signal_queue=dummy_target_signal_queue,
        action_queue=dummy_action_queue,
        rl_obs_queue=dummy_obs_queue,
    )

    # Verify number of actions sent via the action queue
    # based on the done signal
    if done_signal_first:
        # If done immediately, only one action should be sent
        try:
            dummy_action_queue.get(timeout=1)
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")
    else:
        # Otherwise, an additional action should be set, as we iterate
        # for one more step to send the `done = True` signal
        try:
            dummy_action_queue.get(timeout=1)
            dummy_action_queue.get(timeout=1)
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")
