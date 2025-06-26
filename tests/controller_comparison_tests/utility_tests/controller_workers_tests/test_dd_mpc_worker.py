import multiprocessing as mp
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from direct_data_driven_mpc.nonlinear_data_driven_mpc_controller import (
    NonlinearDataDrivenMPCController,
)

from data_driven_quad_control.comparison.utilities.controller_comparison_config import (  # noqa: E501
    DDMPCControllerInitData,
    EnvTargetSignal,
)
from data_driven_quad_control.comparison.utilities.controller_workers.dd_mpc_worker import (  # noqa: E501
    dd_mpc_controller_worker,
)

DD_MPC_CONTROLLER_CREATION_PATCH_PATH = (
    "data_driven_quad_control.comparison.utilities.controller_workers."
    "dd_mpc_worker.create_nonlinear_data_driven_mpc_controller"
)


@patch(DD_MPC_CONTROLLER_CREATION_PATCH_PATH)
def test_dd_mpc_controller_worker(
    mock_create_controller: Mock,
    mock_dd_mpc_controller: NonlinearDataDrivenMPCController,
) -> None:
    # Mock return value of `create_nonlinear_data_driven_mpc_controller`
    mock_create_controller.return_value = mock_dd_mpc_controller

    # Ensure mocked controller uses single stepping to simplify testing
    mock_dd_mpc_controller.n_mpc_step = 1

    # Create dummy queues and initialization data
    dummy_target_signal_queue: mp.Queue = mp.Queue()
    dummy_action_queue: mp.Queue = mp.Queue()
    dummy_obs_queue: mp.Queue = mp.Queue()
    dummy_init_data = DDMPCControllerInitData(
        controller_config=Mock(), u_N=np.zeros((1, 10)), y_N=np.zeros((1, 10))
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
    dummy_obs_queue.put(np.array([[0.0], [0.0]]))
    dummy_obs_queue.put(np.array([[0.1], [0.1]]))

    dd_mpc_controller_worker(
        env_idx=0,
        dd_mpc_controller_init_data=dummy_init_data,
        target_signal_queue=dummy_target_signal_queue,
        action_queue=dummy_action_queue,
        dd_mpc_obs_queue=dummy_obs_queue,
    )

    # Validate that the controller was initialized
    mock_create_controller.assert_called_once()

    # Verify that exactly two control actions were produced by the controller
    try:
        dummy_action_queue.get(timeout=1)
        dummy_action_queue.get(timeout=1)
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")


@pytest.mark.parametrize("target_signal_done_first", [True, False])
@patch(DD_MPC_CONTROLLER_CREATION_PATCH_PATH)
def test_dd_mpc_worker_exit_on_done(
    mock_create_controller: Mock,
    target_signal_done_first: bool,
    mock_dd_mpc_controller: NonlinearDataDrivenMPCController,
) -> None:
    # Mock return value of `create_nonlinear_data_driven_mpc_controller`
    mock_create_controller.return_value = mock_dd_mpc_controller

    # Ensure mocked controller uses single stepping to simplify testing
    mock_dd_mpc_controller.n_mpc_step = 1

    # Create dummy queues and initialization data
    dummy_target_signal_queue: mp.Queue = mp.Queue()
    dummy_action_queue: mp.Queue = mp.Queue()
    dummy_obs_queue: mp.Queue = mp.Queue()
    dummy_init_data = DDMPCControllerInitData(
        controller_config=Mock(),
        u_N=np.zeros((1, 1)),
        y_N=np.zeros((1, 1)),
    )

    # Send dummy target signal and observation to queues
    dummy_drone_pos = torch.zeros((10, 1))
    dummy_target_signal_queue.put(
        EnvTargetSignal(
            target_pos=dummy_drone_pos,
            is_new_target=False,
            done=target_signal_done_first,
        )
    )
    dummy_obs_queue.put(dummy_drone_pos)

    # If the first signal doesn't terminate the worker execution, send a second
    # one with `done = True` to ensure termination after the second iteration
    if not target_signal_done_first:
        dummy_target_signal_queue.put(
            EnvTargetSignal(
                target_pos=dummy_drone_pos,
                is_new_target=False,
                done=True,
            )
        )
        dummy_obs_queue.put(dummy_drone_pos)

    dd_mpc_controller_worker(
        env_idx=0,
        dd_mpc_controller_init_data=dummy_init_data,
        target_signal_queue=dummy_target_signal_queue,
        action_queue=dummy_action_queue,
        dd_mpc_obs_queue=dummy_obs_queue,
    )

    # Verify number of actions sent via the action queue
    # based on the initial done signal
    if target_signal_done_first:
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
