import multiprocessing as mp
from unittest.mock import Mock, patch

import pytest
import torch

from data_driven_quad_control.comparison.utilities.controller_comparison_config import (  # noqa: E501
    EnvTargetSignal,
    TrackingControllerInitData,
)
from data_driven_quad_control.comparison.utilities.controller_workers.tracking_worker import (  # noqa: E501
    tracking_controller_worker,
)
from data_driven_quad_control.controllers.tracking.tracking_controller import (  # noqa: E501
    DroneTrackingController,
)
from data_driven_quad_control.controllers.tracking.tracking_controller_config import (  # noqa: E501
    TrackingCtrlDroneState,
)

TRACKING_CONTROLLER_PATCH_PATH = (
    "data_driven_quad_control.comparison.utilities.controller_workers."
    "tracking_worker.DroneTrackingController"
)


@patch(TRACKING_CONTROLLER_PATCH_PATH)
def test_tracking_worker(
    mock_tracking_controller_class: Mock,
    mock_tracking_controller: DroneTrackingController,
) -> None:
    # Mock return value of `DroneTrackingController`
    mock_tracking_controller_class.return_value = mock_tracking_controller

    # Create dummy queues and initialization data
    dummy_target_signal_queue: mp.Queue = mp.Queue()
    dummy_action_queue: mp.Queue = mp.Queue()
    dummy_obs_queue: mp.Queue = mp.Queue()

    dummy_tracking_drone_state = TrackingCtrlDroneState(
        X=torch.zeros((3, 1)),
        Q=torch.zeros((4, 1)),
    )
    dummy_init_data = TrackingControllerInitData(
        controller_config=Mock(),
        controller_dt=0.1,
        initial_state=dummy_tracking_drone_state,
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
    dummy_obs_queue.put(dummy_tracking_drone_state)
    dummy_obs_queue.put(dummy_tracking_drone_state)

    tracking_controller_worker(
        env_idx=0,
        env_device=torch.device("cpu"),
        tracking_controller_init_data=dummy_init_data,
        target_signal_queue=dummy_target_signal_queue,
        action_queue=dummy_action_queue,
        tracking_obs_queue=dummy_obs_queue,
    )

    # Verify that exactly two control actions were produced by the controller
    try:
        dummy_action_queue.get(timeout=1)
        dummy_action_queue.get(timeout=1)
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")


@pytest.mark.parametrize("done_signal_first", [True, False])
@patch(TRACKING_CONTROLLER_PATCH_PATH)
def test_tracking_worker_exit_on_done(
    mock_tracking_controller_class: Mock,
    done_signal_first: bool,
    mock_tracking_controller: DroneTrackingController,
) -> None:
    # Mock return value of `DroneTrackingController`
    mock_tracking_controller_class.return_value = mock_tracking_controller

    # Create dummy queues and initialization data
    dummy_target_signal_queue: mp.Queue = mp.Queue()
    dummy_action_queue: mp.Queue = mp.Queue()
    dummy_obs_queue: mp.Queue = mp.Queue()

    dummy_tracking_drone_state = TrackingCtrlDroneState(
        X=torch.zeros((3, 1)),
        Q=torch.zeros((4, 1)),
    )
    dummy_init_data = TrackingControllerInitData(
        controller_config=Mock(),
        controller_dt=0.1,
        initial_state=dummy_tracking_drone_state,
    )

    # Send dummy target signal and observation to queues
    dummy_target_signal_queue.put(
        EnvTargetSignal(
            target_pos=torch.zeros((3, 1)),
            is_new_target=False,
            done=done_signal_first,
        )
    )
    dummy_obs_queue.put(dummy_tracking_drone_state)

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
        dummy_obs_queue.put(dummy_tracking_drone_state)

    tracking_controller_worker(
        env_idx=0,
        env_device=torch.device("cpu"),
        tracking_controller_init_data=dummy_init_data,
        target_signal_queue=dummy_target_signal_queue,
        action_queue=dummy_action_queue,
        tracking_obs_queue=dummy_obs_queue,
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
