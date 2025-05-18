from unittest.mock import Mock, patch

import numpy as np
import torch

from data_driven_quad_control.comparison.utilities.controller_comparison_config import (  # noqa: E501
    DDMPCControllerInitData,
)
from data_driven_quad_control.comparison.utilities.dd_mpc_initial_data_collection import (  # noqa: E501
    get_data_driven_mpc_controller_init_data,
)
from data_driven_quad_control.controllers.tracking.tracking_controller import (
    DroneTrackingController,
)
from data_driven_quad_control.envs.hover_env import HoverEnv

HOVER_AT_TARGET_PATCH_PATH = (
    "data_driven_quad_control.comparison.utilities."
    "dd_mpc_initial_data_collection.hover_at_target"
)

COLLECT_INITIAL_IO_DATA_PATCH_PATH = (
    "data_driven_quad_control.comparison.utilities."
    "dd_mpc_initial_data_collection.collect_initial_input_output_data"
)


@patch(COLLECT_INITIAL_IO_DATA_PATCH_PATH)
@patch(HOVER_AT_TARGET_PATCH_PATH)
def test_get_data_driven_mpc_controller_init_data(
    mock_hover_at_target: Mock,
    mock_collect_initial_input_output_data: Mock,
    mock_env: HoverEnv,
    mock_tracking_controller: DroneTrackingController,
) -> None:
    # Define test parameters
    test_u_range = np.array([[-1.0, 1.0]] * 3)
    test_N = 5
    test_dd_mpc_controller_config = {
        "U": np.array([[-1.0, 1.0]] * 3),
        "u_range": test_u_range,
        "N": test_N,
    }
    init_hover_pos = torch.tensor([0.0, 0.0, 1.5])

    # Mock return value of `collect_initial_input_output_data`
    test_u_N = np.zeros((test_N, 3))
    test_y_N = np.ones((test_N, 3))
    mock_collect_initial_input_output_data.return_value = (test_u_N, test_y_N)

    dd_mpc_controller_init_data = get_data_driven_mpc_controller_init_data(
        env=mock_env,
        dd_mpc_env_idx=0,
        dd_mpc_controller_config=test_dd_mpc_controller_config,
        init_hover_pos=init_hover_pos,
        stabilizing_controller=mock_tracking_controller,
        np_random=np.random.default_rng(0),
    )

    # Verify the type of the returned object
    assert isinstance(dd_mpc_controller_init_data, DDMPCControllerInitData)

    # Check that the controller config is passed without modifications
    assert (
        dd_mpc_controller_init_data.controller_config
        == test_dd_mpc_controller_config
    )

    # Check shapes of input-output trajectory arrays
    np.testing.assert_array_equal(dd_mpc_controller_init_data.u_N, test_u_N)
    np.testing.assert_array_equal(dd_mpc_controller_init_data.y_N, test_y_N)
