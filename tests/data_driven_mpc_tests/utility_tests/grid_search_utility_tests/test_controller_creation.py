from unittest.mock import Mock, patch

import numpy as np

from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.controller_creation import (  # noqa: E501
    create_dd_mpc_controller_for_combination,
)
from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.param_grid_search_config import (  # noqa: E501
    DDMPCCombinationParams,
    DDMPCFixedParams,
)

NONLINEAR_CONTROLLER_PATCH_PATH = (
    "data_driven_quad_control.data_driven_mpc.utilities.param_grid_search."
    "controller_creation.NonlinearDataDrivenMPCController"
)


@patch(NONLINEAR_CONTROLLER_PATCH_PATH)
def test_create_dd_mpc_controller_for_combination(
    mock_controller: Mock,
    test_combination_params: DDMPCCombinationParams,
    test_fixed_params: DDMPCFixedParams,
) -> None:
    # Retrieve controller parameters from test objects
    N = test_combination_params.N
    m = test_fixed_params.m
    p = test_fixed_params.p
    np_random = np.random.default_rng(0)

    # Create test parameters
    u_N = np_random.uniform(-1.0, 1.0, (N, m))
    y_N = np.ones((N, p))
    y_r = np.ones((p, 1))

    # Create controller for combination
    nonlinear_dd_mpc_controller = create_dd_mpc_controller_for_combination(
        u_N=u_N,
        y_N=y_N,
        y_r=y_r,
        combination_params=test_combination_params,
        fixed_params=test_fixed_params,
    )

    # Assert that the controller constructor was called and the
    # controller is returned
    assert mock_controller.called
    assert nonlinear_dd_mpc_controller == mock_controller.return_value
