import numpy as np

from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.grid_search_param_loader import (  # noqa: E501
    load_dd_mpc_grid_search_params,
)
from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.param_grid_search_config import (  # noqa: E501
    DDMPCEvaluationParams,
    DDMPCFixedParams,
    DDMPCInitialDataCollectionParams,
    DDMPCParameterGrid,
)


def test_load_dd_mpc_grid_search_params(
    test_grid_search_params_path: str,
) -> None:
    # Load grid search parameters from configuration file
    initial_data_params, fixed_params, eval_params, param_grid = (
        load_dd_mpc_grid_search_params(
            m=3,
            p=3,
            config_path=test_grid_search_params_path,
        )
    )

    # Verify that each component was correctly loaded
    assert isinstance(initial_data_params, DDMPCInitialDataCollectionParams)
    assert isinstance(fixed_params, DDMPCFixedParams)
    assert isinstance(eval_params, DDMPCEvaluationParams)
    assert isinstance(param_grid, DDMPCParameterGrid)

    # Verify loaded parameters based on known values from the test config file
    # Verify comparison parameters
    expected_hover_pos = np.array([0.0, 0.0, 1.0])
    np.testing.assert_equal(
        initial_data_params.init_hover_pos, expected_hover_pos
    )

    # Verify fixed parameters
    expected_Q_weigth = [1, 1, 1]
    assert isinstance(fixed_params.Q_weights, list)
    assert fixed_params.Q_weights == expected_Q_weigth

    # Verify evaluation parameters
    expected_eval_time_steps = 1
    assert eval_params.eval_time_steps == expected_eval_time_steps

    expected_eval_setpoint = np.array([1.0, 1.0, 1.0]).reshape(-1, 1)
    assert isinstance(eval_params.eval_setpoints, list)
    np.testing.assert_equal(
        eval_params.eval_setpoints[0], expected_eval_setpoint
    )

    # Verify parameter grid
    expected_N_list_length = 1
    assert isinstance(param_grid.N, list)
    assert len(param_grid.N) == expected_N_list_length
