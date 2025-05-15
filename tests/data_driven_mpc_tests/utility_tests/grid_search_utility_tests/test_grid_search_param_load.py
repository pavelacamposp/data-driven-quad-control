from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.grid_search_param_load import (  # noqa: E501
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
