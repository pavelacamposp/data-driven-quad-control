"""
Load configuration for nonlinear data-driven MPC parameter grid search

This module provides functionality for loading configuration parameters for a
nonlinear data-driven MPC parameter grid search from a YAML config file.
"""

import numpy as np
from direct_data_driven_mpc.nonlinear_data_driven_mpc_controller import (
    AlphaRegType,
)

from data_driven_quad_control.utilities.config_utils import load_yaml_config

from .param_grid_search_config import (
    DDMPCEvaluationParams,
    DDMPCFixedParams,
    DDMPCGridSearchParams,
    DDMPCInitialDataCollectionParams,
    DDMPCParameterGrid,
)

# Nonlinear Data-Driven MPC: Alpha regularization type map
ALPHA_REG_TYPE_MAP = {
    0: AlphaRegType.APPROXIMATED,
    1: AlphaRegType.PREVIOUS,
    2: AlphaRegType.ZERO,
}


def load_dd_mpc_grid_search_params(
    m: int, p: int, config_path: str, verbose: int = 0
) -> DDMPCGridSearchParams:
    """
    Load configuration parameters for a nonlinear data-driven MPC parameter
    grid search from a YAML config file.

    Args:
        m (int): The number of control inputs.
        p (int): The number of drone system outputs.
        config_path (str): The path to the YAML configuration file containing
            the parameters for a nonlinear data-driven MPC parameter grid
            search.
        verbose (int): The verbosity level: 0 = no output, 1 = minimal output,
            2 = detailed output. Defaults to 0.

    Returns:
        DDMPCGridSearchParams: A tuple containing nonlinear data-driven MPC
            grid search parameters:
            - DDMPCInitialDataCollectionParams: The parameters for the initial
                input-output data collection phase.
            - DDMPCFixedParams: The parameters that remain fixed across the
                grid search.
            - DDMPCEvaluationParams: The parameters used for evaluating
                controllers.
            - DDMPCParameterGrid: The parameter grid defining the combinations
                to search over.
    """
    # Load parameters from config file
    config = load_yaml_config(config_path)

    if verbose > 1:
        print(
            "    Parameters for the nonlinear data-driven MPC parameter grid "
            f"search loaded from {config_path}"
        )

    # Load initial input-output data collection params
    init_data_collection_params_raw = config["initial_data_collection"]
    init_data_collection_params = DDMPCInitialDataCollectionParams(
        init_hover_pos=np.array(
            init_data_collection_params_raw["init_hover_pos"], dtype=float
        ),
        u_range=np.array(
            init_data_collection_params_raw["u_range"], dtype=float
        ),
    )

    # Load fixed Data-Driven MPC params
    fixed_params_raw = config["fixed_params"]
    alpha_reg_type = ALPHA_REG_TYPE_MAP.get(
        fixed_params_raw["alpha_reg_type"], AlphaRegType.APPROXIMATED
    )
    fixed_params = DDMPCFixedParams(
        m=m,
        p=p,
        Q_weights=fixed_params_raw["Q_weights"],
        R_weights=fixed_params_raw["R_weights"],
        S_weights=fixed_params_raw["S_weights"],
        U=np.array(fixed_params_raw["U"], dtype=float),
        Us=np.array(fixed_params_raw["Us"], dtype=float),
        alpha_reg_type=alpha_reg_type,
        ext_out_incr_in=fixed_params_raw["ext_out_incr_in"],
        n_n_mpc_step=fixed_params_raw["n_n_mpc_step"],
    )

    # Load controller evaluation params
    eval_params_raw = config["evaluation_params"]
    setpoint_list = [
        np.array(setpoint, dtype=float).reshape(-1, 1)
        for setpoint in eval_params_raw["eval_setpoints"]
    ]
    eval_params = DDMPCEvaluationParams(
        eval_time_steps=eval_params_raw["eval_time_steps"],
        eval_setpoints=setpoint_list,
        max_target_dist_increment=eval_params_raw["max_target_dist_increment"],
        num_collections_per_N=eval_params_raw["num_collections_per_N"],
    )

    # Load Data-Driven MPC parameter grid
    parm_grid_raw = config["parameter_grid"]
    param_grid = DDMPCParameterGrid(
        N=parm_grid_raw["N"],
        n=parm_grid_raw["n"],
        L=parm_grid_raw["L"],
        lamb_alpha=[float(x) for x in parm_grid_raw["lamb_alpha"]],
        lamb_sigma=[float(x) for x in parm_grid_raw["lamb_sigma"]],
        lamb_alpha_s=[float(x) for x in parm_grid_raw["lamb_alpha_s"]],
        lamb_sigma_s=[float(x) for x in parm_grid_raw["lamb_sigma_s"]],
    )

    # Override unused parameter values if the `alpha`
    # regularization type is not "APPROXIMATED"
    if alpha_reg_type != AlphaRegType.APPROXIMATED:
        param_grid = param_grid._replace(
            lamb_alpha_s=[0.0], lamb_sigma_s=[0.0]
        )

    return (
        init_data_collection_params,
        fixed_params,
        eval_params,
        param_grid,
    )
