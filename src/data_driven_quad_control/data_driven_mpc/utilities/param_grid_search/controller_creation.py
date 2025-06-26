"""
Create nonlinear data-driven MPC controllers for a parameter grid search.

This module provides functionality for creating nonlinear data-driven MPC
controllers using fixed and combination parameters from a controller parameter
grid search.
"""

import numpy as np
from direct_data_driven_mpc.nonlinear_data_driven_mpc_controller import (
    NonlinearDataDrivenMPCController,
)

from .param_grid_search_config import DDMPCCombinationParams, DDMPCFixedParams


def create_dd_mpc_controller_for_combination(
    u_N: np.ndarray,
    y_N: np.ndarray,
    y_r: np.ndarray,
    combination_params: DDMPCCombinationParams,
    fixed_params: DDMPCFixedParams,
) -> NonlinearDataDrivenMPCController:
    """
    Create a Nonlinear Data-Driven MPC controller from initial input-output
    measurement segments, an output setpoint, and a set of controller
    parameters from a grid search (combination and fixed).

    This function initializes and returns a `NonlinearDataDrivenMPCController`
    by constructing the appropriate cost matrices from both combination and
    fixed controller parameters.

    Args:
        u_N (np.ndarray): An array of shape `(N, m)` representing a
            persistently exciting input sequence used for output data
            collection. `N` is the trajectory length and `m` is the number of
            control inputs.
        y_N (np.ndarray): An array of shape `(N, p)` representing the collected
            system output in response to `u_N`. `N` is the trajectory length
            and `p` is the number of drone system outputs.
        y_r (np.ndarray): The position tracking setpoint, shaped `(p, 1)`.
        combination_params (DDMPCCombinationParams): A combination of nonlinear
            data-driven MPC parameters selected from the grid search.
        fixed_params (DDMPCFixedParams): The nonlinear data-driven MPC
            parameters that remain fixed across the grid search.

    Returns:
        NonlinearDataDrivenMPCController: The initialized nonlinear data-driven
            MPC controller instance.
    """
    # Retrieve combination parameters
    n = combination_params.n
    L = combination_params.L
    lamb_alpha = combination_params.lamb_alpha
    lamb_sigma = combination_params.lamb_sigma
    lamb_alpha_s = combination_params.lamb_alpha_s
    lamb_sigma_s = combination_params.lamb_sigma_s

    # Retrieve fixed parameters
    m = fixed_params.m
    p = fixed_params.p
    Q_weights = fixed_params.Q_weights
    R_weights = fixed_params.R_weights
    S_weights = fixed_params.S_weights
    U = fixed_params.U
    Us = fixed_params.Us
    alpha_reg_type = fixed_params.alpha_reg_type
    ext_out_incr_in = fixed_params.ext_out_incr_in
    n_mpc_step = n if fixed_params.n_n_mpc_step else 1

    # Define dependent Nonlinear Data-Driven MPC controller parameters
    # Output and Input weighting matrices based on controller structure
    if ext_out_incr_in:
        # Output weighting matrix Q
        extended_weights = Q_weights + R_weights
        Q = np.kron(np.eye(L + n + 1), np.diag(extended_weights))

        # Input weighting matrix R
        R = 1 * np.eye(m * (L + n + 1))
    else:
        # Output weighting matrix Q
        Q = np.kron(np.eye(L + n + 1), np.diag(Q_weights))

        # Input weighting matrix R
        R = np.kron(np.eye(L + n + 1), np.diag(R_weights))

    # Output setpoint weighting matrix S
    S = np.kron(np.eye(1), np.diag(S_weights))

    # Create Nonlinear Data-Driven MPC controller
    nolinear_dd_mpc_controller = NonlinearDataDrivenMPCController(
        n=n,
        m=m,
        p=p,
        u=u_N,
        y=y_N,
        L=L,
        Q=Q,
        R=R,
        S=S,
        y_r=y_r,
        lamb_alpha=lamb_alpha,
        lamb_sigma=lamb_sigma,
        U=U,
        Us=Us,
        lamb_alpha_s=lamb_alpha_s,
        lamb_sigma_s=lamb_sigma_s,
        alpha_reg_type=alpha_reg_type,
        ext_out_incr_in=ext_out_incr_in,
        update_cost_threshold=None,
        n_mpc_step=n_mpc_step,
    )

    return nolinear_dd_mpc_controller
