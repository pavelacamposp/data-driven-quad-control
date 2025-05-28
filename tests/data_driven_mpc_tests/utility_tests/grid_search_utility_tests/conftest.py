import numpy as np
import pytest
import torch
from direct_data_driven_mpc.nonlinear_data_driven_mpc_controller import (
    AlphaRegType,
)

from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.param_grid_search_config import (  # noqa: E501
    DataDrivenCache,
    DDMPCCombinationParams,
    DDMPCEvaluationParams,
    DDMPCFixedParams,
    DDMPCInitialDataCollectionParams,
    DDMPCParameterGrid,
)
from data_driven_quad_control.envs.hover_env_config import EnvState
from data_driven_quad_control.utilities.vectorized_pid_controller import (
    VectorizedControllerState,
)


@pytest.fixture
def test_init_collection_params() -> DDMPCInitialDataCollectionParams:
    return DDMPCInitialDataCollectionParams(
        init_hover_pos=np.array([[0.0, 0.0, 1.5]]),
        u_range=np.zeros((3, 2)),
    )


@pytest.fixture
def test_fixed_params() -> DDMPCFixedParams:
    return DDMPCFixedParams(
        m=3,
        p=3,
        Q_weights=[1.0, 1.0, 1.0],
        R_weights=[1.0, 1.0, 1.0],
        S_weights=[1.0, 1.0, 1.0],
        U=np.zeros((3, 2)),
        Us=np.zeros((3, 2)),
        alpha_reg_type=AlphaRegType.APPROXIMATED,
        ext_out_incr_in=False,
        n_n_mpc_step=False,
    )


@pytest.fixture
def test_combination_params() -> DDMPCCombinationParams:
    return DDMPCCombinationParams(
        N=40,
        n=2,
        L=5,
        lamb_alpha=10,
        lamb_sigma=1000,
        lamb_alpha_s=10,
        lamb_sigma_s=1000,
    )


@pytest.fixture
def test_param_grid() -> DDMPCParameterGrid:
    return DDMPCParameterGrid(
        N=[40, 45],
        n=[2],
        L=[5],
        lamb_alpha=[10.0],
        lamb_sigma=[1000.0],
        lamb_alpha_s=[10.0],
        lamb_sigma_s=[1000.0],
    )


@pytest.fixture
def test_eval_params() -> DDMPCEvaluationParams:
    return DDMPCEvaluationParams(
        eval_time_steps=1,
        eval_setpoints=[np.array([0.0, 0.0, 1.5])],
        max_target_dist_increment=0.5,
        num_collections_per_N=2,
    )


@pytest.fixture
def test_drone_state() -> EnvState:
    return EnvState(
        base_pos=torch.zeros((1, 3)),
        base_quat=torch.zeros((1, 4)),
        base_lin_vel=torch.zeros((1, 3)),
        base_ang_vel=torch.zeros((1, 3)),
        commands=torch.zeros((1, 3)),
        episode_length=torch.tensor([1.0]),
        last_actions=torch.zeros((1, 3)),
        ctbr_controller_state=VectorizedControllerState(
            integral=torch.zeros((1, 3)),
            prev_error=torch.zeros((1, 3)),
        ),
    )


@pytest.fixture
def test_data_driven_cache(test_drone_state: EnvState) -> DataDrivenCache:
    return DataDrivenCache(
        N_to_entry_indices={40: [0, 1]},
        N={0: 40, 1: 40},
        u_N={0: np.zeros((40, 3)), 1: np.zeros((40, 3))},
        y_N={0: np.zeros((40, 3)), 1: np.zeros((40, 3))},
        drone_state={0: test_drone_state, 1: test_drone_state},
    )


@pytest.fixture
def test_drone_state_cache(test_drone_state: EnvState) -> dict[int, EnvState]:
    return {
        0: test_drone_state,
        1: test_drone_state,
    }
