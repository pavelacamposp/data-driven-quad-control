from dataclasses import fields

import numpy as np
import torch

from data_driven_quad_control.controllers.tracking.tracking_controller import (
    DroneTrackingController,
)
from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.initial_data_collection_cache import (  # noqa: E501
    cache_initial_data_and_states,
)
from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.param_grid_search_config import (  # noqa: E501  # noqa: E501
    DataDrivenCache,
    DDMPCFixedParams,
    DDMPCInitialDataCollectionParams,
    DDMPCParameterGrid,
)
from data_driven_quad_control.envs.config.hover_env_config import EnvState
from data_driven_quad_control.envs.hover_env import HoverEnv
from data_driven_quad_control.utilities.vectorized_pid_controller import (
    VectorizedControllerState,
)


def test_cache_initial_data_and_states(
    mock_env: HoverEnv,
    mock_tracking_controller: DroneTrackingController,
    test_drone_state: EnvState,
    test_init_collection_params: DDMPCInitialDataCollectionParams,
    test_param_grid: DDMPCParameterGrid,
    test_fixed_params: DDMPCFixedParams,
) -> None:
    # Define test parameters
    target_pos = torch.tensor([[0.0, 0.0, 1.0]])
    target_yaw = torch.tensor([0.0])

    # Test function with mocked env and tracking controller
    data_driven_cache, drone_state_cache = cache_initial_data_and_states(
        env=mock_env,
        base_env_idx=0,
        stabilizing_controller=mock_tracking_controller,
        target_pos=target_pos,
        target_yaw=target_yaw,
        init_hovering_state=test_drone_state,
        init_data_collection_params=test_init_collection_params,
        param_grid=test_param_grid,
        fixed_params=test_fixed_params,
        np_random=np.random.default_rng(0),
    )

    # Verify that the data-driven cache was correctly defined
    assert isinstance(data_driven_cache, DataDrivenCache)

    # Verify that the cache collection iterated
    # through all the `N` values from the grid
    N_values = set(test_param_grid.N)
    assert set(data_driven_cache.u_N.keys()) == N_values
    assert set(data_driven_cache.y_N.keys()) == N_values
    assert set(data_driven_cache.drone_state.keys()) == N_values
    assert set(drone_state_cache.keys()) == N_values

    # Verify that the environment states stored in `data_driven_cache`
    # match to the ones stored in `drone_state_cache`
    for N in N_values:
        assert_env_states_equal(
            data_driven_cache.drone_state[N],
            drone_state_cache[N],
        )


def assert_env_states_equal(state_1: EnvState, state_2: EnvState) -> None:
    # Assert that all values in two `EnvState`
    # objects are equal regardless of device
    for field in fields(EnvState):
        name = field.name
        val_1 = getattr(state_1, name)
        val_2 = getattr(state_2, name)

        if val_1 is None and val_2 is None:
            continue

        elif (val_1 is None) != (val_2 is None):
            raise AssertionError(
                f"Mismatch in field '{name}': one is `None`, while the other "
                "is not"
            )

        elif isinstance(val_1, torch.Tensor) and isinstance(
            val_2, torch.Tensor
        ):
            assert torch.allclose(val_1.cpu(), val_2.cpu()), (
                f"Mismatch in tensor '{name}'"
            )

        elif isinstance(val_1, VectorizedControllerState) and isinstance(
            val_2, VectorizedControllerState
        ):
            for sub_field in fields(val_1.__class__):
                sub_name = sub_field.name
                sub_val_1 = getattr(val_1, sub_name)
                sub_val_2 = getattr(val_2, sub_name)

                assert torch.allclose(sub_val_1.cpu(), sub_val_2.cpu()), (
                    f"Mismatch in controller state '{name}.{sub_name}'"
                )

        else:
            raise TypeError(f"Unexpected type for field '{name}'")
