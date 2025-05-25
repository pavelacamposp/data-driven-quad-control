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
    DDMPCEvaluationParams,
    DDMPCFixedParams,
    DDMPCInitialDataCollectionParams,
    DDMPCParameterGrid,
)
from data_driven_quad_control.envs.hover_env import HoverEnv
from data_driven_quad_control.envs.hover_env_config import EnvState
from data_driven_quad_control.utilities.vectorized_pid_controller import (
    VectorizedControllerState,
)


def test_cache_initial_data_and_states(
    mock_env: HoverEnv,
    mock_tracking_controller: DroneTrackingController,
    test_drone_state: EnvState,
    test_init_collection_params: DDMPCInitialDataCollectionParams,
    test_fixed_params: DDMPCFixedParams,
    test_eval_params: DDMPCEvaluationParams,
    test_param_grid: DDMPCParameterGrid,
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
        fixed_params=test_fixed_params,
        eval_params=test_eval_params,
        param_grid=test_param_grid,
        np_random=np.random.default_rng(0),
    )

    # Verify that the data-driven cache was correctly defined
    assert isinstance(data_driven_cache, DataDrivenCache)

    # Verify that the cache collection iterated
    # through all the `N` values from the grid
    data_entry_indices = {
        entry_idx
        for entry_indices in data_driven_cache.N_to_entry_indices.values()
        for entry_idx in entry_indices
    }

    assert set(data_driven_cache.N.keys()) == data_entry_indices
    assert set(data_driven_cache.u_N.keys()) == data_entry_indices
    assert set(data_driven_cache.y_N.keys()) == data_entry_indices
    assert set(data_driven_cache.drone_state.keys()) == data_entry_indices
    assert set(drone_state_cache.keys()) == data_entry_indices

    # Verify that the collected `N` values match the original param grid
    for expected_N in test_param_grid.N:
        entry_indices = data_driven_cache.N_to_entry_indices[expected_N]
        for entry_idx in entry_indices:
            assert data_driven_cache.N[entry_idx] == expected_N

    # Verify that the environment states stored in `data_driven_cache` (CPU)
    # match the ones stored in `drone_state_cache` (GPU-set to CPU for testing)
    for i in data_entry_indices:
        assert_env_states_equal(
            data_driven_cache.drone_state[i],
            drone_state_cache[i],
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
