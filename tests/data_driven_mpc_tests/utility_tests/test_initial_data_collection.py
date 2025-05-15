import numpy as np
import pytest
import torch
from direct_data_driven_mpc.utilities.models.nonlinear_model import (
    NonlinearSystem,
)

from data_driven_quad_control.controllers.tracking.tracking_controller import (
    DroneTrackingController,
)
from data_driven_quad_control.data_driven_mpc.utilities.drone_initial_data_collection import (  # noqa: E501
    collect_initial_input_output_data,
    get_init_hover_pos,
)
from data_driven_quad_control.envs.hover_env import HoverEnv


@pytest.mark.parametrize("use_system_model", [True, False])
def test_collect_initial_input_output_data(
    use_system_model: bool,
    mock_env: HoverEnv,
    mock_system_model: NonlinearSystem,
    mock_tracking_controller: DroneTrackingController,
) -> None:
    # Define test parameters
    N = 100
    m = 3
    p = 3
    input_bounds = np.array(
        [
            [0.0, 1.0],
            [-0.5, 0.5],
            [-0.3, 0.3],
        ]
    )
    u_range = np.array(
        [
            [0.1, 0.2],
            [0.05, 0.1],
            [0.02, 0.05],
        ]
    )
    target_pos = torch.tensor([[0.0, 0.0, 1.0]])
    target_yaw = torch.tensor([0.0])

    # Test function with either a mocked system model or direct env stepping
    drone_system_model = mock_system_model if use_system_model else None
    u_N, y_N = collect_initial_input_output_data(
        env=mock_env,
        base_env_idx=0,
        stabilizing_controller=mock_tracking_controller,
        target_pos=target_pos,
        target_yaw=target_yaw,
        input_bounds=input_bounds,
        u_range=u_range,
        N=N,
        m=m,
        p=p,
        eps_max=0.1,
        np_random=np.random.default_rng(0),
        drone_system_model=drone_system_model,
    )

    # Verify collected input-output array shapes
    assert u_N.shape == (N, m)
    assert y_N.shape == (N, p)

    # Assert that all input and output values are finite
    assert np.all(np.isfinite(u_N)), "Control input contains NaNs or Infs"
    assert np.all(np.isfinite(y_N)), "System output contains NaNs or Infs"

    # Assert that all control inputs are within the specified bounds
    for i in range(m):
        assert np.all(u_N[:, i] >= input_bounds[i, 0])
        assert np.all(u_N[:, i] <= input_bounds[i, 1])


def test_get_init_hover_pos(
    test_controller_params_path: str,
    mock_env: HoverEnv,
) -> None:
    target_pos = get_init_hover_pos(
        config_path=test_controller_params_path,
        controller_key_value="controller_key",
        env=mock_env,
    )

    # Verify target position tensor device and shape
    assert target_pos.device == mock_env.device
    assert target_pos.shape == (3,)
