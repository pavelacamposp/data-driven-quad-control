import torch

from data_driven_quad_control.comparison.utilities.comparison_config_loader import (  # noqa: E501
    load_controller_comparison_params,
)
from data_driven_quad_control.comparison.utilities.controller_comparison_config import (  # noqa: E501
    ControllerComparisonParams,
)


def test_load_controller_comparison_params(
    test_comparison_params_path: str,
) -> None:
    # Load comparison parameters from configuration file
    device = torch.device("cpu")
    controller_comparison_params = load_controller_comparison_params(
        config_path=test_comparison_params_path,
        env_device=device,
    )

    # Verify the type of the returned object
    assert isinstance(controller_comparison_params, ControllerComparisonParams)

    # Verify loaded parameters based on known values from the test config file
    expected_hover_pos = torch.tensor(
        [0.0, 0.0, 1.5], dtype=torch.float, device=device
    )
    torch.testing.assert_close(
        controller_comparison_params.init_hover_pos, expected_hover_pos
    )

    expected_setpoint = torch.tensor(
        [[0.0, 0.0, 1.5]], dtype=torch.float, device=device
    )
    assert isinstance(controller_comparison_params.eval_setpoints, list)
    assert len(controller_comparison_params.eval_setpoints) == 1
    torch.testing.assert_close(
        controller_comparison_params.eval_setpoints[0], expected_setpoint
    )

    assert (
        "ctbr_fixed_yaw_model_1000.pt"
        in controller_comparison_params.ppo_model_path
    )
