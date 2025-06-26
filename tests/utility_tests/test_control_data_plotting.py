from typing import Any
from unittest.mock import Mock, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from data_driven_quad_control.utilities.control_data_plotting import (
    ControlTrajectory,
    load_comparison_plot_params,
    load_plot_params,
    plot_trajectory_comparison,
    relabel_time_axis_ticks,
)

matplotlib.use("Agg")  # Prevent GUI backend


CONTROL_DATA_PLOTTING_PATH = (
    "data_driven_quad_control.utilities.control_data_plotting"
)

PLOT_IO_COMPARISON_PATCH_PATH = (
    CONTROL_DATA_PLOTTING_PATH + ".plot_input_output_comparison"
)
LOAD_YAML_PATCH_PATH = CONTROL_DATA_PLOTTING_PATH + ".load_yaml_config"
LOAD_PLOT_PARAMS_PATCH_PATH = CONTROL_DATA_PLOTTING_PATH + ".load_plot_params"


@patch(PLOT_IO_COMPARISON_PATCH_PATH)
def test_plot_trajectory_comparison(mock_plot_io_comparison: Mock) -> None:
    # Define test parameters
    num_steps = 5
    ctrl_dt = 0.5
    control_inputs = [np.ones((num_steps,)) * i for i in range(3)]
    system_outputs = [np.ones((num_steps,)) * (i + 1) for i in range(3)]
    setpoint = np.ones((num_steps,))

    trajectory = ControlTrajectory(
        control_inputs=control_inputs,
        system_outputs=system_outputs,
        system_setpoint=setpoint,
    )

    u_bounds_list = [(0.0, 1.0)] * 3
    line_params = [{"color": "blue"}] * 3
    setpoints_line_params = {"color": "red"}
    bounds_line_params = {"color": "orange"}
    legend_params = {"fontsize": 10}
    controller_labels = ["A", "B", "C"]
    y_setpoint_labels = ["Y1", "Y2", "Y3"]
    x_axis_labels = ["X1", "X2", "X3"]
    input_y_axis_labels = ["u1", "u2", "u3"]
    output_y_axis_labels = ["y1", "y2", "y3"]

    fig = plot_trajectory_comparison(
        trajectory_data=trajectory,
        u_bounds_list=u_bounds_list,
        inputs_line_param_list=line_params,
        outputs_line_param_list=line_params,
        setpoints_line_params=setpoints_line_params,
        bounds_line_params=bounds_line_params,
        legend_params=legend_params,
        controller_labels=controller_labels,
        y_setpoint_labels=y_setpoint_labels,
        x_axis_labels=x_axis_labels,
        input_y_axis_labels=input_y_axis_labels,
        output_y_axis_labels=output_y_axis_labels,
        ctrl_dt=ctrl_dt,
    )

    assert isinstance(fig, Figure)
    mock_plot_io_comparison.assert_called_once()

    plt.close(fig)


def test_relabel_time_axis_ticks() -> None:
    # Define test parameters
    fig, ax = plt.subplots()
    axs = [ax]
    num_steps = 4
    ctrl_dt = 0.5
    label_step = 2
    expected_ticks = [0, 2, 4]
    expected_labels = ["0", "", "2"]

    relabel_time_axis_ticks(
        axs, num_steps=num_steps, ctrl_dt=ctrl_dt, label_step=label_step
    )

    # Verify that the ticks and labels match the expected values
    ticks = ax.get_xticks().tolist()
    labels = [label.get_text() for label in ax.get_xticklabels()]

    assert ticks == expected_ticks
    assert labels == expected_labels

    plt.close(fig)


@patch(LOAD_YAML_PATCH_PATH)
def test_load_plot_params(mock_load_yaml: Mock) -> None:
    # Mock return value of `load_yaml_config`
    test_yaml: dict[str, Any] = {
        "line_params": {
            "input": {"color": "blue"},
            "output": {"color": "green"},
            "bounds": {"color": "orange"},
            "setpoint": {"color": "red"},
        },
        "legend_params": {"fontsize": 11},
        "figure_params": {"figsize": [9, 12], "dpi": 100},
    }
    mock_load_yaml.return_value = test_yaml

    plot_params: dict[str, Any] = load_plot_params("dummy_path.yaml")

    # Verify correct parameter loading
    assert "inputs_line_params" in plot_params
    assert (
        plot_params["inputs_line_params"] == test_yaml["line_params"]["input"]
    )


@patch(LOAD_YAML_PATCH_PATH)
@patch(LOAD_PLOT_PARAMS_PATCH_PATH)
def test_load_comparison_plot_params(
    mock_load_plot_params: Mock,
    mock_comparison_load_yaml: Mock,
) -> None:
    # Define test parameters
    expected_inputs_line_param_list = [
        {"color": "red"},
        {"color": "green"},
        {"color": "blue"},
    ]
    expected_controller_labels = ["Tracking", "RL", "DD-MPC"]

    # Mock return values of `load_plot_params` (based plot params)
    mock_load_plot_params.return_value = {
        "inputs_line_params": {"color": "blue"},
        "outputs_line_params": {"color": "green"},
        "setpoints_line_params": {"color": "orange"},
        "bounds_line_params": {"color": "red"},
        "legend_params": {"fontsize": 9},
        "figure_params": {"figsize": [8, 6], "dpi": 100},
    }

    # Mock return values of `load_yaml_config` (comparison plot params)
    mock_comparison_load_yaml.return_value = {
        "line_params": {
            "tracking": expected_inputs_line_param_list[0],
            "rl": expected_inputs_line_param_list[1],
            "dd_mpc": expected_inputs_line_param_list[2],
            "setpoints": {"color": "black"},
            "bounds": {"color": "orange"},
        },
        "controller_labels": {
            "tracking": expected_controller_labels[0],
            "rl": expected_controller_labels[1],
            "dd_mpc": expected_controller_labels[2],
        },
        "y_setpoint_labels": ["A", "B", "C"],
        "x_axis_labels": ["X1", "X2", "X3"],
        "input_y_axis_labels": ["U1", "U2", "U3"],
        "output_y_axis_labels": ["Y1", "Y2", "Y3"],
        "figure_params": {"dpi": 200},
    }

    comparison_plot_params = load_comparison_plot_params(
        "dummy_base_path.yaml", "dummy_comparison_path.yaml"
    )

    # Verify correct parameter loading
    assert (
        comparison_plot_params["inputs_line_param_list"]
        == expected_inputs_line_param_list
    )
    assert (
        comparison_plot_params["controller_labels"]
        == expected_controller_labels
    )

    # Verify that base figure params are loaded as expected
    # Overridden, as dpi is included in comparison params
    assert comparison_plot_params["dpi"] == 200

    # Retained, as figsize is not included in comparison params
    assert comparison_plot_params["figsize"] == [8, 6]
