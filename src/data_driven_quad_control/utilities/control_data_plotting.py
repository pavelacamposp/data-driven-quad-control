from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from direct_data_driven_mpc.utilities.visualization.comparison_plot import (
    plot_input_output_comparison,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .config_utils import load_yaml_config


@dataclass
class ControlTrajectory:
    control_inputs: list[np.ndarray]
    system_outputs: list[np.ndarray]
    system_setpoint: np.ndarray


def plot_trajectory_comparison(
    trajectory_data: ControlTrajectory,
    u_bounds_list: list[tuple[float, float]],
    inputs_line_param_list: list[dict[str, Any]],
    outputs_line_param_list: list[dict[str, Any]],
    setpoints_line_params: dict[str, Any],
    bounds_line_params: dict[str, Any],
    legend_params: dict[str, Any],
    controller_labels: list[str],
    y_setpoint_labels: list[str],
    x_axis_labels: list[str],
    input_y_axis_labels: list[str],
    output_y_axis_labels: list[str],
    figsize: tuple[int, int] = (14, 8),
    dpi: int = 300,
    fontsize: int = 12,
    title: str | None = "Control Comparison",
    ctrl_dt: float | None = None,
) -> Figure:
    """
    Plot multiple input-output trajectories with setpoints in a Matplotlib
    figure for visual comparison of different controllers.

    This function creates a figure with two rows of subplots: the first row
    for control inputs, and the second for system outputs. Each subplot shows
    the trajectories of each data series alongside its setpoint line.

    Args:
        trajectory_data (ControlTrajectory): An object containing control
            input, output, and setpoint trajectories to be plotted.
        u_bounds_list (list[tuple[float, float]]): A list of tuples
            (lower_bound, upper_bound) specifying bounds for each input data
            sequence.
        inputs_line_param_list (list[dict[str, Any]]): A list of dictionaries,
            one per input trajectory, specifying Matplotlib properties for each
            controller's input lines.
        outputs_line_param_list (list[dict[str, Any]]): A list of dictionaries,
            one per output trajectory, specifying Matplotlib properties for
            each controller's output lines.
        setpoints_line_params (dict[str, Any]): A dictionary of Matplotlib
            properties for customizing setpoint lines.
        bounds_line_params (dict[str, Any]): A dictionary of Matplotlib
            properties for customizing input bound lines.
        legend_params (dict[str, Any]): A dictionary of Matplotlib
            properties for customizing the plot legends.
        controller_labels (list[str]): A list of strings specifying custom
            legend labels for each controller. Used for both input and output
            subplots.
        y_setpoint_labels (list[str]): A list of strings specifying custom
            legend labels for output setpoints.
        x_axis_labels (list[str]): A list of strings specifying custom X-axis
            labels for each subplot.
        input_y_axis_labels (list[str]): A list of strings specifying custom
            Y-axis labels for each input subplot.
        output_y_axis_labels (list[str]): A list of strings specifying custom
            Y-axis labels for each output subplot.
        figsize (tuple[int, int]): The (width, height) dimensions of the
            created Matplotlib figure.
        dpi (int): The DPI resolution of the figure.
        fontsize (int): The fontsize for labels, legends and axes ticks.
        title (str | None): The title for the created plot figure.
        ctrl_dt (float | None): The control time step used for each controller
            data (in seconds). If provided, the X-axis ticks will be relabeled
            from discrete time steps to time in seconds.

    Returns:
        Figure: The created Matplotlib figure containing the trajectory
            comparison plot.
    """
    plot_input_output_comparison(
        u_data=trajectory_data.control_inputs,
        y_data=trajectory_data.system_outputs,
        y_s=trajectory_data.system_setpoint,
        u_bounds_list=u_bounds_list,
        inputs_line_param_list=inputs_line_param_list,
        outputs_line_param_list=outputs_line_param_list,
        setpoints_line_params=setpoints_line_params,
        bounds_line_params=bounds_line_params,
        legend_params=legend_params,
        figsize=figsize,
        dpi=dpi,
        fontsize=fontsize,
        title=title,
        input_labels=controller_labels,
        output_labels=controller_labels,
        y_setpoint_labels=y_setpoint_labels,
        x_axis_labels=x_axis_labels,
        input_y_axis_labels=input_y_axis_labels,
        output_y_axis_labels=output_y_axis_labels,
        show=False,  # Disable showing plot for external modifications
    )

    # Get figure and axes for external plot modification
    fig = plt.gcf()
    axs = fig.axes

    # Relabel the X-axis from time steps to seconds
    # if the controller time step is provided
    if ctrl_dt:
        T = trajectory_data.control_inputs[0].shape[0]
        relabel_time_axis_ticks(
            axs, num_steps=T, ctrl_dt=ctrl_dt, label_step=1
        )

    return fig


def relabel_time_axis_ticks(
    axs: list[Axes], num_steps: int, ctrl_dt: float, label_step: int = 1
) -> None:
    """
    Relabel X-axis ticks from time steps to seconds.

    Args:
        axs (list[Axes]): A list of Matplotlib axes to update.
        num_steps (int): The number of time steps of the data plotted in each
            `axs` axis.
        ctrl_dt (float): The control time step in seconds.
        label_step (int): The number of time steps used to create labels. Each
            label will be created every `label_step` seconds. Defaults to 1.
    """
    total_time = num_steps * ctrl_dt

    tick_locs = np.arange(0, total_time + ctrl_dt, step=1.0)
    tick_indices = (tick_locs / ctrl_dt).astype(int)
    tick_labels = [
        str(int(t)) if t % label_step == 0 else "" for t in tick_locs
    ]

    for ax in axs:
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels)


def load_plot_params(config_path: str) -> dict[str, Any]:
    """
    Load Matplotlib plot parameters from a YAML config file.

    The YAML configuration file must have the following structure:
        line_params:
          input:
            color: "blue"
            ...
          output:
            color: "green"
            ...
          bounds:
            color: "orange"
            ...
          setpoint:
            color: "red"
            ...

        legend_params:
          fontsize: 11
          ...

        figure_params:
          figsize: [9, 12]
          ...

    Args:
        config_path (str): The path to the YAML configuration file containing
            Matplotlib plot parameters.

    Returns:
        dict[str, Any]: A dictionary of the loaded Matplotlib properties, with
            keys "inputs_line_params", "outputs_line_params",
            "setpoints_line_params", "bounds_line_params", "legend_params", and
            figure-related keys like "figsize" and "dpi".
    """
    # Load plot parameters from config file
    plot_params = load_yaml_config(config_path)

    line_params = plot_params["line_params"]
    legend_params = plot_params["legend_params"]
    figure_params = plot_params["figure_params"]

    return {
        "inputs_line_params": line_params["input"],
        "outputs_line_params": line_params["output"],
        "setpoints_line_params": line_params["setpoint"],
        "bounds_line_params": line_params["bounds"],
        "legend_params": legend_params,
        "figure_params": figure_params,
    }


def load_comparison_plot_params(
    base_params_config_path: str,
    comparison_params_config_path: str,
) -> dict[str, Any]:
    """
    Load and merge Matplotlib plot parameters for comparison plots from two
    YAML configuration files: a base config file (see `load_plot_params` for
    its structure) and a comparison-specific config file.

    The comparison plot YAML configuration file must have the following
    structure:
        line_params:
          tracking:
            color: "red"
            ...
          rl:
            color: "green"
            ...
          dd_mpc:
            color: "blue"
            ...
          setpoints:
            color: "black"
            ...
          bounds:
            color: "orange"
            ...

        controller_labels:
          tracking: "Tracking"
          rl: "RL - CTBR Fixed Yaw"
          dd_mpc: "DD-MPC"

        y_setpoint_labels:
          - "Setpoint Label A"
          - "Setpoint Label B"
          - "Setpoint Label C"

        x_axis_labels:
          - "X-axis Label A"
          - "X-axis Label B"
          - "X-axis Label C"

        input_y_axis_labels:
          - "Input Y-axis Label A"
          - "Input Y-axis Label B"
          - "Input Y-axis Label C"

        output_y_axis_labels:
          - "Output Y-axis Label A"
          - "Output Y-axis Label B"
          - "Output Y-axis Label C"

        figure_params:
          figsize: [9, 12]
          ...

    Args:
        base_params_config_path (str): The path to the YAML configuration file
            containing Matplotlib plot parameters used as a base. Parameters
            from `comparison_params_config_path` will override these
            parameters.
        comparison_params_config_path (str): The path to the YAML configuration
            file containing Matplotlib parameters for comparison plots.

    Returns:
        dict[str, Any]: A dictionary containing merged Matplotlib plot
            parameters used by comparison plots.
    """
    # Load base plot params
    base_plot_params = load_plot_params(base_params_config_path)

    # Load comparison-specific plot params
    comparison_plot_params = load_yaml_config(comparison_params_config_path)
    controller_line_params = comparison_plot_params["line_params"]
    controller_labels = list(
        comparison_plot_params["controller_labels"].values()
    )
    y_setpoint_labels = comparison_plot_params["y_setpoint_labels"]
    x_axis_labels = comparison_plot_params["x_axis_labels"]
    input_y_axis_labels = comparison_plot_params["input_y_axis_labels"]
    output_y_axis_labels = comparison_plot_params["output_y_axis_labels"]

    data_line_param_list = [
        controller_line_params["tracking"],
        controller_line_params["rl"],
        controller_line_params["dd_mpc"],
    ]
    setpoints_line_params = controller_line_params["setpoints"]
    bounds_line_params = controller_line_params["bounds"]

    # Construct figure params
    figure_params = {
        **base_plot_params["figure_params"],
        **comparison_plot_params["figure_params"],
    }

    return {
        "inputs_line_param_list": data_line_param_list,
        "outputs_line_param_list": data_line_param_list,
        "setpoints_line_params": setpoints_line_params,
        "bounds_line_params": bounds_line_params,
        "legend_params": base_plot_params["legend_params"],
        "controller_labels": controller_labels,
        "y_setpoint_labels": y_setpoint_labels,
        "x_axis_labels": x_axis_labels,
        "input_y_axis_labels": input_y_axis_labels,
        "output_y_axis_labels": output_y_axis_labels,
        **figure_params,
    }
