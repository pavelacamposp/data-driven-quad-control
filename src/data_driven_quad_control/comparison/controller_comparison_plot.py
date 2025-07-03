"""
Trajectory Plotter for Data-driven Controller Comparison

This script loads one or more control trajectory datasets collected from
controller comparison simulations (via `controller_comparison.py`) and
generates an input-output comparison plot.

If multiple datasets are found in the `logs/comparison` directory and share
the same target setpoint data, the script computes and plots the mean and
standard deviation of the control inputs and system outputs across runs for
each controller. This allows visual comparison of both control performance and
trajectory variability across controllers.

To generate multiple datasets with consistent target setpoint data, set the
`steps_per_setpoint` parameter in the controller comparison configuration file
(`configs/comparison/controller_comparison_config.yaml`) to a constant value.

The generated plots can be configured using the YAML configuration files in
`configs/plots/`:
  - `base_plot_params.yaml`: General plot parameters.
  - `comparison_plot_params.yaml`: Controller comparison-specific plot
        parameters.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from data_driven_quad_control.envs.hover_env_config import (
    EnvActionBounds,
    get_cfgs,
)
from data_driven_quad_control.utilities.control_data_plotting import (
    ControlTrajectory,
    load_comparison_plot_params,
    plot_trajectory_comparison,
)

# Define plot params config file paths
BASE_PLOT_PARAMS_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../configs/plots/base_plot_params.yaml",
)

COMPARISON_PLOT_PARAMS_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../configs/plots/comparison_plot_params.yaml",
)

# Directory where controller comparison data files are saved
COMPARISON_LOGS_DIR = "logs/comparison"


def main() -> None:
    # Load plot params from config file
    plot_params = load_comparison_plot_params(
        base_params_config_path=BASE_PLOT_PARAMS_CONFIG_PATH,
        comparison_params_config_path=COMPARISON_PLOT_PARAMS_CONFIG_PATH,
    )

    # Load controller comparison data (control trajectories)
    data_files = os.listdir(COMPARISON_LOGS_DIR)
    data_files = [file for file in data_files if "control_trajectory" in file]
    data_files.sort()  # Sort files by timestamp

    # Ectract control data from each trajectory file
    control_inputs_list = []
    system_outputs_list = []
    plot_setpoint = None
    plot_multiple_data = len(data_files) > 1

    for file in data_files:
        control_data_file = os.path.join(COMPARISON_LOGS_DIR, file)
        with open(control_data_file, "rb") as f:
            control_trajectory_data: ControlTrajectory = pickle.load(f)

        control_inputs_list.append(
            np.array(control_trajectory_data.control_inputs)
        )
        system_outputs_list.append(
            np.array(control_trajectory_data.system_outputs)
        )

        # Disable plotting mean/std elements if
        # any trajectory has different setpoints
        file_setpoint = control_trajectory_data.system_setpoint
        if plot_setpoint is not None:
            if not np.array_equal(file_setpoint, plot_setpoint):
                plot_multiple_data = False

        plot_setpoint = file_setpoint

    # Compute mean and std if multiple runs will be plotted
    if plot_multiple_data:
        control_inputs_array = np.stack(control_inputs_list, axis=0)
        system_outputs_array = np.stack(system_outputs_list, axis=0)

        mean_inputs = np.mean(control_inputs_array, axis=0)
        std_inputs = np.std(control_inputs_array, axis=0)

        mean_outputs = np.mean(system_outputs_array, axis=0)
        std_outputs = np.std(system_outputs_array, axis=0)

        # Update control trajectory data with mean arrays for plotting
        control_trajectory_data.control_inputs = [
            mean_inputs[i] for i in range(mean_inputs.shape[0])
        ]
        control_trajectory_data.system_outputs = [
            mean_outputs[i] for i in range(mean_outputs.shape[0])
        ]

    # Construct input bounds from env config
    u_bounds_list = [
        (0.0, EnvActionBounds.MAX_THRUST),
        (-EnvActionBounds.MAX_ANG_VELS[0], EnvActionBounds.MAX_ANG_VELS[0]),
        (-EnvActionBounds.MAX_ANG_VELS[1], EnvActionBounds.MAX_ANG_VELS[1]),
    ]

    # Calculate controller time step from environment configuration
    env_cfg, _, _, _ = get_cfgs()
    ctrl_dt = env_cfg["dt"] * env_cfg["decimation"]

    plot_trajectory_comparison(
        trajectory_data=control_trajectory_data,
        u_bounds_list=u_bounds_list,
        **plot_params,
        ctrl_dt=ctrl_dt,
        x_axis_tick_step=2,
    )

    # Plot mean +/- std areas if multiple runs are plotted
    if plot_multiple_data:
        # Get mean +/- std region colors from config
        colors = [
            line_params["color"]
            for line_params in plot_params["inputs_line_param_list"]
        ]

        # Get figure and axes
        fig = plt.gcf()
        axs = fig.axes

        # Plot mean +/- std areas
        num_sim = mean_inputs.shape[0]
        T, m = mean_inputs[0].shape

        for sim_idx in range(num_sim):
            mean_input_data = mean_inputs[sim_idx]
            std_input_data = std_inputs[sim_idx]
            mean_output_data = mean_outputs[sim_idx]
            std_output_data = std_outputs[sim_idx]

            # Plot mean +/- std area for inputs
            for i, ax in enumerate(axs[:m]):
                input_min, input_max = u_bounds_list[i]
                lower = mean_input_data[:, i] - std_input_data[:, i]
                upper = mean_input_data[:, i] + std_input_data[:, i]

                # Clip area to input bounds to ensure it stays within limits
                # and prevent confusion (e.g., it might appear as if control
                # inputs are unbounded)
                ax.fill_between(
                    range(T),
                    np.clip(lower, input_min, input_max),
                    np.clip(upper, input_min, input_max),
                    alpha=0.3,
                    color=colors[sim_idx],
                )

            # Plot mean +/- std area for outputs
            for j, ax in enumerate(axs[m:]):
                ax.fill_between(
                    range(T),
                    mean_output_data[:, j] - std_output_data[:, j],
                    mean_output_data[:, j] + std_output_data[:, j],
                    alpha=0.3,
                    color=colors[sim_idx],
                )

    plt.show()


if __name__ == "__main__":
    main()
