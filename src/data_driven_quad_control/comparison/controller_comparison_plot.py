"""
Trajectory Plotter for Data-driven Controller Comparison

This script loads control trajectory data collected from a controller
comparison simulation (via `controller_comparison.py`) and generates an
input-output comparison plot.

The generated plots can be configured using the YAML configuration files in
`configs/plots/`:
  - `base_plot_params.yaml`: General plot parameters.
  - `comparison_plot_params.yaml`: Controller comparison-specific plot
        parameters.
"""

import os
import pickle

import matplotlib.pyplot as plt

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
    # Load controller comparison data (control trajectories)
    control_data_file = os.path.join(
        COMPARISON_LOGS_DIR, "control_trajectory.pkl"
    )
    with open(control_data_file, "rb") as f:
        control_trajectory_data: ControlTrajectory = pickle.load(f)

    # Load plot params from config file
    plot_params = load_comparison_plot_params(
        base_params_config_path=BASE_PLOT_PARAMS_CONFIG_PATH,
        comparison_params_config_path=COMPARISON_PLOT_PARAMS_CONFIG_PATH,
    )

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
    )

    plt.show()


if __name__ == "__main__":
    main()
