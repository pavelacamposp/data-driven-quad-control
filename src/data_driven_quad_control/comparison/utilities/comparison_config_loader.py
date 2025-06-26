"""
Load configuration parameters for controller comparison.

This module provides functionality for loading configuration parameters from a
YAML config file for instantiating controllers and defining the scenario in a
controller comparison simulation.
"""

import torch
from direct_data_driven_mpc.utilities.controller.controller_params import (
    get_nonlinear_data_driven_mpc_controller_params,
)

from data_driven_quad_control.utilities.config_utils import load_yaml_config

from .controller_comparison_config import ControllerComparisonParams


def load_controller_comparison_params(
    config_path: str, env_device: torch.device, verbose: int = 0
) -> ControllerComparisonParams:
    """
    Load configuration parameters for a data-driven controller comparison from
    a YAML config file.

    The YAML configuration file must have the following structure:

        tracking_controller:
          config_path: configs/tracking_controller_params.yaml

        dd_mpc_controller:
          config_path: configs/dd_mpc_controller_params.yaml
          controller_key: nonlinear_dd_mpc_key

        rl_ppo_model:
          model_path: models/trained_model.pt

        comparison_params:
          init_hover_pos: [0.0, 0.0, 1.5]
          eval_setpoints:
            - [0.0, 0.0, 2.5]
          steps_per_setpoint: 150

        camera_config:
          pos: [3.0, 0.0, 3.5]
          lookat: [0.0, 0.0, 1.5]
          fov: 40

    Args:
        config_path (str): The path to the YAML configuration file containing
            the controller comparison parameters.
        env_device (torch.device): The drone environment device.
        verbose (int): The verbosity level: 0 = no output, 1 = minimal output,
            2 = detailed output. Defaults to 0.

    Returns:
        ControllerComparisonParams: A `NamedTuple` containing configuration
            parameters for the controller comparison.
    """
    # Load parameters from config file
    config = load_yaml_config(config_path)

    if verbose > 1:
        print(
            "  Parameters for the controller comparison loaded from "
            f"{config_path}"
        )

    # Load tracking controller parameters
    tracking_controller_config_path = config["tracking_controller"][
        "config_path"
    ]
    tracking_controller_config = load_yaml_config(
        tracking_controller_config_path
    )

    # Load RL controller parameters: trained PPO model path
    ppo_model_path = config["rl_ppo_model"]["model_path"]

    # Load data-driven MPC controller parameters
    dd_mpc_controller_config_path = config["dd_mpc_controller"]["config_path"]
    dd_mpc_controller_key = config["dd_mpc_controller"]["controller_key"]
    m = 3  # Number of inputs (considering CTBR_FIXED_YAW env actions)
    p = 3  # Number of outputs (drone position)

    dd_mpc_controller_config = get_nonlinear_data_driven_mpc_controller_params(
        config_file=dd_mpc_controller_config_path,
        controller_key=dd_mpc_controller_key,
        m=m,
        p=p,
        verbose=verbose,
    )

    # Load controller comparison parameters
    comparison_params_raw = config["comparison_params"]
    init_hover_pos = torch.as_tensor(
        comparison_params_raw["init_hover_pos"],
        dtype=torch.float,
        device=env_device,
    )
    eval_setpoints = [
        torch.as_tensor(
            setpoint, dtype=torch.float, device=env_device
        ).unsqueeze(0)
        for setpoint in comparison_params_raw["eval_setpoints"]
    ]
    steps_per_setpoint = comparison_params_raw["steps_per_setpoint"]

    # Load video recording parameters
    camera_config_params_raw = config["camera_config"]
    camera_config = {
        "res": camera_config_params_raw["res"],
        "pos": camera_config_params_raw["pos"],
        "lookat": camera_config_params_raw["lookat"],
        "fov": camera_config_params_raw["fov"],
    }

    return ControllerComparisonParams(
        tracking_controller_config,
        ppo_model_path,
        dd_mpc_controller_config,
        init_hover_pos,
        eval_setpoints,
        steps_per_setpoint,
        camera_config,
    )
