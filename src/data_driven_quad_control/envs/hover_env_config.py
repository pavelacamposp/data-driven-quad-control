from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch

from data_driven_quad_control.controllers.ctbr.ctbr_controller_config import (
    CTBRControllerConfig,
)
from data_driven_quad_control.drone_config.drone_params import DroneParams
from data_driven_quad_control.utilities.config_utils import load_yaml_config
from data_driven_quad_control.utilities.vectorized_pid_controller import (
    VectorizedControllerState,
)

# Config file paths for drone and CTBR controller parameters
DRONE_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "../../../configs/drone/cf2x_drone_params.yaml"
)

CTBR_CONTROLLER_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../configs/controllers/ctbr/ctbr_controller_params.yaml",
)


# Drone environment configuration
CfgDict = dict[str, Any]


def get_cfgs() -> tuple[CfgDict, CfgDict, CfgDict, CfgDict]:
    env_cfg = {
        # simulation
        "dt": 0.01,  # sim freq = 100 Hz
        "decimation": 4,  # ctrl freq = 1 / (0.01 * 4) = 25 Hz
        # actions
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        # noise (std dev) added to actuators (rotor RPMs)
        "actuator_noise_std": 0.0,  # [RPM]
        # termination
        "termination_if_roll_greater_than": 180,  # degree
        "termination_if_pitch_greater_than": 180,
        "termination_if_close_to_ground": 0.1,
        "termination_if_x_greater_than": 3.0,
        "termination_if_y_greater_than": 3.0,
        "termination_if_z_greater_than": 2.0,
        "termination_if_ang_vel_greater_than": 20,  # rad/s
        "termination_if_lin_vel_greater_than": 20,  # m/s
        # drone initial pose
        "base_init_pos": [0.0, 0.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        # episode config
        "episode_length_s": 15.0,
        "at_target_threshold": 0.1,
        "min_hover_time_s": 0.5,  # Min time (sec) at target before updating
        "resampling_time_s": 3.0,
        # visualization
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 100,  # 1 / dt = 100 Hz
    }
    obs_cfg = {
        "obs_scales": {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
        },
        # noise (std dev) added to normalized observations
        "obs_noise_std": 0.0,
    }
    reward_cfg = {
        "yaw_lambda": -10.0,
        "reward_scales": {
            "target": 10.0,
            "hover_time": 0.01,
            "smooth": -1e-4,
            "yaw": 0.01,
            "angular": -2e-4,
            "crash": -10.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "pos_x_range": [-1.0, 1.0],
        "pos_y_range": [-1.0, 1.0],
        "pos_z_range": [1.0, 1.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


# Define drone physical and rotor parameters
class EnvDroneParams:
    # Load drone parameters from YAML file
    _params: DroneParams = load_yaml_config(DRONE_PARAMS_PATH)

    # Retrieve drone config parameters
    _DRONE_PHYSICAL_PARAMS = _params["drone_physical_params"]
    _DRONE_ROTOR_PARAMS = _params["drone_rotor_params"]

    MASS = _DRONE_PHYSICAL_PARAMS["mass"]
    WEIGHT = MASS * 9.81  # Define weight from mass [N]
    INERTIA = _DRONE_PHYSICAL_PARAMS["inertia"]

    KF = _DRONE_ROTOR_PARAMS["kf"]
    KM = _DRONE_ROTOR_PARAMS["km"]
    ARM_LENGTH = _DRONE_ROTOR_PARAMS["arm_length"]
    ROTOR_ANGLES_DEG = _DRONE_ROTOR_PARAMS["rotor_angles_deg"]
    ROTOR_SPIN_DIRECTIONS = _DRONE_ROTOR_PARAMS["rotor_spin_directions"]

    @classmethod
    def get(cls) -> DroneParams:
        return cls._params


# Define types of environment actions
class EnvActionType(Enum):
    # Individual rotor RPM values
    # Action: [RPM_1, RPM_2, RPM_3, RPM_4]
    #   where RPM_i is the RPM value for the i-th rotor.
    ROTOR_RPMS = 0

    # Collective Thrust and Body Rates (CTBR)
    # Action: [F_z, w_x, w_y, w_z]
    #   where:
    #     - F_z: The total thrust force [N] of the drone's propellers.
    #     - w_x: The roll angular velocity [rad/s].
    #     - w_y: The pitch angular velocity [rad/s].
    #     - w_z: The yaw angular velocity [rad/s].
    CTBR = 1

    # Collective Thrust and Body Rates (CTBR) with fixed yaw
    # Action: [F_z, w_x, w_y]
    #   where:
    #     - F_z: The total thrust force [N] of the drone's propellers.
    #     - w_x: The roll angular velocity [rad/s].
    #     - w_y: The pitch angular velocity [rad/s].
    # Note: The yaw angular velocity w_z is set to 0.
    CTBR_FIXED_YAW = 2

    @classmethod
    def get_num_actions(cls, action_type: EnvActionType) -> int:
        num_actions_mapping = {
            cls.ROTOR_RPMS: 4,
            cls.CTBR: 4,
            cls.CTBR_FIXED_YAW: 3,
        }
        if action_type not in num_actions_mapping:
            raise ValueError(f"Unknown action_type: {action_type}")

        return num_actions_mapping[action_type]

    @classmethod
    def get_num_obs(cls, action_type: EnvActionType) -> int:
        num_obs_mapping = {
            cls.ROTOR_RPMS: 17,
            cls.CTBR: 17,
            cls.CTBR_FIXED_YAW: 16,
        }
        if action_type not in num_obs_mapping:
            raise ValueError(f"Unknown action_type: {action_type}")

        return num_obs_mapping[action_type]


# Define action min-max bounds
class EnvActionBounds:
    # Rotor RPM bounds
    BASE_RPM = 14468.429183500699  # Propeller RPMs for hovering
    MIN_RPM = 0.2 * BASE_RPM
    MAX_RPM = 1.8 * BASE_RPM

    # Collective Thrust and Body Rates bounds
    MAX_THRUST = 2 * EnvDroneParams.WEIGHT  # Max total thrust force [N]
    MAX_ANG_VELS = [15.0, 15.0, 10.0]  # Max roll, pitch, yaw ang vels [rad/s]


# Define CTBR controller configuration parameters
class EnvCTBRControllerConfig:
    # Load CTBR controller configuration from YAML file
    _config: CTBRControllerConfig = load_yaml_config(
        CTBR_CONTROLLER_CONFIG_PATH,
    )

    RATE_PID_GAINS = _config["ctbr_controller_params"]["rate_pid_gains"]

    @classmethod
    def get(cls) -> CTBRControllerConfig:
        return cls._config


# Drone environment state for saving and loading
@dataclass
class EnvState:
    base_pos: torch.Tensor
    base_quat: torch.Tensor
    base_lin_vel: torch.Tensor
    base_ang_vel: torch.Tensor
    commands: torch.Tensor
    episode_length: torch.Tensor
    last_actions: torch.Tensor
    ctbr_controller_state: Optional[VectorizedControllerState] = None

    def to(self, device: torch.device | str) -> EnvState:
        return EnvState(
            base_pos=self.base_pos.to(device),
            base_quat=self.base_quat.to(device),
            base_lin_vel=self.base_lin_vel.to(device),
            base_ang_vel=self.base_ang_vel.to(device),
            commands=self.commands.to(device),
            episode_length=self.episode_length.to(device),
            last_actions=self.last_actions.to(device),
            ctbr_controller_state=(
                self.ctbr_controller_state.to(device)
                if self.ctbr_controller_state
                else None
            ),
        )

    def to_cpu(self) -> EnvState:
        return self.to("cpu")
