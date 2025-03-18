import os
from enum import Enum
from typing import Any

import yaml

# Config file paths for CTBR controller parameters
DRONE_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../controllers/ctbr/config/cf2x_drone_params.yaml",
)

CTBR_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../controllers/ctbr/config/ctbr_controller_params.yaml",
)


def load_yaml_config(path: str) -> Any:
    with open(path, "r") as file:
        return yaml.safe_load(file)


# Define drone-related parameters
class EnvDrone:
    MASS = 0.027
    WEIGHT = MASS * 9.81


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


# Define action min-max bounds
class EnvActionBounds:
    # Rotor RPM bounds
    BASE_RPM = 14468.429183500699  # Propeller RPMs for hovering
    MIN_RPM = 0.2 * BASE_RPM
    MAX_RPM = 1.8 * BASE_RPM

    # Collective Thrust and Body Rates bounds
    MAX_THRUST = 2 * EnvDrone.WEIGHT  # Max total thrust force
    MAX_ANG_VELS = [0.1, 0.1, 5.0]  # Max roll, pitch, yaw angular velocities


# Load CTBR controller configuration for env CTBR controller initialization
class EnvCTBRControllerConfig:
    @staticmethod
    def get_drone_config() -> Any:
        return load_yaml_config(path=DRONE_CONFIG_PATH)

    @staticmethod
    def get_controller_config() -> Any:
        return load_yaml_config(path=CTBR_CONFIG_PATH)
