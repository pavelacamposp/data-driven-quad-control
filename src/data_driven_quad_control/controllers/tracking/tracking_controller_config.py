from dataclasses import dataclass
from typing import TypedDict

import torch


class TrackingControllerParams(TypedDict):
    pos_pid_gains: list[list[float]]
    kR: float


class TrackingControllerConfig(TypedDict):
    tracking_controller_params: TrackingControllerParams


@dataclass
class TrackingCtrlDroneState:
    X: torch.Tensor  # Position
    Q: torch.Tensor  # Rotation quaternion
