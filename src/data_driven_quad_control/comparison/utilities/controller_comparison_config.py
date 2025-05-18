from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch
from direct_data_driven_mpc.utilities.controller.controller_params import (
    NonlinearDataDrivenMPCParams,
)

from data_driven_quad_control.controllers.tracking.tracking_controller_config import (  # noqa: E501
    TrackingControllerConfig,
    TrackingCtrlDroneState,
)


class ControllerComparisonParams(NamedTuple):
    tracking_controller_config: TrackingControllerConfig
    ppo_model_path: str
    dd_mpc_controller_config: NonlinearDataDrivenMPCParams
    init_hover_pos: torch.Tensor
    eval_setpoints: list[torch.Tensor]


class TrackingControllerInitData(NamedTuple):
    controller_config: TrackingControllerConfig
    controller_dt: float
    initial_state: TrackingCtrlDroneState


class DDMPCControllerInitData(NamedTuple):
    controller_config: NonlinearDataDrivenMPCParams
    u_N: np.ndarray
    y_N: np.ndarray


class RLControllerInitData(NamedTuple):
    train_cfg: dict[str, Any]
    model_path: str
    initial_observation: torch.Tensor


@dataclass
class SimInfo:
    in_progress: bool = True
    at_target_steps: int = 0
    stabilized_at_target: bool = False
    current_target_idx: int = 0
    num_targets: int = 0


class EnvTargetSignal(NamedTuple):
    target_pos: torch.Tensor
    is_new_target: bool
    done: bool
