"""Define mock classes and objects."""

from typing import Any

import numpy as np
import torch

from data_driven_quad_control.envs.config.hover_env_config import (
    EnvActionType,
    EnvState,
)
from data_driven_quad_control.utilities.vectorized_pid_controller import (
    VectorizedControllerState,
)


class MockHoverEnv:
    def __init__(self) -> None:
        self.num_envs = 1
        self.num_actions = 3
        self.device = torch.device("cpu")
        self.base_pos = torch.tensor([[0.0, 0.0, 1.0]])
        self.base_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.action_bounds = torch.tensor(
            [
                [0.0, 1.0],  # Thrust
                [-0.5, 0.5],  # Roll rate
                [-0.3, 0.3],  # Pitch rate
            ]
        )
        self.action_type = EnvActionType.CTBR_FIXED_YAW

    def step(self, actions: torch.Tensor) -> Any:
        pass

    def get_current_state(self, envs_idx: torch.Tensor) -> EnvState:
        return EnvState(
            base_pos=torch.zeros((1, 3)),
            base_quat=torch.zeros((1, 4)),
            base_lin_vel=torch.zeros((1, 3)),
            base_ang_vel=torch.zeros((1, 3)),
            commands=torch.zeros((1, 3)),
            episode_length=torch.tensor([1.0]),
            last_actions=torch.zeros((1, 3)),
            ctbr_controller_state=VectorizedControllerState(
                integral=torch.zeros((1, 3)),
                prev_error=torch.zeros((1, 3)),
            ),
        )

    def restore_from_state(
        self, envs_idx: torch.Tensor, saved_state: EnvState
    ) -> None:
        pass


class MockDroneSystemModel:
    def __init__(self) -> None:
        self.m = 3
        self.p = 3

    def simulate_step(self, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        return u + w


class MockDroneTrackingController:
    def compute(
        self, state_setpoint: torch.Tensor, state_measurement: torch.Tensor
    ) -> torch.Tensor:
        return torch.tensor([[0.1, 0.0, 0.0, 0.0]])

    def reset(self) -> None:
        pass
