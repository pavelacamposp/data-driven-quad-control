"""Define mock classes and objects."""

from typing import Any

import numpy as np
import torch

from data_driven_quad_control.envs.hover_env_config import (
    EnvActionType,
    EnvState,
)
from data_driven_quad_control.utilities.vectorized_pid_controller import (
    VectorizedControllerState,
)


class MockHoverEnv:
    def __init__(self) -> None:
        self.num_envs = 1
        self.num_obs = 3
        self.num_actions = 3
        self.device = torch.device("cpu")
        self.step_dt = 0.1
        self.max_episode_length = 1

        self.obs_scales = {
            "rel_pos": 1.0,
            "lin_vel": 1.0,
            "ang_vel": 1.0,
        }

        self.actions = torch.tensor([[-0.5, 0.0, 0.5]])
        self.last_actions = self.actions.clone()
        self.base_pos = torch.tensor([[0.0, 0.0, 1.0]])
        self.base_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.base_lin_vel = torch.tensor([[0.1, 0.2, 0.3]])
        self.base_ang_vel = torch.tensor([[0.1, 0.2, 0.3]])
        self.action_bounds = torch.tensor(
            [
                [0.0, 1.0],  # Thrust
                [-0.5, 0.5],  # Roll rate
                [-0.3, 0.3],  # Pitch rate
            ]
        )
        self.action_type = EnvActionType.CTBR_FIXED_YAW

        self.actuator_noise_std = 0.0
        self.obs_noise_std = 0.0

        self.rel_pos = self.base_pos.clone()

        self.obs_buf = torch.zeros((self.num_envs, self.num_obs))
        self.rew_buf = torch.zeros((self.num_envs,))
        self.reset_buf = torch.zeros((self.num_envs,))

    def step(self, actions: torch.Tensor) -> Any:
        return self.obs_buf, self.rew_buf, self.reset_buf, {}

    def _add_noise(
        self, input_tensor: torch.Tensor, noise_std: float
    ) -> torch.Tensor:
        return input_tensor + self.obs_noise_std

    def update_target_pos(
        self, envs_idx: torch.Tensor, target_pos: torch.Tensor
    ) -> None:
        return

    def get_current_state(self, envs_idx: torch.Tensor) -> EnvState:
        return EnvState(
            base_pos=torch.zeros((self.num_envs, 3)),
            base_quat=torch.zeros((self.num_envs, 4)),
            base_lin_vel=torch.zeros((self.num_envs, 3)),
            base_ang_vel=torch.zeros((self.num_envs, 3)),
            commands=torch.zeros((self.num_envs, 3)),
            episode_length=torch.tensor([1.0]),
            last_actions=torch.zeros((self.num_envs, 3)),
            ctbr_controller_state=VectorizedControllerState(
                integral=torch.zeros((self.num_envs, 3)),
                prev_error=torch.zeros((self.num_envs, 3)),
            ),
        )

    def restore_from_state(
        self, envs_idx: torch.Tensor, saved_state: EnvState
    ) -> None:
        pass

    def get_pos(self, add_noise: bool = True) -> torch.Tensor:
        return torch.zeros((self.num_envs, 3))

    def get_quat(self, add_noise: bool = True) -> torch.Tensor:
        return torch.zeros((self.num_envs, 4))


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


class MockRLPolicy:
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return obs


class MockRLController:
    def __init__(self) -> None:
        self.n_mpc_step = 1

    def compute(
        self, state_setpoint: torch.Tensor, state_measurement: torch.Tensor
    ) -> torch.Tensor:
        return torch.tensor([[0.1, 0.0, 0.0, 0.0]])

    def reset(self) -> None:
        pass


class MockNonlinearDDMPCController:
    def __init__(self) -> None:
        self.n = 2
        self.N = 40
        self.y_r = np.zeros((3, 1))
        self.n_mpc_step = 1

    def update_and_solve_data_driven_mpc(self) -> None:
        pass

    def get_optimal_control_input_at_step(self, n_step: int) -> np.ndarray:
        return np.array([0.1, 0.1, 0.1])

    def get_du_value_at_step(self, n_step: int) -> np.ndarray:
        return np.zeros(3)

    def store_input_output_measurement(
        self,
        u_current: np.ndarray,
        y_current: np.ndarray,
        du_current: np.ndarray,
    ) -> None:
        pass

    def set_output_setpoint(self, y_r: np.ndarray) -> None:
        return
