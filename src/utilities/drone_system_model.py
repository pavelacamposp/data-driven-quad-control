from functools import partial
from typing import Optional

import numpy as np
import torch
from direct_data_driven_mpc.utilities.models.nonlinear_model import (
    NonlinearSystem,
)

from src.envs.hover_env import HoverEnv


def drone_dynamics(
    x: np.ndarray, u: np.ndarray, env: HoverEnv, env_idx: int
) -> np.ndarray:
    action = u  # Note: action is within the [-1, 1] range

    # Convert action to tensor
    action_tensor = torch.tensor(action, device=env.device).unsqueeze(0)

    # Step simulation
    env.step(action_tensor)

    # Get system state from environment
    # We assume the state to be the base position, omitting other variables
    # since we only require input-output data for controlling the drone
    state = env.base_pos[env_idx].cpu().numpy()

    return state


def drone_output(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return x


def create_system_model(
    env: HoverEnv, env_idx: int, initial_state: Optional[np.ndarray] = None
) -> NonlinearSystem:
    # Define drone system dynamics function
    drone_dynamics_pre_bound = partial(
        drone_dynamics, env=env, env_idx=env_idx
    )

    # Define system model (simulation)
    n = 3  # Number of system states (only necessary for storage
    # since the simulation is handled by the env)
    m = 4  # Number of control inputs
    p = 3  # Number of system outputs
    eps_max = 0.0  # Upper bound of the system measurement noise
    system_model = NonlinearSystem(
        f=drone_dynamics_pre_bound,
        h=drone_output,
        n=n,
        m=m,
        p=p,
        eps_max=eps_max,
    )

    # Set system model's state to match the environment's
    # initial observation if provided
    if initial_state is not None:
        system_model.x = initial_state

    return system_model


def drone_tracking_callback(
    step: int,
    system_model: NonlinearSystem,
    u_sys_k: np.ndarray,
    y_sys_k: np.ndarray,
    y_r: np.ndarray,
    U_k: np.ndarray,
    Y_k: np.ndarray,
    info_dict: dict,
    initial_distance: float,
    distance_threshold: float,
) -> None:
    # Update external variables
    U_k[step, :] = u_sys_k
    Y_k[step, :] = y_sys_k
    info_dict["last_step"] = step

    # Stop simulation by raising a ValueError if the distance from the drone
    # to its setpoint position is greater than the initial distance
    current_drone_pos = np.array(system_model.x[:3]).reshape(-1, 1)
    current_distance = np.linalg.norm(current_drone_pos - y_r)
    if current_distance - initial_distance > distance_threshold:
        raise ValueError(
            f"Drone Drone moved away from its goal by {distance_threshold} "
            "from its starting position."
        )
