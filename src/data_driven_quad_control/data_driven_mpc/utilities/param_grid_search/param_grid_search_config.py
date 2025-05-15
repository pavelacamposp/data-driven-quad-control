from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np
from direct_data_driven_mpc.nonlinear_data_driven_mpc_controller import (
    AlphaRegType,
)

from data_driven_quad_control.envs.config.hover_env_config import EnvState


class DDMPCInitialDataCollectionParams(NamedTuple):
    """
    Parameters for the initial input-output data collection phase in a
    Data-Driven MPC parameter grid search.
    """

    init_hover_pos: np.ndarray
    u_range: np.ndarray


class DDMPCFixedParams(NamedTuple):
    """
    Parameters that remain fixed during a Data-Driven MPC parameter grid
    search.
    """

    m: int
    p: int
    Q_weights: list[float]
    R_weights: list[float]
    S_weights: list[float]
    U: np.ndarray
    Us: np.ndarray
    alpha_reg_type: AlphaRegType
    ext_out_incr_in: bool
    n_n_mpc_step: bool


class DDMPCEvaluationParams(NamedTuple):
    """
    Parameters used for evaluating controllers during a Data-Driven MPC
    controller parameter grid search.
    """

    eval_time_steps: int
    eval_setpoints: list[np.ndarray]
    max_target_dist_increment: float


class DDMPCCombinationParams(NamedTuple):
    """
    A single combination of parameters used in a Data-Driven MPC parameter
    grid search.
    """

    N: int
    n: int
    L: int
    lamb_alpha: float
    lamb_sigma: float
    lamb_alpha_s: float
    lamb_sigma_s: float


class DDMPCParameterGrid(NamedTuple):
    """
    A grid of Data-Driven MPC parameter combinations for use in a grid
    search.
    """

    N: list[int]
    n: list[int]
    L: list[int]
    lamb_alpha: list[float]
    lamb_sigma: list[float]
    lamb_alpha_s: list[float]
    lamb_sigma_s: list[float]


DDMPCGridSearchParams = tuple[
    DDMPCInitialDataCollectionParams,
    DDMPCFixedParams,
    DDMPCEvaluationParams,
    DDMPCParameterGrid,
]


class DataDrivenCache(NamedTuple):
    """
    Cache of initial input-output measurements used for creating Data-Driven
    MPC controllers, and the corresponding drone environment state obtained
    after collection. Entries are grouped by an integer key, which corresponds
    to the input-output trajectory length used for measurements.
    """

    u_N: dict[int, np.ndarray]
    y_N: dict[int, np.ndarray]
    drone_state: dict[int, EnvState]


@dataclass
class EnvSimInfo:
    """
    Tracks the environment reset status and the closed-loop simulation step
    progress for a single environment.

    Used in controller evaluations for the following tasks:
    - Communicate drone state resets to the main process.
    - Track closed-loop simulation step progress. This allows sending dummy
      values to the main process and retrieving data from it via
      multiprocessing queues in case of interruptions, ensuring synchronized
      communication and preventing deadlocks.
    """

    reset_state: bool = False
    sim_step_progress: int = 0


class CtrlEvalStatus(Enum):
    """The outcome status of a controller evaluation."""

    SUCCESS = 0
    FAILURE = 1


class EnvResetSignal(NamedTuple):
    """
    Message object used for communicating environment reset and termination
    signals between worker processes and the main process during controller
    parameter evaluations in a grid search.

    Attributes:
        env_idx (int): The index of the environment instance (worker).
        reset (bool): Indicates whether the environment should be reset.
        done (bool): Indicates whether the worker has finished all evaluations.
        N (int | None): The input-output trajectory length associated with the
            evaluation. Not required when signaling task completion.
    """

    env_idx: int
    reset: bool
    done: bool
    N: int | None
