from typing import Any

from data_driven_quad_control.envs.hover_env_config import EnvActionType


def get_reward_cfg_by_action_type(
    action_type: EnvActionType,
) -> dict[str, Any]:
    if action_type == EnvActionType.ROTOR_RPMS:
        reward_cfg = {
            "yaw_lambda": -10.0,
            "reward_scales": {
                "target": 10.0,
                "closeness": 1.5,
                "hover_time": 0.01,
                "smooth": -1e-4,
                "yaw": 0.01,
                "angular": -2e-4,
                "crash": -10.0,
            },
        }
    elif action_type == EnvActionType.CTBR:
        reward_cfg = {
            "yaw_lambda": -10.0,
            "reward_scales": {
                "target": 10.0,
                "closeness": 1.5,
                "hover_time": 0.01,
                "smooth": -0.01,
                "yaw": 0.01,
                "angular": -2e-4,
                "crash": -10.0,
            },
        }
    elif action_type == EnvActionType.CTBR_FIXED_YAW:
        reward_cfg = {
            "yaw_lambda": -10.0,
            "reward_scales": {
                "target": 10.0,
                "closeness": 1.5,
                "hover_time": 0.01,
                "smooth": -0.01,
                "yaw": 0.0,  # Cannot be controlled, as the Yaw action is fixed
                "angular": -2e-4,
                "crash": -10.0,
            },
        }
    else:
        raise ValueError(f"Unsupported action type: {action_type}")

    return reward_cfg
