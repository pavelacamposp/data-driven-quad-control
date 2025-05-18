from typing import Any

from data_driven_quad_control.envs.hover_env_config import EnvActionType

# Mapping for env action types
ENV_ACTION_TYPES_MAP = {
    "rpms": EnvActionType.ROTOR_RPMS,
    "ctbr": EnvActionType.CTBR,
    "ctbr_fixed_yaw": EnvActionType.CTBR_FIXED_YAW,
}


# Training configuration
def get_train_cfg(exp_name: str, max_iterations: int) -> dict[str, Any]:
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.004,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "class_name": "ActorCritic",
            "activation": "tanh",
            "actor_hidden_dims": [128, 128],
            "critic_hidden_dims": [128, 128],
            "init_noise_std": 1.0,
        },
        "experiment_name": exp_name,
        "max_iterations": max_iterations,
        "num_steps_per_env": 100,
        "save_interval": 100,
        "empirical_normalization": False,
        "logger": "tensorboard",
    }

    return train_cfg_dict
