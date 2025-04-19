"""
Collective Thrust and Body Rates (CTBR) Controller Example Script

This script demonstrates the initialization, operation, state saving/loading,
and state reset of a `DroneCTBRController` using mock drone body rate
measurements.
"""

import argparse

import numpy as np
import torch

from data_driven_quad_control.controllers.ctbr.ctbr_controller import (
    DroneCTBRController,
)
from data_driven_quad_control.controllers.ctbr.ctbr_controller_config import (
    CTBRControllerConfig,
)
from data_driven_quad_control.drone_config.drone_params import (
    DroneConfig,
)
from data_driven_quad_control.utilities.vectorized_pid_controller import (
    VectorizedControllerState,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drone CTBR controller example in Genesis"
    )

    parser.add_argument(
        "--num_envs",
        type=int,
        default=2,
        help="The number of parallel envs the controller is used in .",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_envs = args.num_envs

    # Define controller and drone configurations
    drone_config: DroneConfig = {
        "drone_params": {
            "mass": 0.027,
            "inertia": {
                "Jxx": 1.4e-5,
                "Jxy": 0.0,
                "Jxz": 0.0,
                "Jyy": 1.4e-5,
                "Jyz": 0.0,
                "Jzz": 2.17e-5,
            },
        },
        "drone_rotor_params": {
            "kf": 3.16e-10,
            "km": 7.94e-12,
            "arm_length": 0.0397,
            "rotor_angles_deg": [45, 135, 225, 315],
            "rotor_spin_directions": [-1, 1, -1, 1],
        },
    }

    controller_config: CTBRControllerConfig = {
        "ctbr_controller_params": {
            "dt": 0.04,  # 25 Hz
            "pid_coefficients": [
                [1.0, 0.1, 1.0],  # Roll rate
                [1.0, 0.1, 1.0],  # Pitch rate
                [1.0, 0.1, 1.0],  # Yaw rate
            ],
        }
    }

    # Initialize `DroneCTBRController`
    device = "cuda"
    controller = DroneCTBRController(
        drone_config=drone_config,
        controller_config=controller_config,
        num_envs=num_envs,
        device=device,
    )
    print("Initialized `DroneCTBRController`")

    # Mock rate setpoints and measurements
    rate_setpoints = torch.tensor(
        [0.1, 0.0, -0.1], dtype=torch.float, device=device
    ).expand(num_envs, -1)

    rate_measurements = torch.tensor(
        [0.1, 0.2, 0.3], dtype=torch.float, device=device
    ).expand(num_envs, -1)

    # Mock thrust setpoints
    drone_mass = drone_config["drone_params"]["mass"]
    thrust_setpoints = torch.tensor(
        [drone_mass * 9.81] * num_envs, device=device
    )

    print("Mock Setpoints and Measurements:")
    print("  Rate Setpoints:")
    print_formatted_tensor(rate_setpoints, indentation_level=2)
    print("  Rate Measurements:")
    print_formatted_tensor(rate_measurements, indentation_level=2)
    print("  Thrust Setpoints:")
    print_formatted_tensor(thrust_setpoints, indentation_level=2)

    # ----- Control state saving/loading -----
    print("\nControl State Saving/Loading:")
    print("-" * 29)

    # Run controller to modify its internal state
    print("Running controller to modify its internal state...")
    for _ in range(10):
        controller_rpms = controller.compute(
            rate_measurements=rate_measurements,
            rate_setpoints=rate_setpoints,
            thrust_setpoints=thrust_setpoints,
        )

    # Save controller initial state
    init_ctbr_state = controller.get_state()

    print("\nSaving controller initial state:")
    print_formatted_controller_state(init_ctbr_state)

    # Run controller with initial state
    n_iter = 10
    compare_iter = 5
    rpms_before_reload = None

    print("\nRunning controller before state reload...")
    for i in range(n_iter + 1):
        controller_rpms = controller.compute(
            rate_measurements=rate_measurements,
            rate_setpoints=rate_setpoints,
            thrust_setpoints=thrust_setpoints,
        )

        if i == compare_iter:
            rpms_before_reload = controller_rpms.clone()
            print(f"  Controller RPMs at iteration ({i}):")
            print_formatted_tensor(rpms_before_reload, indentation_level=2)

    print(f"  Controller RPMs at iteration ({n_iter}):")
    print_formatted_tensor(controller_rpms, indentation_level=2)

    print("\nController final state:")
    print_formatted_controller_state(controller.get_state())

    # Reload controller state and recompute
    controller.load_state(init_ctbr_state)

    print("\nReloading controller state with saved initial state...")
    print("\nController loaded state:")
    print_formatted_controller_state(controller.get_state())

    print("\nRunning controller after state reload...")
    for _ in range(compare_iter + 1):
        controller_rpms = controller.compute(
            rate_measurements=rate_measurements,
            rate_setpoints=rate_setpoints,
            thrust_setpoints=thrust_setpoints,
        )

    print(f"  Controller RPMs at iteration ({compare_iter}):")
    print_formatted_tensor(controller_rpms, indentation_level=2)

    # Compare RPMs before and after reload
    assert rpms_before_reload is not None
    max_diff = torch.abs(controller_rpms - rpms_before_reload).max()
    print(
        "\nMax diff of controller RPMs before and after state reload: "
        f"{max_diff:.4e}"
    )

    # ----- Control state reset -----
    print("\nControl State Reset:")
    print("-" * 20)

    print("Current controller state:")
    print_formatted_controller_state(controller.get_state())

    # Reset controller
    controller.reset()

    # Print controller state
    print("\nController state after reset:")
    print_formatted_controller_state(controller.get_state())


def print_formatted_tensor(
    data_tensor: torch.Tensor, indentation_level: int = 0
) -> None:
    data_array = data_tensor.cpu().numpy()
    formatted_array = np.array2string(
        data_array, precision=4, suppress_small=True
    )
    indented = "\n".join(
        "  " * indentation_level + line
        for line in formatted_array.splitlines()
    )
    print(indented)


def print_formatted_controller_state(
    ctrl_state: VectorizedControllerState,
) -> None:
    print("VectorizedControllerState(")
    print("  integral:")
    print_formatted_tensor(ctrl_state.integral, indentation_level=2)
    print("  prev_error:")
    print_formatted_tensor(ctrl_state.prev_error, indentation_level=2)
    print(")")


if __name__ == "__main__":
    main()
