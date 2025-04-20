import torch

from data_driven_quad_control.drone_config.drone_params import (
    DroneParams,
)
from data_driven_quad_control.utilities.vectorized_pid_controller import (
    VectorizedControllerState,
    VectorizedPIDController,
)

from .ctbr_controller_config import CTBRControllerConfig


class DroneCTBRController:
    """
    A Collective Thrust and Body Rates (CBTR) drone controller.

    This controller computes rotor RPMs for multirotor drones based on desired
    total thrust and roll, pitch, and yaw (RPY) angular rate setpoints, along
    with measured RPY angular rates. It supports multiple independent drones
    in parallel across vectorized environments.
    """

    def __init__(
        self,
        drone_params: DroneParams,
        controller_config: CTBRControllerConfig,
        num_envs: int,
        device: torch.device | str = "cuda",
    ):
        """
        Initialize the CTBR controller.

        Args:
            drone_params (DroneParams): The drone configuration parameters.
            controller_config (CTBRControllerConfig): The CTBR controller
                configuration parameters.
            num_envs (int): The number of drone environments the controller is
                used in (i.e., the number of drones simulated in a vectorized
                environment).
            device (torch.device | str): The device to run the controller on.
        """
        self.num_envs = num_envs
        self.device = device

        # Retrieve drone and controller parameters
        drone_physical_params = drone_params["drone_physical_params"]
        drone_rotor_params = drone_params["drone_rotor_params"]
        controller_params = controller_config["ctbr_controller_params"]

        # Drone parameters
        self.inertia = drone_physical_params["inertia"]
        # Force coefficient
        self.KF = torch.as_tensor(
            drone_rotor_params["kf"], dtype=torch.float, device=self.device
        )
        # Moment coefficient
        self.KM = torch.as_tensor(
            drone_rotor_params["km"], dtype=torch.float, device=self.device
        )

        # Inertia matrix J
        self.J = torch.tensor(
            [
                [
                    self.inertia["Jxx"],
                    self.inertia["Jxy"],
                    self.inertia["Jxz"],
                ],
                [
                    self.inertia["Jxy"],
                    self.inertia["Jyy"],
                    self.inertia["Jyz"],
                ],
                [
                    self.inertia["Jxz"],
                    self.inertia["Jyz"],
                    self.inertia["Jzz"],
                ],
            ],
            dtype=torch.float,
            device=self.device,
        )

        # Rotor positions and spin directions
        self.arm_length = drone_rotor_params["arm_length"]
        self.rotor_angles = torch.deg2rad(
            torch.as_tensor(
                drone_rotor_params["rotor_angles_deg"],
                dtype=torch.float,
                device=self.device,
            )
        )
        self.rotor_spin_directions = torch.as_tensor(
            drone_rotor_params["rotor_spin_directions"],
            dtype=torch.int,
            device=self.device,
        )

        # Controller parameters
        self.dt = controller_params["dt"]
        self.rate_pid_params = torch.as_tensor(
            controller_params["pid_coefficients"],
            dtype=torch.float,
            device=self.device,
        )

        # Calculate allocation matrix A
        self.A = self._calculate_allocation_matrix()
        self.A_inv = torch.linalg.pinv(self.A)

        # Construct control mixer
        self.mixer = self._construct_mixer()

        # Initialize PID controller for rates
        self.rate_pid_controller = VectorizedPIDController(
            pid_params=self.rate_pid_params,
            num_envs=self.num_envs,
            dt=self.dt,
            device=self.device,
        )
        self.ctrl_ang_acc_thrust = torch.zeros(
            (self.num_envs, 4), dtype=torch.float, device=self.device
        )

        # Precompute values
        self.inv_sqrt_KF = torch.rsqrt(self.KF)

    def _calculate_allocation_matrix(self) -> torch.Tensor:
        """
        Calculate the allocation matrix A for a drone.

        The allocation matrix maps individual rotor thrust forces to drone
        torques and total thrust force. For a quadcopter, it is defined as
        follows:

            [t_x] = A @ [F_1],
            [t_y]       [F_2]
            [t_z]       [F_3]
            [F_z]       [F_4]
        where:
        - t_x: The roll torque.
        - t_y: The pitch torque.
        - t_z: The yaw torque.
        - F_z: The total thrust force along the drone's Z-axis.
        - F_1 - F_4: The individual thrust forces of each drone rotor.

        The allocation matrix A is a (4 x `n`) matrix, where `n` is the number
        of drone rotors. For a quadcopter, it is constructed as follows:

            A = [   -l * sin(theta_1)     ...    -l * sin(theta_4)    ],
                [   -l * cos(theta_1)     ...    -l * cos(theta_4)    ]
                [ KM / KF * rotor_spin_1  ...  KM / KF * rotor_spin_4 ]
                [           1             ...            1            ]
        where:
        - l: The drone arm's length.
        - theta_1 - theta_4: The angular position of each drone rotor relative
            to the drone's center, measured from the forward direction. (e.g.,
            45°, 135°, 225°, 315° for a quadcopter).
        - KM: The rotor moment coefficient.
        - KF: The rotor force coefficient.
        - rotor_spin_1 - rotor_spin_4: The spin direction of each drone
            rotor (+1 for counterclockwise, -1 for clockwise).

        Returns:
            torch.Tensor: The drone's allocation matrix.
        """
        A = torch.vstack(
            [
                -torch.sin(self.rotor_angles) * self.arm_length,
                -torch.cos(self.rotor_angles) * self.arm_length,
                self.KM / self.KF * self.rotor_spin_directions,
                torch.ones_like(self.rotor_angles),
            ]
        )

        return A

    def _construct_mixer(self) -> torch.Tensor:
        """
        Construct the mixer matrix for the rate controller.

        The mixer matrix M maps drone angular accelerations and total thrust
        force to individual rotor thrust forces. For a quadcopter, it is
        defined as follows:

            [F_1] = M @ [w_x_dot],
            [F_2]       [w_y_dot]
            [F_3]       [w_z_dot]
            [F_4]       [  F_z  ]
        where:
          - F_1 - F_4: The individual thrust forces of each drone rotor.
          - w_x_dot: The roll angular acceleration.
          - w_y_dot: The pitch angular acceleration.
          - w_z_dot: The yaw angular acceleration.
          - F_z: The total thrust force along the drone's Z-axis.

        The M matrix is constructed as:

            M = inv(A) @ [J_x  0   0   0],
                         [ 0  J_y  0   0]
                         [ 0   0  J_z  0]
                         [ 0   0   0   1]
        where A is the drone's allocation matrix, and J_x, J_y, and J_z are
        the drone's inertia values along its X, Y, and Z axis, respectively.

        Returns:
            torch.Tensor: The rate controller's mixer matrix.
        """
        # Expand inertia matrix to include the total thrust
        J_exp = torch.eye(4, dtype=torch.float, device=self.device)
        J_exp[:3, :3] = self.J

        # Construct mixer
        mixer = self.A_inv @ J_exp

        return mixer

    def compute(
        self,
        rate_measurements: torch.Tensor,
        rate_setpoints: torch.Tensor,
        thrust_setpoints: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute rotor RPMs based on angular rate and thrust setpoints.

        This method uses a PID controller to compute angular acceleration
        commands, then maps them into rotor thrusts using the mixer matrix.

        Args:
            rate_measurements (torch.Tensor): The measured angular rates with
                a (`num_envs`, 3) shape.
            rate_setpoints (torch.Tensor): The desired roll, pitch, and yaw
                angular rates with a (`num_envs`, 3) shape.
            thrust_setpoints (torch.Tensor): The desired collective thrust
                with a (`num_envs`,) shape.

        Returns:
            torch.Tensor: The drone rotor RPMs of shape (`num_envs`, 4).
        """
        ctrl_ang_acc = self.rate_pid_controller.compute(
            setpoints=rate_setpoints, measurements=rate_measurements
        )

        # Construct control action
        self.ctrl_ang_acc_thrust[:, :3] = ctrl_ang_acc
        self.ctrl_ang_acc_thrust[:, 3] = thrust_setpoints

        # Compute individual rotor thrust forces from drone
        # angular accelerations and total thrust using mixer
        rotor_thrusts = self.mixer @ self.ctrl_ang_acc_thrust.transpose(0, 1)
        # Clamp thrust values to prevent negative values
        # that cause NaNs in RPM calculation
        rotor_thrusts = torch.clamp(rotor_thrusts, min=0)

        # Calculate rotor RPMs from thrusts
        # thrust_i = RPM_i ** 2 * KF
        rotor_rpms = torch.sqrt(rotor_thrusts) * self.inv_sqrt_KF

        # Return drone rotor RPMs in a (`num_envs`, 4) shape
        return rotor_rpms.transpose(0, 1)

    def reset(self) -> None:
        """
        Reset the internal state of the controller.

        This method clears the accumulated integral and previous error terms
        in the controller's internal vectorized PID controller.
        """
        self.rate_pid_controller.reset()

    def get_state(self) -> VectorizedControllerState:
        """
        Return the current CTBR controller state.

        Returns:
            VectorizedControllerState: The internal state of the controller.
        """
        return self.rate_pid_controller.get_state()

    def load_state(self, state: VectorizedControllerState) -> None:
        """
        Restore the CTBR controller state from a given state dictionary.

        Args:
            state (VectorizedControllerState): The state to which the
                controller is restored.
        """
        self.rate_pid_controller.load_state(state)
