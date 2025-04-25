import torch
import torch.nn.functional as F

from data_driven_quad_control.utilities.math_utils import (
    quaternion_to_matrix,
    yaw_from_quaternion,
)
from data_driven_quad_control.utilities.vectorized_pid_controller import (
    VectorizedPIDController,
)

from .tracking_controller_config import (
    TrackingControllerConfig,
    TrackingCtrlDroneState,
)


class DroneTrackingController:
    """
    A simplified SE3 controller that outputs Collective Thrust and Body Rates
    (CTBR) action commands.

    This controller computes CTBR commands consisting of total thrust and roll,
    pitch, and yaw angular rate setpoints for controlling the position and yaw
    orientation of a drone. The CTBR commands are intended to be followed by an
    external lower level CTBR controller. The controller supports multiple
    independent drones in parallel across vectorized environments.
    """

    def __init__(
        self,
        drone_mass: float,
        controller_config: TrackingControllerConfig,
        dt: float,
        num_envs: int,
        device: torch.device | str = "cuda",
    ):
        """
        Initialize the tracking controller.

        Args:
            drone_mass (float): The drone mass.
            controller_config (TrackingControllerConfig): The tracking
                controller configuration parameters.
            dt (float): The controller time step in seconds.
            num_envs (int): The number of drone environments the controller is
                used in (i.e., the number of drones simulated in a vectorized
                environment).
            device (torch.device | str): The device to run the controller on.
        """
        self.device = device
        self.num_envs = num_envs

        # Controller parameters
        self.dt = dt
        controller_params = controller_config["tracking_controller_params"]
        self.pos_pid_params = torch.as_tensor(
            controller_params["pos_pid_gains"],
            dtype=torch.float,
            device=self.device,
        )
        self.kR = controller_params["kR"]

        # Vectorized PID controller for 3D position
        self.controller = VectorizedPIDController(
            pid_params=self.pos_pid_params,
            num_envs=self.num_envs,
            dt=self.dt,
            device=self.device,
        )

        # Gravity compensation
        self.drone_mass = drone_mass
        self.drone_gravity_comp = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device
        )
        self.drone_gravity_comp[:, 2] = 9.81  # Gravity [m/s^2]

        # Preallocate tensors used in control action calculation
        self.x_yaw_des = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device
        )
        self.R_des = torch.zeros(
            (self.num_envs, 3, 3), dtype=torch.float, device=self.device
        )
        self.e_R = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device
        )

    def compute(
        self,
        state_measurement: TrackingCtrlDroneState,
        state_setpoint: TrackingCtrlDroneState,
    ) -> torch.Tensor:
        """
        Compute CTBR (Collective Thrust and Body Rates) commands based on the
        measured and desired drone states.

        Each drone state consists of the position and orientation (quaternion)
        of the drone.

        Args:
            state_measurement (TrackingCtrlDroneState): The measured drone
                state.
            state_setpoint (TrackingCtrlDroneState): The desired drone state.

        Returns:
            torch.Tensor: The CTBR commands of shape (`num_envs`, 4), where
                each command consists of [total thrust (N), `w_x` (rad/s),
                `w_y` (rad/s), `w_z` (rad/s)].
        """
        # Calculate required acceleration in world frame coordinates
        acc_world = self.controller.compute(
            setpoints=state_setpoint.X, measurements=state_measurement.X
        )

        # Add gravity compensation
        acc_world += self.drone_gravity_comp

        # Calculate required force vector in world frame coordinates
        force_world = acc_world * self.drone_mass

        # Calculate the required drone thrust in drone coordinates
        R_current = quaternion_to_matrix(state_measurement.Q)
        thrust = torch.bmm(R_current, force_world.unsqueeze(-1)).squeeze(-1)
        thrust = thrust[:, 2]

        # Normalize the desired force direction to get the desired z axis
        z_des = F.normalize(force_world, dim=1)

        # Calculate the desired body x-axis based on yaw reference
        yaw_des = yaw_from_quaternion(state_setpoint.Q)
        self.x_yaw_des[:, 0] = torch.cos(yaw_des)
        self.x_yaw_des[:, 1] = torch.sin(yaw_des)

        # Form orthonormal desired body frame
        y_des = F.normalize(torch.linalg.cross(z_des, self.x_yaw_des), dim=1)
        x_des = F.normalize(torch.linalg.cross(y_des, z_des), dim=1)

        # Construct desired attitude matrix R_des
        self.R_des[..., 0] = x_des
        self.R_des[..., 1] = y_des
        self.R_des[..., 2] = z_des

        # Calculate relative rotation matrix
        R_des_R = torch.bmm(self.R_des.transpose(1, 2), R_current)

        # Construct e_R matrix to calculate the attitude tracking error
        # e_R_matrix = 0.5 (R_des^T @ R - R^T @ R_des)
        e_R_matrix = 0.5 * (R_des_R - R_des_R.transpose(1, 2))

        # Extract attitude tracking error using the Vee operator
        # e_R = e_R_matrix^V
        self.e_R[:, 0] = e_R_matrix[:, 2, 1]
        self.e_R[:, 1] = e_R_matrix[:, 0, 2]
        self.e_R[:, 2] = e_R_matrix[:, 1, 0]

        # Calculate body rates using a proportional controller
        body_rates = -self.kR * self.e_R

        # Output CTBR: [total thrust, w_x, w_y, w_z]
        ctbr_action = torch.cat([thrust.unsqueeze(1), body_rates], dim=1)

        return ctbr_action

    def reset(self) -> None:
        """
        Reset the internal state of the controller.

        This method clears the accumulated integral and previous error terms
        in the controller's internal vectorized PID controller.
        """
        self.controller.reset()
