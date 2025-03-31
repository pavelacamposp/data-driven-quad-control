import torch


class VectorizedPIDController:
    """
    A vectorized PID controller that handles multiple independent controllers
    across multiple environments.

    Attributes:
        kp (torch.Tensor): Proportional gains, shaped (`num_envs`,
            `num_ctrls`).
        ki (torch.Tensor): Integral gains, shaped (`num_envs`, `num_ctrls`).
        kd (torch.Tensor): Derivative gains, shaped (`num_envs`, `num_ctrls`).
        integral (torch.Tensor): The accumulated integral error for each
            independent controller.
        prev_error (torch.Tensor): The previous step error for each
            independent controller.
        dt (float): The environment simulation time step.
        num_envs (int): The number of environments the controller is used in.
        device (torch.device | str): The device on which the controller runs.
        num_ctrls (int): The number of independent PID controllers the
            vectorized controller manages.
    """

    def __init__(
        self,
        pid_params: torch.Tensor,
        num_envs: int,
        dt: float,
        device: torch.device | str = "cuda",
    ):
        """
        Initialize the vectorized PID controller.

        Args:
            pid_params (torch.Tensor): A tensor of shape (`num_ctrls`, 3)
                containing the P, I, and D gains for each of the `num_ctrls`
                controllers.
            num_envs (int): The number of environments the controller is used
                in (i.e., the number of parallel controller instances).
            dt (float): The environment simulation time step. Used for
                integration and differentiation.
            device (torch.device | str): The device to run the controller on.

        Raises:
            ValueError: If `pid_params` does not have shape (`num_ctrls`, 3).
        """
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.num_ctrls = pid_params.shape[0]

        # PID parameters
        if pid_params.shape[1] != 3:
            raise ValueError(
                "PID params must have shape (`num_ctrls`, 3), representing "
                "the P, I, and D gains for each of the `num_ctrls` "
                "controllers."
            )

        self.kp = pid_params[:, 0].expand(num_envs, -1)
        self.ki = pid_params[:, 1].expand(num_envs, -1)
        self.kd = pid_params[:, 2].expand(num_envs, -1)

        self.integral = torch.zeros(
            (self.num_envs, self.num_ctrls),
            dtype=torch.float,
            device=self.device,
        )
        self.prev_error = torch.zeros(
            (self.num_envs, self.num_ctrls),
            dtype=torch.float,
            device=self.device,
        )

    def compute(
        self, setpoints: torch.Tensor, measurements: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the vectorized controller output.

        Args:
            setpoints (torch.Tensor): The setpoints for each independent
                controller. Must have a (`num_envs`, `num_ctrls`) shape.
            measurements (torch.Tensor): The current measurements for each
                independent controller. Must have a (`num_envs`, `num_ctrls`)
                shape.

        Returns:
            torch.Tensor: The control output as a (`num_envs`, `num_ctrls`)
                tensor.
        """
        error = setpoints - measurements

        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt

        self.prev_error = error.detach().clone()

        # Calculate the control action
        output = (
            self.kp * error + self.ki * self.integral + self.kd * derivative
        )

        return output

    def reset(self) -> None:
        """
        Reset the internal state of the controller.

        This clears the accumulated integral and previous error terms.
        """
        self.integral.zero_()
        self.prev_error.zero_()

    def get_state(self) -> dict[str, torch.Tensor]:
        """
        Retrieve the current internal state of the controller, which consists
        of the accumulated integral term and the previous error.

        Returns:
            dict[str, torch.Tensor]: The current controller state.
        """
        return {
            "integral": self.integral.clone(),
            "prev_error": self.prev_error.clone(),
        }

    def load_state(self, state: dict[str, torch.Tensor]) -> None:
        """
        Restore the controller's internal state from a given state dictionary.

        Args:
            state (dict[str, torch.Tensor]): The state to which the controller
                is restored.
        """
        self.integral = state["integral"]
        self.prev_error = state["prev_error"]
