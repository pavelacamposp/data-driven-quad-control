import torch


def gs_rand_float(
    lower: float,
    upper: float,
    shape: int | tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def linear_interpolate(
    x: torch.Tensor,
    x_min: float | torch.Tensor,
    x_max: float | torch.Tensor,
    y_min: float | torch.Tensor,
    y_max: float | torch.Tensor,
) -> torch.Tensor:
    return y_min + (x - x_min) * (y_max - y_min) / (x_max - x_min)
