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


def quaternion_to_matrix(quats: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of quaternions to rotation matrices.

    Args:
        quats (torch.Tensor): Batch of quaternions of shape (..., 4) with
            quaternions in a (w, x, y, z) format.

    Returns:
        torch.Tensor: Rotation matrices of shape (..., 3, 3).
    """
    w, x, y, z = quats.unbind(-1)

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w

    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rot = torch.stack(
        [
            ww + xx - yy - zz,
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            ww - xx + yy - zz,
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            ww - xx - yy + zz,
        ],
        dim=-1,
    )

    return rot.reshape(quats.shape[:-1] + (3, 3))


def yaw_from_quaternion(quats: torch.Tensor) -> torch.Tensor:
    """
    Calculate yaw angles from a batch of quaternions.

    Note:
        This function assumes Tait-Bryan angles, applied in ZYX order.

    Args:
        quats (torch.Tensor): Batch of quaternions of shape (..., 4) with
            quaternions in a (w, x, y, z) format.

    Returns:
        torch.Tensor: Yaw angles in radians, shape (...,).
    """
    w, x, y, z = quats.unbind(-1)

    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    return yaw


def yaw_to_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    """
    Convert yaw angles to quaternions in a (w, x, y, z) format assuming zero
    roll and pitch values.

    Note:
        This function assumes Tait-Bryan angles, applied in ZYX order.

    Args:
        yaw (torch.Tensor): Tensor of shape (...) containing yaw angles in
            radians.

    Returns:
        torch.Tensor: Tensor of shape (..., 4) containing quaternions in a
            (w, x, y, z) format.
    """
    quat = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=yaw.dtype)
    quat[:, 0] = torch.cos(yaw / 2)  # w
    quat[:, 3] = torch.sin(yaw / 2)  # z

    return quat


if __name__ == "__main__":
    quats = torch.tensor(
        [
            [0.7071, 0.0000, 0.0000, 0.7071],  # r: 0°, p: 0°, y: 90°
            [0.9239, 0.3827, 0.0000, 0.0000],  # r: 45°, p: 0°, y: 0°
            [0.0000, 0.0000, 0.3827, 0.9239],  # r: 45°, p: 0°, y: 180°
            [0.099, -0.2391, 0.3696, 0.8924],  # r: 45°, p: 30°, y: 180°
        ],
        dtype=torch.float,
    )

    rotation_matrices = quaternion_to_matrix(quats)
    yaw_vals = yaw_from_quaternion(quats)

    print(yaw_vals)

    yaw_tensor = torch.tensor([90.0], dtype=torch.float)
    yaw_tensor = torch.deg2rad(yaw_tensor)

    quat = yaw_to_quaternion(yaw_tensor)
    print(quat)
