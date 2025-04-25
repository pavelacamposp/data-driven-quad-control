import torch
from torch.testing import assert_close

from data_driven_quad_control.utilities.math_utils import (
    gs_rand_float,
    linear_interpolate,
    quaternion_to_matrix,
    yaw_from_quaternion,
    yaw_to_quaternion,
)


def test_gs_rand_float() -> None:
    lower = 2.0
    upper = 5.0
    shape = (3, 2)
    device = torch.device("cpu")

    result = gs_rand_float(lower, upper, shape, device)

    assert result.shape == shape
    assert result.device == device
    assert torch.all(result >= lower)
    assert torch.all(result < upper)


def test_linear_interpolate_float_bounds() -> None:
    x = torch.tensor([0.0, 0.5, 1.0])
    x_min = 0.0
    x_max = 1.0
    y_min = 10.0
    y_max = 20.0

    result = linear_interpolate(x, x_min, x_max, y_min, y_max)
    expected = torch.tensor([10.0, 15.0, 20.0])

    assert_close(result, expected)


def test_linear_interpolate_tensor_bounds() -> None:
    x = torch.tensor([0.25, 2.5])
    x_min = torch.tensor([0.0, 0.0])
    x_max = torch.tensor([1.0, 5.0])
    y_min = torch.tensor([100.0, 200.0])
    y_max = torch.tensor([200.0, 500.0])

    result = linear_interpolate(x, x_min, x_max, y_min, y_max)
    expected = torch.tensor([125.0, 350.0])

    assert_close(result, expected)


def test_quaternion_to_matrix_known_yaw() -> None:
    # 90° yaw only quaternion
    yaw_deg = torch.tensor([0.0, 90.0])
    yaw_rad = torch.deg2rad(yaw_deg)

    quat = yaw_to_quaternion(yaw_rad)

    expected_matrix = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    rot_matrix = quaternion_to_matrix(quat)

    assert rot_matrix.shape == (2, 3, 3)
    assert_close(rot_matrix, expected_matrix)


def test_yaw_from_quaternion() -> None:
    quats = torch.tensor(
        [
            [0.7071, 0.0000, 0.0000, 0.7071],  # r: 0°, p: 0°, y: 90°
            [0.9239, 0.3827, 0.0000, 0.0000],  # r: 45°, p: 0°, y: 0°
            [0.0000, 0.0000, 0.3827, 0.9239],  # r: 45°, p: 0°, y: 180°
            [0.0990, -0.2391, 0.3696, 0.8924],  # r: 45°, p: 30°, y: 180°
        ],
        dtype=torch.float,
    )

    expected_yaws_deg = torch.tensor([90.0, 0.0, 180.0, 180.0])
    expected_yaws_rad = torch.deg2rad(expected_yaws_deg)

    computed_yaws = yaw_from_quaternion(quats)

    assert computed_yaws.shape == (4,)

    # Compare sin/cos to avoid angle wraparound issues (i.e., 180° != -180°)
    assert_close(
        torch.sin(computed_yaws),
        torch.sin(expected_yaws_rad),
        rtol=0.0,
        atol=1e-4,
    )
    assert_close(
        torch.cos(computed_yaws),
        torch.cos(expected_yaws_rad),
        rtol=0.0,
        atol=1e-4,
    )


def test_yaw_to_quaternion() -> None:
    yaw_deg = torch.tensor([0.0, 90.0])
    yaw_rad = torch.deg2rad(yaw_deg)

    quat = yaw_to_quaternion(yaw_rad)

    expected = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.7071, 0.0, 0.0, 0.7071]], dtype=torch.float
    )

    assert quat.shape == (2, 4)
    assert_close(quat, expected, rtol=0.0, atol=1e-4)
