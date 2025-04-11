import torch

from src.utilities.math_utils import gs_rand_float, linear_interpolate


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

    assert torch.allclose(result, expected)


def test_linear_interpolate_tensor_bounds() -> None:
    x = torch.tensor([0.25, 2.5])
    x_min = torch.tensor([0.0, 0.0])
    x_max = torch.tensor([1.0, 5.0])
    y_min = torch.tensor([100.0, 200.0])
    y_max = torch.tensor([200.0, 500.0])

    result = linear_interpolate(x, x_min, x_max, y_min, y_max)
    expected = torch.tensor([125.0, 350.0])

    assert torch.allclose(result, expected)
