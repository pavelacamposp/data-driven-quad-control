import pytest

from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.isolated_execution import (  # noqa: E501
    run_in_isolated_process,
)


def test_run_in_isolated_process_success() -> None:
    # Define test function to verify normal execution
    def add(a: int, b: int) -> int:
        return a + b

    result = run_in_isolated_process(add, 2, 3)

    # Verify that the function executed correctly in an isolated process
    assert result == 5


def test_run_in_isolated_process_exception() -> None:
    # Define a test function that raises an exception to simulate failure
    def fail(c: int) -> None:
        raise ValueError("Raised exception in internal function")

    # Verify that the exception is raised and is correctly propagated and
    # wrapped in a `ValueError` through the isolated process
    with pytest.raises(
        ValueError, match="Raised exception in internal function"
    ):
        run_in_isolated_process(fail, 5)
