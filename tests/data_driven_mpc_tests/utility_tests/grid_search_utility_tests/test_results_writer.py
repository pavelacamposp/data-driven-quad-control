import io
import math
import os
from pathlib import Path
from typing import Any

import pytest

from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.controller_evaluation import (  # noqa: E501
    CtrlEvalStatus,
)
from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.param_grid_search_config import (  # noqa: E501
    DDMPCEvaluationParams,
    DDMPCFixedParams,
    DDMPCInitialDataCollectionParams,
    DDMPCParameterGrid,
)
from data_driven_quad_control.data_driven_mpc.utilities.param_grid_search.results_writer import (  # noqa: E501
    format_result_dict,
    write_dd_mpc_grid_search_params_to_file,
    write_result_section,
    write_results_to_file,
)


def test_write_results_to_file(
    tmp_path: Path,
    test_init_collection_params: DDMPCInitialDataCollectionParams,
    test_fixed_params: DDMPCFixedParams,
    test_eval_params: DDMPCEvaluationParams,
    test_param_grid: DDMPCParameterGrid,
) -> None:
    # Define test parameters
    test_output_dir = str(tmp_path)
    elapsed_time = 123.123
    test_results = {
        CtrlEvalStatus.SUCCESS: [
            {"N": 40, "average_RMSE": 1.5},
            {"N": 35, "average_RMSE": 1.0},
            {"N": 30, "average_RMSE": 2.5},
        ],
        CtrlEvalStatus.FAILURE: [
            {"n_successful_runs": 1, "N": 40, "average_RMSE": 1.5},
            {"n_successful_runs": 1, "N": 35, "average_RMSE": 1.0},
            {"n_successful_runs": 3, "N": 30, "average_RMSE": 2.5},
        ],
    }

    output_path = write_results_to_file(
        output_dir=test_output_dir,
        elapsed_time=elapsed_time,
        init_data_collection_params=test_init_collection_params,
        fixed_params=test_fixed_params,
        eval_params=test_eval_params,
        param_grid=test_param_grid,
        results=test_results,
    )

    assert os.path.exists(test_output_dir)
    assert os.path.isfile(output_path)

    # Read contents from output file
    with open(output_path) as f:
        output = f.read()

    # Verify expected written values from output
    assert "Grid search complete in 0h 2m 3.12s." in output
    assert "n_n_mpc_step: False" in output
    assert "lamb_sigma_s: [1000.0]" in output
    assert "Successful Results (3/6):" in output
    assert "n_successful_runs=1, N=35, average_RMSE=1.0" in output


def test_write_dd_mpc_grid_search_params_to_file(
    test_fixed_params: DDMPCFixedParams,
) -> None:
    # Create test in-memory text file
    f = io.StringIO()

    # Write parameters to file
    write_dd_mpc_grid_search_params_to_file(f, test_fixed_params)

    # Read written values from file
    output = f.getvalue()

    # Verify expected written values from output
    assert "alpha_reg_type: APPROXIMATED" in output
    assert "U:" in output
    assert "- [0.0, 0.0]" in output
    assert "m: 3" in output


@pytest.mark.parametrize("success_result_status", [True, False])
def test_write_result_section(success_result_status: bool) -> None:
    # Create test in-memory text file
    f = io.StringIO()

    # Create test results
    if success_result_status:
        results = [
            {"N": 40, "average_RMSE": 1.5},
            {"N": 35, "average_RMSE": 1.0},
            {"N": 30, "average_RMSE": 2.5},
        ]

        def sort_key(r: dict[str, Any]) -> Any:
            return r["average_RMSE"]

        reverse = False
        title = "Successful Results"
    else:
        results = [
            {"n_successful_runs": 1, "N": 40, "average_RMSE": 1.5},
            {"n_successful_runs": 1, "N": 35, "average_RMSE": 1.0},
            {"n_successful_runs": 3, "N": 30, "average_RMSE": 2.5},
        ]

        def sort_key(r: dict[str, Any]) -> Any:
            return (r["n_successful_runs"], -r["average_RMSE"])

        reverse = True
        title = "Failed Results"

    # Write result section to file
    write_result_section(
        f=f,
        title=title,
        result_list=results,
        total_searches=4,
        sort_key=sort_key,
        reverse=reverse,
    )

    # Read written values from file
    output = f.getvalue()

    # Verify expected written values from output
    assert f"{title} (3/4):" in output
    lines = output.strip().splitlines()

    if success_result_status:
        # Verify sorting by RMSE in ascending order
        assert "average_RMSE=1.0" in lines[1]
        assert "average_RMSE=2.5" in lines[3]
    else:
        # Verify sorting by n_successful_runs (descending) first,
        # and then by average RMSE (ascending)
        assert "n_successful_runs=3" in lines[1]
        assert "average_RMSE=1.5" in lines[3]


def test_write_result_section_empty() -> None:
    # Create test in-memory text file
    f = io.StringIO()

    # Write result section with empty results
    write_result_section(
        f=f,
        title="Empty Results",
        result_list=[],
        total_searches=4,
        sort_key=lambda r: r["average_RMSE"],
        reverse=False,
    )

    # Read written values from file
    output = f.getvalue()

    # Verify expected written value from file
    assert "No results for any parameter combination." in output


def test_format_result_dict_various_types() -> None:
    # Define test result value
    result = {
        "N": 40,
        "average_RMSE": math.nan,
        "failure_reason": "Test failure reason.",
    }

    formatted = format_result_dict(result)

    # Verify expected formatted values
    assert "N=40" in formatted
    assert "average_RMSE=NaN" in formatted
    assert 'failure_reason="Test failure reason."' in formatted
