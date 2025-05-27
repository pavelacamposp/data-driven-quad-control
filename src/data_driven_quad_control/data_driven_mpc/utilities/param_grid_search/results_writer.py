"""
File writing utilities for data-driven MPC grid search results

This module provides functionality for writing parameter configurations and
controller evaluation results to a file after a data-driven MPC parameter
grid search. It supports structured output for fixed parameters, parameter
grids, and evaluation metrics, and formats results by status (success or
failure) to facilitate scanning.
"""

import math
import os
from datetime import datetime
from typing import Any, Callable, NamedTuple, TextIO

from .param_grid_search_config import (
    CtrlEvalStatus,
    DDMPCEvaluationParams,
    DDMPCFixedParams,
    DDMPCInitialDataCollectionParams,
    DDMPCParameterGrid,
)


def write_results_to_file(
    output_dir: str,
    elapsed_time: float,
    num_processes: int,
    init_data_collection_params: DDMPCInitialDataCollectionParams,
    fixed_params: DDMPCFixedParams,
    eval_params: DDMPCEvaluationParams,
    param_grid: DDMPCParameterGrid,
    results: dict[CtrlEvalStatus, list[dict[str, Any]]],
    file_name: str | None = None,
) -> str:
    """
    Write a complete report of the results and configurations of a data-driven
    MPC grid search to a file.

    Args:
        output_dir (str): The directory to save the summary report.
        elapsed_time (float): The grid search duration in seconds.
        num_processes (int): The number of parallel worker processes used
            during the grid search.
        fixed_params (DDMPCFixedParams): The fixed parameters used in the grid
            search.
        init_data_collection_params (DDMPCInitialDataCollectionParams): The
            parameters for collecting initial input-output data.
        eval_params (DDMPCEvaluationParams): The parameters that define the
            evaluation procedure for each controller parameter combination in
            the grid search.
        param_grid (DDMPCParameterGrid): The parameter grid used in the grid
            search.
        results (dict[CtrlEvalStatus, list[dict[str, Any]]]): A dictionary
            mapping evaluation statuses to lists of result dictionaries, each
            containing evaluation metrics and context.
        file_name (str | None): The name of the output file. If `None`, a
            timestamped file name will be used.

    Returns:
        str: The full path to the saved report file.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now()

    # Set a default name if file name is not provided
    if file_name is None:
        name_timestamp = timestamp.strftime("%Y%m%d_%H%M%S")
        file_name = f"grid_search_results_{name_timestamp}.txt"

    output_file = os.path.join(output_dir, file_name)

    with open(output_file, "w") as f:
        report_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Report generated at: {report_timestamp}\n\n")

        # Write grid search elapsed time
        hours, remaining = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remaining, 60)
        f.write(
            f"Grid search duration: {int(hours)}h {int(minutes)}m "
            f"{seconds:.2f}s\n"
        )

        # Write number of parallel processes used in the grid search
        f.write(f"Number of parallel processes: {num_processes}\n")
        f.write("\n")

        # Write initial data collection parameter summary
        f.write("Initial data collection parameters:\n")
        write_dd_mpc_grid_search_params_to_file(f, init_data_collection_params)
        f.write("\n")

        # Write fixed parameter summary
        f.write("Fixed parameters:\n")
        write_dd_mpc_grid_search_params_to_file(f, fixed_params)
        f.write("\n")

        # Write evaluation parameter summary
        f.write("Evaluation parameters:\n")
        write_dd_mpc_grid_search_params_to_file(f, eval_params)
        f.write("\n")

        # Write parameter grid summary
        f.write("Grid Search conducted over the following parameters:\n")
        write_dd_mpc_grid_search_params_to_file(f, param_grid)
        f.write("\n")

        # Write separate sections for successful and failed evaluation results
        total_searches = sum(len(v) for v in results.values())

        # Sort successful results by average RMSE (ascending)
        write_result_section(
            f=f,
            title="Successful Results",
            result_list=results.get(CtrlEvalStatus.SUCCESS, []),
            total_searches=total_searches,
            sort_key=lambda r: r["average_RMSE"],
        )
        f.write("\n")

        # Sort failed results by number of successful runs (descending),
        # and then by average RMSE (ascending)
        write_result_section(
            f=f,
            title="Failed Results",
            result_list=results.get(CtrlEvalStatus.FAILURE, []),
            total_searches=total_searches,
            sort_key=lambda r: (r["n_successful_runs"], -r["average_RMSE"]),
            reverse=True,
        )

    return output_file


def write_dd_mpc_grid_search_params_to_file(
    f: TextIO, dd_mpc_params: NamedTuple
) -> None:
    """
    Write the contents of a data-driven MPC search parameter (`NamedTuple`) to
    a file in a readable, structured format.

    Args:
        f (TextIO): The file object to write to.
        dd_mpc_params (NamedTuple): The `NamedTuple` containing data-driven MPC
            grid search parameters.
    """
    for key, value in dd_mpc_params._asdict().items():
        # Write `alpha_reg_type` enum name
        if key == "alpha_reg_type":
            f.write(f"  {key}: {value.name}\n")

        # Write arrays in a structured format
        elif key in ("u_range", "U", "Us", "eval_setpoints"):
            f.write(f"  {key}:\n")
            for arr in value:
                arr_list = arr.flatten().tolist()
                f.write(f"    - {arr_list}\n")

        # Write other parameters
        else:
            f.write(f"  {key}: {value}\n")


def write_result_section(
    f: TextIO,
    title: str,
    result_list: list[dict[str, Any]],
    total_searches: int,
    sort_key: Callable[[dict[str, Any]], Any],
    reverse: bool = False,
) -> None:
    """
    Write a formatted section of sorted evaluation results to a file.

    Args:
        f (TextIO): The file object to write to.
        title (str): The result section title (e.g., "Successful Results").
        result_list (list[dict[str, Any]]): A list of result dictionaries.
        total_searches (int): The total number of evaluated parameter
            combinations.
        sort_key (Callable): The key function for sorting the result list.
        reverse (bool): Whether to sort the result list in descending order.
            Defaults to `False`.
    """
    sorted_results = sorted(result_list, key=sort_key, reverse=reverse)

    f.write(f"{title} ({len(sorted_results)}/{total_searches}):\n")

    if not sorted_results:
        f.write("  No results for any parameter combination.\n")
    else:
        for result in sorted_results:
            f.write(f"  {format_result_dict(result)}\n")


def format_result_dict(result: dict[str, Any]) -> str:
    """
    Format a result dictionary as a readable string.

    Args:
        result (dict[str, Any]): A dictionary containing evaluation results.

    Returns:
        str: A string-formatted version of the result.
    """
    formatted = {}
    for key, value in result.items():
        if isinstance(value, float) and math.isnan(value):
            formatted[key] = "NaN"
        elif isinstance(value, str):
            formatted[key] = f'"{value}"'
        else:
            formatted[key] = value

    return ", ".join(f"{key}={value}" for key, value in formatted.items())
