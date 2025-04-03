"""
Utility functions for logging CPU and memory usage for the system and
current process.
"""

import logging
import os

import psutil

# Retrieve main process logger
logger = logging.getLogger(__name__)


def indent_log(msg: str, level: int = 0, indent_size: int = 4) -> str:
    indent = " " * (level * indent_size)
    return f"{indent}{msg}"


def log_system_resources(
    indent_level: int = 0, one_line: bool = False
) -> None:
    """
    Log current CPU and memory usage for the system and current process.

    Note:
        Logs resource usage information via a logger retrieved from the main
        process using `logging.getLogger(__name__)`.

    Args:
        indent_level (int): The number of indent levels to apply to each log
            line (4 spaces per level). Defaults to 0.
        one_line (bool): If `True`, logs all information in a single line.
            Defaults to `False`.
    """
    # Retrieve system usage information
    vm = psutil.virtual_memory()
    total_memory = vm.total / (1024 * 1024)
    sys_mem_use_perc = vm.percent

    process = psutil.Process(os.getpid())
    cpu_usage = psutil.cpu_percent(interval=1)
    proc_mem_use = process.memory_info().rss / (1024 * 1024)
    proc_mem_use_perc = (proc_mem_use / total_memory) * 100

    avail_mem = vm.available / (1024 * 1024)
    avail_mem_perc = (avail_mem / total_memory) * 100

    # Create local function for logging indented messages
    def log(msg: str) -> None:
        logger.info(indent_log(msg, level=indent_level))

    # Log system resource usage
    if one_line:
        log(
            f"System Resources | CPU: {cpu_usage:.2f}% RAM: {proc_mem_use:.2f}"
            f"MB ({proc_mem_use_perc:.2f}%) TotalRAM: {total_memory:.2f}MB ("
            f"{sys_mem_use_perc:.2f}%) AvailableRAM: {avail_mem:.2f}MB ("
            f"{avail_mem_perc:.2f}%)"
        )
    else:
        log("- System Resources -")
        log("-" * 20)
        log(f"Process CPU usage: {cpu_usage:.2f}%")
        log(
            f"Process RAM usage: {proc_mem_use:.2f} MB ("
            f"{proc_mem_use_perc:.2f}%)"
        )
        log(
            f"Total system RAM usage: {total_memory:.2f} MB ("
            f"{sys_mem_use_perc:.2f}%)"
        )
        log(
            f"Available system RAM: {avail_mem:.2f} MB ({avail_mem_perc:.2f}%)"
        )
