"""
Isolated execution of functions within a subprocess

This module provides a wrapper that enables the execution of functions within
an internal subprocess, ensuring isolation and correct exception propagation.

This isolation is especially important for preventing memory leaks during
parallel grid searches, as they repeatedly create CVXPY objects when creating
data-driven MPC controllers. Executing controller evaluations in isolation
ensures CVXPY objects are cleaned up after each evaluation. In contrast,
executing evaluations directly within the parallel grid search causes CVXPY
objects to persist across runs.
"""

from typing import Any, Callable

import torch.multiprocessing as mp


def run_in_isolated_process(
    target_func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a function in an isolated subprocess and return the result."""

    def _wrapper(queue: mp.Queue, *args: Any, **kwargs: Any) -> None:
        try:
            result = target_func(*args, **kwargs)
            queue.put(("ok", result))
        except Exception as e:
            queue.put(("error", str(e)))

    queue: mp.Queue = mp.Queue()
    p = mp.Process(target=_wrapper, args=(queue, *args), kwargs=kwargs)
    p.start()
    p.join()

    if not queue.empty():
        status, payload = queue.get()
        if status == "ok":
            return payload
        else:
            raise ValueError(payload)
    else:
        raise RuntimeError("Isolated subprocess failed with no output.")
