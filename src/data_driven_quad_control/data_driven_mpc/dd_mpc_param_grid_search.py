"""
Grid Search for Nonlinear Data-Driven MPC Parameters for Drone Position Control

This script implements a parallel grid search for identifying effective
parameter combinations for nonlinear data-driven Model Predictive Control (MPC)
controllers for drone position control in Genesis (via the `HoverEnv`
environment).

Workflow overview:

1. A vectorized drone environment is initialized with `num_processes` drones.

2. A stabilizing controller (`DroneTrackingController`) stabilizes the drones
   at a target position. Once stabilized, the environment state of a single
   drone, referred to as the initial hovering state, is stored for use during
   data collection.

3. The same drone is then used for collecting initial input-output data, which
   is required for creating data-driven MPC controllers. Before each data
   collection process, the drone's state is reset to the initial hovering
   state. The collected data and the resulting drone state (obtained
   immediately after collection) are stored in a cache (data-driven cache) for
   later use. This process is repeated for each `N` value defined in the
   parameter grid.

   For each `N`, data collection is repeated `num_collections_per_N` times
   (as specified in the configuration file) to enable controller evaluations
   with different initial data entries. This allows indirect validation of
   controller robustness based on their performance.

4. Based on the grid search configuration parameters (defined in a YAML
   configuration file), the main process of the grid search spawns multiple
   parallel worker processes. Each worker evaluates multiple unique controller
   parameter combinations by creating a nonlinear data-driven MPC controller
   from each combination and evaluating each controller independently.

   Each controller is evaluated on its ability to command a drone toward a
   series of target positions. This is implemented in a closed-loop simulation
   that sends control actions to the main process and receives observations
   from it at each simulation step. This architecture is necessary because the
   drone environment is vectorized and steps all drones simultaneously.

   If the controller performs poorly, defined as the drone's distance to its
   target increasing by more than a threshold relative to the initial distance,
   the evaluation terminates early and the parameter combination is deemed a
   failure.

   Controller parameter evaluations are executed within isolated subprocesses
   to prevent memory leaks between runs. These leaks are caused by CVXPY
   objects (created during controller creation) not being properly cleaned up,
   allowing them to persist across runs.

   The main process also manages the vectorized environment stepping and
   the drone environment state resets required between each controller
   evaluation run (using the drone states stored in the data-driven cache). It
   communicates with the workers via multiprocessing queues in tight
   synchronization, preventing deadlocks.

   The drone environment state resets are key to ensure that the past
   input-output measurements used in the MPC problem formulation for each
   controller align with the collected data.

5. If a worker has no remaining parameter combinations to evaluate, it
   terminates. This repeats for every worker until all the parameter
   combinations in the grid have been evaluated.

6. At the end of the grid search, a log file is written summarizing the grid
   search results.

The drone environment is configured to use a Collective Thrust and Body Rates
(CTBR) controller internally, with a fixed yaw angular rate of 0 rad/s. This
simplifies the drone control system and sets the number of control inputs to 3:
[total thrust [N], roll angular rate [rad/s], pitch angular rate [rad/s]].

The nonlinear data-driven MPC controller used in the grid search is available
at: https://github.com/pavelacamposp/direct_data_driven_mpc

Parallelism is implemented using `torch.multiprocessing`, ensuring fast,
efficient, and safe parameter searches.
"""

import argparse
import logging
import os
import time
import warnings
from itertools import product

import genesis as gs
import numpy as np
import torch

from data_driven_quad_control.envs.hover_env_config import (
    get_cfgs,
)
from data_driven_quad_control.utilities.drone_environment import (
    create_env,
    get_current_env_state,
)
from data_driven_quad_control.utilities.drone_tracking_controller import (
    create_drone_tracking_controller,
    hover_at_target,
)

from .utilities.param_grid_search.grid_search_param_loader import (
    load_dd_mpc_grid_search_params,
)
from .utilities.param_grid_search.initial_data_collection_cache import (
    cache_initial_data_and_states,
)
from .utilities.param_grid_search.parallel_grid_search import (
    parallel_grid_search,
)
from .utilities.param_grid_search.param_grid_search_config import (
    DDMPCCombinationParams,
)
from .utilities.param_grid_search.results_writer import write_results_to_file

# Suppress all warnings from CVXPY to maintain clean terminal output
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")

# Directory for storing Data-Driven MPC parameter grid search results
GRID_SEARCH_RESULTS_DIR = "logs/dd_mpc_grid_search"
os.makedirs(GRID_SEARCH_RESULTS_DIR, exist_ok=True)

# Configure main process logger
LOG_FILENAME = os.path.join(
    GRID_SEARCH_RESULTS_DIR, "parallel_grid_search.log"
)
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

# Data-Driven MPC Grid Search configuration file
DEFAULT_DD_MPC_GRID_SEARCH_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../configs/data_driven_mpc/dd_mpc_grid_search_params.yaml",
)


def disable_debug_logging() -> None:
    """Disable logging from child loggers."""
    logger.info(
        "Debug logging is disabled. Run grid search with the `--debug` "
        "argument to enable it."
    )

    for handler in logger.handlers:
        handler.setLevel(logging.CRITICAL + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid Search for Nonlinear Data-Driven MPC Parameters "
        "for Drone Position Control",
    )

    parser.add_argument(
        "--num_processes",
        type=int,
        default=10,
        help="The number of processes used for parallelization.",
    )
    parser.add_argument(
        "--grid_search_config_path",
        type=str,
        default=DEFAULT_DD_MPC_GRID_SEARCH_CONFIG_PATH,
        help="The path to the YAML configuration file containing the "
        "parameters for the nonlinear data-driven MPC parameter grid search.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable the GUI to visualize the simulation during the grid "
        "search.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The Random Number Generator seed for reproducibility. Defaults "
        "to `None`.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="The verbosity level: 0 = no output, 1 = minimal output, 2 = "
        "detailed output.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debugging logging.",
    )

    return parser.parse_args()


def main() -> None:
    program_start_time = time.time()

    # Parse arguments
    args = parse_args()
    num_processes = args.num_processes
    grid_search_config_path = args.grid_search_config_path
    gui = args.gui
    seed = args.seed
    verbose = args.verbose
    debug = args.debug

    # Disable logger if not in debug mode
    if not debug:
        disable_debug_logging()

    if verbose:
        print(
            "--- Parallel Grid Search for Nonlinear Data-Driven MPC "
            "Controller Parameters ---"
        )
        print("-" * 80)

        if debug:
            print("Debug logging enabled")
            print(f"  Logging debug output to: {LOG_FILENAME}")

    logger.info("[DD-MPC-GS] Started parallel grid search execution")

    # Grid Search parameters
    if verbose:
        print(f"Number of parallel processes: {num_processes}\n")

    logger.info(f"[DD-MPC-GS] Number of parallel processes: {num_processes}")

    # Create a Random Number Generator for reproducibility
    np_random = np.random.default_rng(seed=seed)

    # Initialize Genesis simulator
    if verbose:
        print("Initializing Genesis simulator")

        if verbose > 1 and seed is not None:
            print(f"  RNG seed: {seed}")

    logger.info("[DD-MPC-GS] Initializing Genesis simulator")

    gs.init(seed=seed, backend=gs.gpu, logging_level="error")

    # Load environment configuration
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    # Enable observation noise
    obs_cfg["obs_noise_std"] = 1e-4  # Observation noise std (normalized)
    # Note:
    # Observation noise is added to normalized observations. For drone position
    # observations, an std of 1e-4 corresponds to a physical position error std
    # of 1e-4 / obs_cfg["obs_scales"]["rel_pos"] = 1e-4 / (1 / 3) = 0.3 mm
    # after denormalization and rescaling (since x_norm = x * scale).

    # Increase episode length and spatial bounds to allow sufficient
    # time and space for the data-driven MPC parameter grid search
    env_cfg["episode_length_s"] = 100000
    env_cfg["termination_if_close_to_ground"] = 0.0
    env_cfg["termination_if_x_greater_than"] = 100.0
    env_cfg["termination_if_y_greater_than"] = 100.0
    env_cfg["termination_if_z_greater_than"] = 10.0

    # Create vectorized environment
    if verbose:
        print(f"Creating vectorized environment with {num_processes} drones")

    logger.info("[DD-MPC-GS] Creating drone environment")

    num_envs = num_processes
    show_viewer = gui
    env = create_env(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
    )

    # Reset env
    env.reset()

    # Load parameters for the data-driven MPC controller parameter grid search
    if verbose:
        print("Loading grid search parameters from configuration file")

    logger.info("[DD-MPC-GS] Loading grid search parameters from config file")

    m = env.num_actions  # Number of inputs
    p = 3  # Number of outputs (drone position)
    init_data_collection_params, fixed_params, eval_params, param_grid = (
        load_dd_mpc_grid_search_params(
            m=m, p=p, config_path=grid_search_config_path, verbose=verbose
        )
    )

    # Initial input-output data collection
    if verbose:
        print("\nInitial Input-Output Data Collection")
        print("-" * 36)

    # Select an env idx from the vectorized environment for obtaining
    # initial input-output data and initial drone states
    base_env_idx = 0

    # Create a controller to stabilize the drone at a specific
    # position for initial input-output data collection
    stabilizing_controller = create_drone_tracking_controller(env=env)

    # Define hover target
    target_pos = torch.tensor(
        init_data_collection_params.init_hover_pos,
        device=env.device,
        dtype=torch.float,
    )
    target_yaw = torch.tensor([0.0], device=env.device, dtype=torch.float)

    # Command drones to hover at target
    if verbose:
        print(f"Hovering drones at target {target_pos.tolist()}")

    logger.info(f"[DD-MPC-GS] Hovering drones at target {target_pos.tolist()}")

    target_pos = target_pos.expand(env.num_envs, -1)
    target_yaw = target_yaw.expand(env.num_envs)
    hover_at_target(
        env=env,
        tracking_controller=stabilizing_controller,
        target_pos=target_pos,
        target_yaw=target_yaw,
        ctbr_controller=None,
    )

    # Save hovering drone state
    if verbose:
        print("Saving hovering drone state for resets during data collection")

    logger.info("[DD-MPC-GS] Saving hovering drone state for resets")

    init_hovering_state = get_current_env_state(env=env, env_idx=base_env_idx)

    # Create data-driven cache:
    # Collect initial input-output data and save drone env states
    if verbose:
        print("Starting data-driven cache creation")

    logger.info("[DD-MPC-GS] Starting data-driven cache creation")

    data_driven_cache, drone_state_cache = cache_initial_data_and_states(
        env=env,
        base_env_idx=base_env_idx,
        stabilizing_controller=stabilizing_controller,
        target_pos=target_pos,
        target_yaw=target_yaw,
        init_hovering_state=init_hovering_state,
        init_data_collection_params=init_data_collection_params,
        fixed_params=fixed_params,
        eval_params=eval_params,
        param_grid=param_grid,
        verbose=verbose,
        np_random=np_random,
    )

    if verbose:
        print(
            "Data-driven cache created from collected data and saved "
            "drone states"
        )

    logger.info("[DD-MPC-GS] Data-driven cache created")

    # Perform grid search in parallel
    if verbose:
        print("\nNonlinear Data-Driven MPC Parameter Grid Search")
        print("-" * 47)

    parameter_combinations = [
        DDMPCCombinationParams(*combination)
        for combination in product(*param_grid)
    ]

    if verbose:
        total_combinations = len(parameter_combinations)
        num_eval_runs_per_comb = (
            len(eval_params.eval_setpoints) * eval_params.num_collections_per_N
        )
        total_eval_runs = total_combinations * num_eval_runs_per_comb

        print(
            f"Starting parameter grid search with {total_combinations} "
            "combinations"
        )
        print(
            f"  Total number of evaluation runs: {total_eval_runs} ("
            f"{num_eval_runs_per_comb} runs per combination)"
        )

        if verbose > 1:
            ext_out_incr_in = fixed_params.ext_out_incr_in
            alpha_reg_type = fixed_params.alpha_reg_type
            n_n_mpc_step = fixed_params.n_n_mpc_step

            print("  Grid Search conducted over the following parameters:")
            for key, values in param_grid._asdict().items():
                print(f"    {key}: {values}")
            print(f"  Extended output and input increments: {ext_out_incr_in}")
            print(f"  Alpha regularization type: {alpha_reg_type.name}")
            print(f"  n-step Data-Driven MPC: {n_n_mpc_step}")

    logger.info(
        f"Starting parameter grid search with {len(parameter_combinations)} "
        f"combinations - {total_eval_runs} total evaluation runs"
    )

    try:
        with torch.no_grad():
            results = parallel_grid_search(
                env=env,
                num_processes=num_processes,
                parameter_combinations=parameter_combinations,
                fixed_params=fixed_params,
                eval_params=eval_params,
                data_driven_cache=data_driven_cache,
                drone_state_cache=drone_state_cache,
            )

        # Write results to a file
        output_dir = GRID_SEARCH_RESULTS_DIR
        elapsed_time = time.time() - program_start_time
        output_file = write_results_to_file(
            output_dir=output_dir,
            elapsed_time=elapsed_time,
            num_processes=num_processes,
            init_data_collection_params=init_data_collection_params,
            fixed_params=fixed_params,
            eval_params=eval_params,
            param_grid=param_grid,
            results=results,
        )

        if verbose:
            print(f"\nGrid search results written to {output_file}.")

        logger.info(f"Grid search results written to {output_file}.")

    except Exception as e:
        print(f"Parallel grid search failed with error: {str(e)}")
        logger.exception("Fatal error in main()")

    finally:
        print("Parallel grid search finished.")
        logger.info("Parallel grid search finished.")


if __name__ == "__main__":
    main()
