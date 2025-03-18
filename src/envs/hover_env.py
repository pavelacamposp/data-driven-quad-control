# Drone Environment based on the `hover_env.py` example 
# from the Genesis repository.
#
# Repository: Genesis-Embodied-AI/Genesis
# URL: https://github.com/Genesis-Embodied-AI/Genesis
# License: Apache License 2.0 (See `LICENSE-APACHE` for details)
#
# Key modifications:
#   - Implemented action decimation (number of simulation steps
#     to take for each task step).
#   - Implemented methods for saving and loading environment states.
#   - Added parameter for disabling automatic target position (command)
#     updates.

from typing import Any, Union

import torch
import math
import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat)

from src.utilities.math_utils import gs_rand_float

from genesis.engine.entities.drone_entity import DroneEntity
from genesis.engine.entities.rigid_entity import RigidEntity

class HoverEnv:
    BASE_RPM = 14468.429183500699  # Propeller RPMs for hovering
    def __init__(
        self,
        num_envs: int,
        env_cfg: dict[str, Any],
        obs_cfg: dict[str, Any],
        reward_cfg: dict[str, Any],
        command_cfg: dict[str, Any],
        show_viewer: bool = False,
        device: str = "cuda",
        auto_target_updates: bool = True):
        self._is_closed = False  # Env closing status
        
        self.device = torch.device(device)

        self.dt = env_cfg["dt"]
        self.decimation = env_cfg["decimation"]
        self.step_dt = self.dt * self.decimation

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.step_dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=1),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=self.num_envs),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add target
        if self.env_cfg["visualize_target"]:
            self.target: RigidEntity = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.04,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.5, 0.5),
                    ),
                ),
            )
        else:
            self.target = None

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # add drone
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"],
                                          device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"],
                                           device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.drone: DroneEntity = self.scene.add_entity(
            gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

        # build scene
        self.scene.build(n_envs=num_envs, env_spacing=(0.4, 0.4))

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.step_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,),
                                                  device=self.device,
                                                  dtype=gs.tc_float)

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs),
                                   device=self.device,
                                   dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,),
                                   device=self.device,
                                   dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,),
                                    device=self.device,
                                    dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,),
                                              device=self.device,
                                              dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands),
                                    device=self.device,
                                    dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions),
                                   device=self.device,
                                   dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3),
                                    device=self.device,
                                    dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4),
                                     device=self.device,
                                     dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3),
                                        device=self.device,
                                        dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3),
                                        device=self.device,
                                        dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        self.extras = dict()  # extra information for logging

        # Enable automatic target updates (commands)
        # If True, target positions are updated when reached.
        # If False, they must be manually changed.
        self.auto_target_updates = auto_target_updates

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(
            *self.command_cfg["pos_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(
            *self.command_cfg["pos_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(
            *self.command_cfg["pos_z_range"], (len(envs_idx),), self.device)

        if self.target is not None:
            self.target.set_pos(self.commands[envs_idx],
                                zero_velocity=True,
                                envs_idx=envs_idx)

    def _at_target(self) -> torch.Tensor:
        at_target = (torch.norm(self.rel_pos, dim=1) <
                     self.env_cfg["at_target_threshold"])
        at_target = at_target.nonzero(as_tuple=False).flatten()

        return at_target
    
    def reset(self) -> tuple[torch.Tensor, None]:
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    def step(
        self,
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor, dict[str, Any]]:
        self.actions = torch.clip(actions,
                                  -self.env_cfg["clip_actions"],
                                  self.env_cfg["clip_actions"])
        
        exec_actions = (self.last_actions
                        if self.simulate_action_latency
                        else self.actions)

        # perform physics stepping
        for _ in range(self.decimation):
            self.drone.set_propellels_rpm(
                (1 + exec_actions * 0.8) * self.BASE_RPM)
            self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat
            )
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(),
                                                 inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(),
                                                 inv_base_quat)

        # resample commands automatically if enabled
        if self.auto_target_updates:
            envs_idx = self._at_target()
            self._resample_commands(envs_idx)

        # check termination and reset
        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) >
             self.env_cfg["termination_if_pitch_greater_than"]) |
            (torch.abs(self.base_euler[:, 0]) >
             self.env_cfg["termination_if_roll_greater_than"]) |
            (torch.abs(self.rel_pos[:, 0]) >
             self.env_cfg["termination_if_x_greater_than"]) |
            (torch.abs(self.rel_pos[:, 1]) >
             self.env_cfg["termination_if_y_greater_than"]) |
            (torch.abs(self.rel_pos[:, 2]) >
             self.env_cfg["termination_if_z_greater_than"]) |
            (self.base_pos[:, 2] <
             self.env_cfg["termination_if_close_to_ground"])
        )
        self.reset_buf = (
            (self.episode_length_buf > self.max_episode_length) |
            self.crash_condition
        )

        time_out_idx = (self.episode_length_buf > self.max_episode_length)
        time_out_idx = time_out_idx.nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf,
                                                    device=self.device,
                                                    dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"],
                           -1, 1),
                self.base_quat,
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"],
                           -1, 1),
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"],
                           -1, 1),
                self.last_actions,
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]

        # Return obs, privileged_obs, rewards, dones, infos
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras
    
    def close(self) -> None:
        if not self._is_closed:
            del self.scene

            # update closing status
            self._is_closed = True

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(self.base_pos[envs_idx],
                           zero_velocity=True,
                           envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx],
                            zero_velocity=True,
                            envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() /
                self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)
    
    def get_observations(self) -> torch.Tensor:
        return self.obs_buf
    
    # ------------ target position update ------------
    def update_target_pos(
        self,
        envs_idx: torch.Tensor,
        target_pos: torch.Tensor
    ) -> None:
        self.commands[envs_idx, :] = target_pos

        if self.target is not None:
            self.target.set_pos(self.commands[envs_idx],
                                zero_velocity=True,
                                envs_idx=[envs_idx])
    
    # ------------ save/load state ------------
    def get_current_state(
        self,
        envs_idx: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        state = {
            "base_pos": self.base_pos[envs_idx].clone(),
            "base_quat": self.base_quat[envs_idx].clone(),
            "base_lin_vel": self.base_lin_vel[envs_idx].clone(),
            "base_ang_vel": self.base_ang_vel[envs_idx].clone(),
            "commands": self.commands[envs_idx].clone(),
            "episode_length": self.episode_length_buf[envs_idx].clone(),
            "last_actions": self.last_actions[envs_idx].clone(),
        }
        
        return state
    
    def restore_from_state(
        self,
        envs_idx: torch.Tensor,
        saved_state: dict[str, torch.Tensor]
    ) -> None:
        # Retrieve variables from saved state
        base_pos = saved_state['base_pos']
        base_quat = saved_state['base_quat']
        base_lin_vel = saved_state['base_lin_vel']
        base_ang_vel = saved_state['base_ang_vel']
        commands = saved_state['commands']
        episode_length = saved_state['episode_length']
        last_actions = saved_state['last_actions']

        # Reset base
        self.base_pos[envs_idx] = base_pos
        self.last_base_pos[envs_idx] = base_pos
        self.rel_pos[envs_idx] = commands - base_pos
        self.last_rel_pos[envs_idx] = commands - base_pos
        self.base_quat[envs_idx] = base_quat.reshape(1, -1)

        # Apply position and orientation
        self.drone.set_pos(self.base_pos[envs_idx],
                           zero_velocity=False,
                           envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx],
                            zero_velocity=False,
                            envs_idx=envs_idx)
        
        # Apply linear and angular velocities
        self.base_lin_vel[envs_idx] = base_lin_vel
        self.base_ang_vel[envs_idx] = base_ang_vel

        qvel = torch.cat([self.base_lin_vel[envs_idx],
                          self.base_ang_vel[envs_idx]],
                         dim=-1)
        self.drone.set_dofs_velocity(qvel, envs_idx=[envs_idx])

        # Reset buffers
        self.last_actions[envs_idx] = last_actions
        self.episode_length_buf[envs_idx] = episode_length
        self.reset_buf[envs_idx] = True

        # Fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() /
                self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0


    # ------------ reward functions----------------
    def _reward_target(self) -> torch.Tensor:
        target_rew = (torch.sum(torch.square(self.last_rel_pos), dim=1) -
                      torch.sum(torch.square(self.rel_pos), dim=1))
        return target_rew

    def _reward_smooth(self) -> torch.Tensor:
        smooth_rew = torch.sum(
            torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_yaw(self) -> torch.Tensor:
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159  # (rad)
        yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))
        return yaw_rew

    def _reward_angular(self) -> torch.Tensor:
        angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=1)
        return angular_rew

    def _reward_crash(self) -> torch.Tensor:
        crash_rew = torch.zeros((self.num_envs,),
                                device=self.device,
                                dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew
