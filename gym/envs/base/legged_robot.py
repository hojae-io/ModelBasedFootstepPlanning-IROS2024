# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from gym.envs.base.base_task import BaseTask
from gym.utils.terrain import Terrain
from gym.utils.math import *
from gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from gym.utils import KeyboardInterface

class LeggedRobot(BaseTask):
    
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg()
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True


    def step(self):
        """ Apply actuations, simulate, call self.post_physics_step()

        Args:
            actuations (torch.Tensor): Tensor of shape (num_envs, num_actuations_per_env)
        """
        self.reset_buffers()
        self.pre_physics_step()
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques()
            
            if self.cfg.asset.disable_motors:
                self.torques[:] = 0.

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

    def pre_physics_step(self):
        pass


    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_heightmap_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback() 

        self.check_termination()
        
        nact = self.num_actuators
        self.actuation_history[:, 2*nact:] = self.actuation_history[:, nact:2*nact]
        self.actuation_history[:, nact:2*nact] = self.actuation_history[:, :nact]
        self.actuation_history[:, :nact] = self.dof_pos_target*self.cfg.control.actuation_scale
        self.dof_vel_history[:, nact:] = self.dof_vel_history[:, :nact]
        self.dof_vel_history[:, :nact] = self.dof_vel

        if not self.headless:
            self._visualization()


    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.terminated = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.timed_out = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf = self.terminated | self.timed_out


    def reset_envs(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        # if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
        if self.cfg.commands.curriculum:
            self._update_command_curriculum(env_ids)
        if self.cfg.rewards.curriculum:
            self._update_reward_curriculum(env_ids)

        # reset robot states
        self._reset_system(env_ids)
        self._resample_commands(env_ids)
        # reset buffers
        self.actuation_history[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
            

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, 
                                       self.graphics_device_id, 
                                       self.physics_engine, 
                                       self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()


    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        self.gym.set_camera_location(self.camera_handle, self.envs[0], cam_pos, cam_target)


    #------------- Callbacks --------------
    def _setup_keyboard_interface(self):
        self.keyboard_interface = KeyboardInterface(self)

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props


    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r \
                                           *self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props


    def _process_rigid_body_props(self, props, env_id):
        if env_id==0:
            m = 0
            for i, p in enumerate(props):
                m += p.mass
                self.rigid_body_mass[i] = p.mass
            #     print(f"Mass of body {i}: {p.mass} (before randomization)")
            # print(f"Total mass {m} (before randomization)")
            self.mass_total = m
            
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props


    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        if self.cfg.commands.resampling_time == -1 :
            pass  # when the joystick is used, the self.commands variables are overridden
        elif self.cfg.commands.resampling_time == 0:
            self._resample_commands(env_ids)
        else:
            env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
            self._resample_commands(env_ids)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()


    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], 
                                                     self.command_ranges["lin_vel_x"][1], 
                                                     (len(env_ids), 1), 
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(-self.command_ranges["lin_vel_y"], 
                                                     self.command_ranges["lin_vel_y"], 
                                                     (len(env_ids), 1), 
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(-self.command_ranges["yaw_vel"], 
                                                     self.command_ranges["yaw_vel"], 
                                                     (len(env_ids), 1), 
                                                     device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        self.commands[env_ids, 2] *= (torch.abs(self.commands[env_ids, 2]) > 0.2)


    def _compute_torques(self):
        """ Compute torques from actuations.
            actuations can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actuations (torch.Tensor): Actuations

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        if self.cfg.control.exp_avg_decay:
            self.dof_pos_avg = exp_avg_filter(self.dof_pos_target, self.dof_pos_avg,
                                              self.cfg.control.exp_avg_decay)
            dof_pos_target = self.dof_pos_avg
        else:
            dof_pos_target = self.dof_pos_target

        # pd controller
        torques = self.p_gains * (dof_pos_target * self.cfg.control.actuation_scale \
                                + self.default_dof_pos \
                                - self.dof_pos) \
                    - self.d_gains * self.dof_vel

        return torch.clip(torques, -self.torque_limits, self.torque_limits)


    def _reset_system(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environment ids

        # todo: make separate methods for each reset type, cycle through `reset_mode` and call appropriate method. That way the base ones can be implemented once in legged_robot.
        """

        if hasattr(self, self.cfg.init_state.reset_mode):
            eval(f"self.{self.cfg.init_state.reset_mode}(env_ids)")
        else:
            raise NameError(f"Unknown default setup: {self.cfg.init_state.reset_mode}")

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # start base position shifted in X-Y plane
        if self.custom_origins:
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # xy position within 1m of the center
            # self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.root_states[env_ids] # FIXME: maybe this should be self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids] 

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                    gymtorch.unwrap_tensor(self.root_states),
                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def reset_to_basic(self, env_ids):
        """
        Reset to a single initial state
        """
        # dof states
        self.dof_pos[env_ids] = self.default_dof_pos 
        self.dof_vel[env_ids] = 0 

        # base states
        self.root_states[env_ids] = self.base_init_state


    def reset_to_range(self, env_ids):
        """
        Reset to a uniformly random distribution of states, sampled from a
        range for each state
        """
        # dof states
        self.dof_pos[env_ids] = random_sample(env_ids,
                                    self.dof_pos_range[:, 0],
                                    self.dof_pos_range[:, 1],
                                    device=self.device)
        self.dof_vel[env_ids] = random_sample(env_ids,
                        self.dof_vel_range[:, 0],
                        self.dof_vel_range[:, 1],
                        device=self.device)

        # base states
        random_com_pos = random_sample(env_ids,
                                    self.root_pos_range[:, 0],
                                    self.root_pos_range[:, 1],
                                    device=self.device)

        quat = quat_from_euler_xyz(random_com_pos[:, 3],
                                        random_com_pos[:, 4],
                                        random_com_pos[:, 5]) 

        self.root_states[env_ids, 0:7] = torch.cat((random_com_pos[:, 0:3],
                                    quat_from_euler_xyz(random_com_pos[:, 3],
                                                        random_com_pos[:, 4],
                                                        random_com_pos[:, 5])),
                                                    1)
        self.root_states[env_ids, 7:13] = random_sample(env_ids,
                                    self.root_vel_range[:, 0],
                                    self.root_vel_range[:, 1],
                                    device=self.device)


    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))


    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]


    def _update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _update_reward_curriculum(self, env_ids):
        # If the move in direction reward is above 80% of the maximum, remove that reward
        if "move_in_direction" in self.reward_scales.keys():
            if torch.mean(self.episode_sums["tracking_ee_pos_target"][env_ids]) * self.cfg.commands.resampling_time / self.cfg.env.episode_length_s >\
            0.5 * self.reward_scales["tracking_ee_pos_target"]:
            # if self.common_step_counter > 25 * 400:
                self.reward_scales.pop("move_in_direction")
                self.reward_functions.remove(self._reward_move_in_direction)
                self.reward_names.remove("move_in_direction")
        if "heading" in self.reward_scales.keys():
            if torch.mean(self.episode_sums["heading"][env_ids]) / self.max_episode_length > 0.75 * self.reward_scales["heading"]:
            # if self.common_step_counter > 25 * 400:
                self.reward_scales.pop("heading")
                self.reward_functions.remove(self._reward_heading)
                self.reward_names.remove("heading")


    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # find ids for end effectors defined in env. specific config files
        ee_ids = []
        kp_ids = []
        for body_name in self.cfg.asset.end_effectors:
            ee_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], body_name)
            ee_ids.append(ee_id)
        for keypoint in self.cfg.asset.keypoints:
            kp_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], keypoint)
            kp_ids.append(kp_id)
        self.end_eff_ids = to_torch(ee_ids, device=self.device, dtype=torch.long)
        self.keypoint_ids = to_torch(kp_ids, device=self.device, dtype=torch.long)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "legged_robot")
        mass_matrix_tensor = self.gym.acquire_mass_matrix_tensor(self.sim, "legged_robot")

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self.jacobians = gymtorch.wrap_tensor(jacobian_tensor)
        self.mass_matrices = gymtorch.wrap_tensor(mass_matrix_tensor)
        
        self.rigid_body_state = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)
        self._rigid_body_pos = self.rigid_body_state[..., 0:3]
        self._rigid_body_vel = self.rigid_body_state[..., 7:10]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx),
                                device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.],
                                device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actuators,
                                   dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(self.num_actuators, dtype=torch.float,
                                    device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actuators, dtype=torch.float,
                                    device=self.device, requires_grad=False)
        self.dof_pos_target = torch.zeros(self.num_envs, self.num_actuators,
                                        dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.dof_vel_history = torch.zeros(self.num_envs, self.num_actuators*2,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False)
        # * additional buffer for last ctrl: whatever is actually used for PD control (which can be shifted compared to actuations)
        self.actuation_history = torch.zeros(self.num_envs, self.num_actuators*3,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.commands = torch.zeros(self.num_envs,
                                    self.cfg.commands.num_commands,
                                    dtype=torch.float, device=self.device,
                                    requires_grad=False) # x vel, y vel, yaw vel, heading
        self.base_lin_vel = quat_rotate_inverse(self.base_quat,
                                                self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat,
                                                self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat,
                                                     self.gravity_vec)

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            self.measured_heights = self._get_heights()
        else:
            self.measured_heights = 0

        if self.cfg.control.exp_avg_decay:
            self.dof_pos_avg = torch.zeros(self.num_envs, self.num_actuators,
                                            dtype=torch.float,
                                            device=self.device, requires_grad=False)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float,
                                           device=self.device,
                                           requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # * check that init range highs and lows are consistent
        # * and repopulate to match 
        if self.cfg.init_state.reset_mode == "reset_to_range":
            self.dof_pos_range = torch.zeros(self.num_dof, 2,
                                            dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False)
            self.dof_vel_range = torch.zeros(self.num_dof, 2,
                                            dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False)

            for joint, vals in self.cfg.init_state.dof_pos_range.items():
                for i in range(self.num_dof):
                    if joint in self.dof_names[i]:
                        self.dof_pos_range[i, :] = to_torch(vals)

            for joint, vals in self.cfg.init_state.dof_vel_range.items():
                for i in range(self.num_dof):
                    if joint in self.dof_names[i]:
                        self.dof_vel_range[i, :] = to_torch(vals)

            self.root_pos_range = torch.tensor(self.cfg.init_state.root_pos_range,
                    dtype=torch.float, device=self.device, requires_grad=False)
            self.root_vel_range = torch.tensor(self.cfg.init_state.root_vel_range,
                    dtype=torch.float, device=self.device, requires_grad=False)
            # todo check for consistency (low first, high second)


    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)


    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.cfg.terrain.horizontal_scale
        hf_params.row_scale = self.cfg.terrain.horizontal_scale
        hf_params.vertical_scale = self.cfg.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.cfg.terrain.border_size 
        hf_params.transform.p.y = -self.cfg.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution
        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)


    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)


    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        asset_options.use_physx_armature = True

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        dof_props_asset["armature"] = self.cfg.asset.rotor_inertia
        dof_props_asset["damping"] = self.cfg.asset.angular_damping
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.rigid_body_idx = {name: i for i, name in enumerate(body_names)}
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.dof_idx = {name: i for i, name in enumerate(self.dof_names)}
        self.num_bodies = len(body_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            legged_robot_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "legged_robot", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, legged_robot_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, legged_robot_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, legged_robot_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(legged_robot_handle)

        self.feet_ids = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_ids[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])


    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.


    def _parse_cfg(self):
        super()._parse_cfg()
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False

    def _visualization(self):
        self.gym.clear_lines(self.viewer)
        self._draw_heightmap_vis()

    def _draw_heightmap_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.cfg.terrain.measure_heights:
            return
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.01, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 


    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(np.linspace(*self.cfg.terrain.measured_points_y_range, self.cfg.terrain.measured_points_y_num_sample), device=self.device, requires_grad=False)
        x = torch.tensor(np.linspace(*self.cfg.terrain.measured_points_x_range, self.cfg.terrain.measured_points_x_num_sample), device=self.device, requires_grad=False)

        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points


    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return -torch.square(self.base_lin_vel[:, 2])


    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity (x:roll, y:pitch)
        return -torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)


    def _reward_orientation(self):
        # Penalize non flat base orientation
        return -torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)


    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return -torch.square(base_height - self.cfg.rewards.base_height_target)


    def _reward_torques(self):
        # Penalize torques
        return -torch.sum(torch.square(self.torques), dim=1)


    def _reward_dof_vel(self):
        # Penalize dof velocities
        return -torch.sum(torch.square(self.dof_vel), dim=1)


    def _reward_actuation_rate(self):
        # Penalize changes in actuations
        nact = self.num_actuators
        dt2 = (self.dt*self.cfg.control.decimation)**2
        error = torch.square(self.actuation_history[:, :nact] - self.actuation_history[:, nact:2*nact])/dt2
        return -torch.sum(error, dim=1)


    def _reward_actuation_rate2(self):
        # Penalize changes in actuations
        nact = self.num_actuators
        dt2 = (self.dt*self.cfg.control.decimation)**2
        error = torch.square(self.actuation_history[:, :nact]  \
                            - 2*self.actuation_history[:, nact:2*nact]  \
                            + self.actuation_history[:, 2*nact:])/dt2
        return -torch.sum(error, dim=1)


    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return -torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)


    def _reward_termination(self):
        # Terminal reward / penalty
        return -(self.reset_buf * ~self.timed_out).float()


    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return -torch.sum(out_of_limits, dim=1)


    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return -torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)


    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return -torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)


    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        error = torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2])
        error = torch.exp(-error/self.cfg.rewards.tracking_sigma)
        return -torch.sum(error, dim=1)


    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return -torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return -torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)


    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return -torch.sum((torch.norm(self.contact_forces[:, self.feet_ids, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

# * ######################### HELPER FUNCTIONS ############################## * #

    def _neg_exp(self, x, a=1):
        """ shorthand helper for negative exponential e^(-x/a)
            a: range of x
        """
        return torch.exp(-(x/a)/self.cfg.rewards.tracking_sigma)

    def _negsqrd_exp(self, x, a=1):
        """ shorthand helper for negative squared exponential e^(-(x/a)^2)
            a: range of x
        """
        return torch.exp(-torch.square(x/a)/self.cfg.rewards.tracking_sigma)