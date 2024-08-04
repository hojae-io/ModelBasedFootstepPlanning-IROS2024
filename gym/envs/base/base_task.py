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

import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch
from gym.utils.helpers import class_to_dict
from typing import Union
from .legged_robot_config import LeggedRobotCfg
from .fixed_robot_config import FixedRobotCfg
from gym.utils import BaseKeyboardInterface

# Base class for RL tasks
class BaseTask:

    def __init__(self, cfg: Union[LeggedRobotCfg, FixedRobotCfg], sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_actuators = cfg.env.num_actuators

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.timed_out = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None
        self.record = False
        self.record_done = False
        self.screenshot = False

        # if running with a viewer, set up keyboard shortcuts and camera
        if not self.headless:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self._setup_keyboard_interface()
            self.camera_handle = self.gym.create_camera_sensor(self.envs[0], gymapi.CameraProperties())

        # * prepare dicts of functions
        self.reward_names = []
        self.termination_reward_names = []

    def _setup_keyboard_interface(self):
        self.keyboard_interface = BaseKeyboardInterface(self)

    def get_states(self, obs_list):
        return torch.cat([self.get_state(obs) for obs in obs_list], dim=-1)

    def get_state(self, name):
        if name in self.scales.keys():
            return getattr(self, name)*self.scales[name]
        else:
            return getattr(self, name)

    def set_states(self, state_list, values):
        idx = 0
        for state in state_list:
            state_dim = getattr(self, state).shape[1]
            self.set_state(state, values[:, idx:idx+state_dim])
            idx += state_dim
        assert(idx == values.shape[1]), "Actions don't equal tensor shapes"

    def set_state(self, name, value):
        try:
            if name in self.scales.keys():
                setattr(self, name, value/self.scales[name])
            else:
                setattr(self, name, value)
        except AttributeError:
            print("Value for " + name + " does not match tensor shape")

    def reset_envs(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_envs(torch.arange(self.num_envs, device=self.device))
        self.step()
    
    def reset_buffers(self):
        self.rew_buf[:] = 0.
        self.reset_buf[:] = False
        self.manual_reset_flag = False

    def _parse_cfg(self):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.scales = class_to_dict(self.cfg.scaling)
        self.reward_weights = class_to_dict(self.cfg.rewards.weights)
        self.termination_reward_weights = class_to_dict(self.cfg.rewards.termination_weights)
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to
        compute the total reward. Looks for self._reward_<REWARD_NAME>, where
        <REWARD_NAME> are names of all non zero reward weights in the cfg.
        """

        # * remove zero weights, and multiply non-zero ones by dt except for termination weights
        for name in list(self.reward_weights.keys()):
            if self.reward_weights[name] == 0:
                self.reward_weights.pop(name) 
            else:
                self.reward_weights[name] *= self.dt
                self.reward_names.append(name)
        
        for name in list(self.termination_reward_weights.keys()):
            if self.termination_reward_weights[name] == 0:
                self.termination_reward_weights.pop(name) 
            else:
                self.termination_reward_names.append(name)

        # * reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float,
                                               device=self.device,
                                               requires_grad=False)
                             for name in self.reward_weights.keys()}
        self.episode_sums.update({name: torch.zeros(self.num_envs, dtype=torch.float,
                                                    device=self.device,
                                                    requires_grad=False)
                                  for name in self.termination_reward_weights.keys()})

    def compute_reward(self):
        for name in self.reward_names:
            rew = self.reward_weights[name] * self.eval_reward(name)
            self.rew_buf += rew
            self.episode_sums[name] += rew

        for name in self.termination_reward_names:
            rew = self.termination_reward_weights[name] * self.eval_reward(name)
            self.rew_buf += rew
            self.episode_sums[name] += rew

    def eval_reward(self, name):
        return eval("self._reward_"+name+"()")

    def step(self, actions):
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            self.keyboard_interface.update()

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
                # self.gym.draw_viewer(self.viewer, self.sim, True)

            # render the camera sensors
            self.gym.render_all_camera_sensors(self.sim)
