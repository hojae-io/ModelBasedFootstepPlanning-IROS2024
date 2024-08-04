# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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
from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from gym.envs.base.fixed_robot import FixedRobot

class Pendulum(FixedRobot):

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.pendulum_is_upright = torch.cos(self.dof_pos[:,0]) > 0.9

    def compute_observations(self):
        self.obs_buf = torch.cat((
            self.dof_pos,    # [1] "actuator" pos
            self.dof_vel,    # [1] "actuator" vel
        ), dim=-1)
        
        if self.cfg.env.num_critic_obs:
            self.critic_obs_buf = self.obs_buf
        
        if self.add_noise:
            self.obs_buf += (2*torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:1] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[1:] = noise_scales.dof_vel * self.obs_scales.dof_vel
                
        noise_vec = noise_vec * noise_level
        return noise_vec
    
    def _compute_torques(self, actions):
        return torch.clip(actions, -self.torque_limits, self.torque_limits)

    def _reward_pendulum_vel(self):
        return -self.dof_vel[:, 0].square() * self.pendulum_is_upright

    def _reward_upright_pendulum(self):
        return torch.exp(-torch.square(self.dof_pos[:,0]) / 0.25)