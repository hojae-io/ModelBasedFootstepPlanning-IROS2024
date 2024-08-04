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

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class LivePlotter:
    plt.ion()
    def __init__(self, log_states, log_commands, log_rewards, robot_index, joint_index, dt):
        self.log_states = log_states
        self.log_commands = log_commands
        self.log_rewards = log_rewards
        self.robot_index = robot_index
        self.joint_index = joint_index
        self.dt = dt
        self.episode_length = 0

        self.states_dict = defaultdict(lambda: defaultdict(list)) # defaultdict(list)
        self.states_plot_handles = defaultdict(list)

        for state in self.log_states:
            fig_handles, axs_handles = plt.subplots(3)
            self.states_plot_handles[state] = {'fig_handles': fig_handles, 'axs_handles': axs_handles}

        self.commands_dict = defaultdict(lambda: defaultdict(list))

        self.rewards_dict = defaultdict(list)
        self.rewards_plot_handles = defaultdict(list)

        for reward in self.log_rewards:
            fig_handles, axs_handles = plt.subplots(1)
            self.rewards_plot_handles[reward] = {'fig_handles': fig_handles, 'axs_handles': axs_handles}

        self.extras_dict = defaultdict(list)

    def log(self, reset_flag, manual_reset_flag, states_dict, commands_dict, rewards_dict, extras_dict):
        if reset_flag or manual_reset_flag:
            self.reset()

        for key, value in states_dict.items():
            if key == 'root_states':
                self.states_dict[key]['x'].append(value[7].item())
                self.states_dict[key]['y'].append(value[8].item())
                self.states_dict[key]['z'].append(value[9].item())
            else:
                self.states_dict[key]['x'].append(value[0].item())
                self.states_dict[key]['y'].append(value[1].item())
                self.states_dict[key]['z'].append(value[2].item())

        for key, value in commands_dict.items():
            if key == 'commands':
                self.commands_dict[key]['lin_vel_x'].append(value[0].item())
                self.commands_dict[key]['lin_vel_y'].append(value[1].item())
                self.commands_dict[key]['yaw_vel'].append(value[2].item())
            if key == 'step_commands':
                self.commands_dict[key]['left_x'].append(value[0,0].item())
                self.commands_dict[key]['left_y'].append(value[0,1].item())
                self.commands_dict[key]['right_x'].append(value[1,0].item())
                self.commands_dict[key]['right_y'].append(value[1,1].item())

        for key, value in rewards_dict.items():
            self.rewards_dict[key].append(value.item())

        for key, value in extras_dict.items():
            self.extras_dict[key].append(value)

        self.episode_length += 1

    def reset(self):
        for state in self.log_states:
            for axs_handle in self.states_plot_handles[state]['axs_handles']:
                axs_handle.cla()

        for reward in self.log_rewards:
            self.rewards_plot_handles[reward]['axs_handles'].cla()

        self.states_dict.clear()
        self.commands_dict.clear()
        self.rewards_dict.clear()
        self.extras_dict.clear()
        self.episode_length = 0

    def plot(self):
        for state in self.log_states:
            if state == 'foot_states_left_log':
                self.states_plot_handles[state]['axs_handles'][0].plot(np.arange(self.episode_length), self.states_dict[state]['x'], 'c-', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][0].plot(np.arange(self.episode_length), self.commands_dict['step_commands']['left_x'], 'b:', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][0].set(ylabel='X [m]')
                self.states_plot_handles[state]['axs_handles'][1].plot(np.arange(self.episode_length), self.states_dict[state]['y'], 'c-', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][1].plot(np.arange(self.episode_length), self.commands_dict['step_commands']['left_y'], 'b:', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][1].set(ylabel='Y [m]')
                self.states_plot_handles[state]['axs_handles'][2].plot(np.arange(self.episode_length), self.states_dict[state]['z'], 'c-', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][2].axhline(y=0, color = 'b', linestyle = 'dotted', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][2].set(xlabel='episode length', ylabel='Z [m]')
            
            elif state == 'foot_states_right_log':
                self.states_plot_handles[state]['axs_handles'][0].plot(np.arange(self.episode_length), self.states_dict[state]['x'], 'm-', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][0].plot(np.arange(self.episode_length), self.commands_dict['step_commands']['right_x'], 'r:', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][0].set(ylabel='X [m]')
                self.states_plot_handles[state]['axs_handles'][1].plot(np.arange(self.episode_length), self.states_dict[state]['y'], 'm-', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][1].plot(np.arange(self.episode_length), self.commands_dict['step_commands']['right_y'], 'r:', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][1].set(ylabel='Y [m]')
                self.states_plot_handles[state]['axs_handles'][2].plot(np.arange(self.episode_length), self.states_dict[state]['z'], 'm-', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][2].axhline(y=0, color = 'r', linestyle = 'dotted', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][2].set(xlabel='episode length', ylabel='Z [m]')

            elif state == 'base_lin_vel':
                self.states_plot_handles[state]['axs_handles'][0].plot(np.arange(self.episode_length), self.states_dict[state]['x'], 'k-', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][0].plot(np.arange(self.episode_length), self.commands_dict['commands']['lin_vel_x'], 'k--', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][0].set(ylabel='X [m/s]')
                self.states_plot_handles[state]['axs_handles'][0].legend(['measured', 'commanded'], loc='upper right')
                self.states_plot_handles[state]['axs_handles'][1].plot(np.arange(self.episode_length), self.states_dict[state]['y'], 'k-', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][1].plot(np.arange(self.episode_length), self.commands_dict['commands']['lin_vel_y'], 'k--', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][1].set(ylabel='Y [m/s]')
                self.states_plot_handles[state]['axs_handles'][2].plot(np.arange(self.episode_length), self.states_dict[state]['z'], 'k-', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][2].set(xlabel='episode length', ylabel='Z [m/s]')
            
            elif state == 'base_ang_vel':
                self.states_plot_handles[state]['axs_handles'][0].plot(np.arange(self.episode_length), self.states_dict[state]['x'], 'k-', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][0].set(ylabel='X [rad/s]')
                self.states_plot_handles[state]['axs_handles'][1].plot(np.arange(self.episode_length), self.states_dict[state]['y'], 'k-', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][1].set(ylabel='Y [rad/s]')
                self.states_plot_handles[state]['axs_handles'][2].plot(np.arange(self.episode_length), self.states_dict[state]['z'], 'k-', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][2].plot(np.arange(self.episode_length), self.commands_dict['commands']['yaw_vel'], 'k--', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][2].set(xlabel='episode length', ylabel='Z [rad/s]')

            elif state == 'base_pos':
                self.states_plot_handles[state]['axs_handles'][0].plot(np.arange(self.episode_length), self.states_dict[state]['x'], 'k-', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][0].set(ylabel='X [m]')
                self.states_plot_handles[state]['axs_handles'][1].plot(np.arange(self.episode_length), self.states_dict[state]['y'], 'k-', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][1].set(ylabel='Y [m]')
                self.states_plot_handles[state]['axs_handles'][2].plot(np.arange(self.episode_length), self.states_dict[state]['z'], 'k-', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][2].set(xlabel='episode length', ylabel='Z [m]')

            elif state == 'root_states':
                self.states_plot_handles[state]['axs_handles'][0].plot(np.arange(self.episode_length), self.states_dict[state]['x'], 'k-', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][0].plot(np.arange(self.episode_length), self.commands_dict['commands']['lin_vel_x'], 'k--', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][0].set(ylabel='v_x [m/s]')
                self.states_plot_handles[state]['axs_handles'][1].plot(np.arange(self.episode_length), self.states_dict[state]['y'], 'k-', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][1].plot(np.arange(self.episode_length), self.commands_dict['commands']['lin_vel_y'], 'k--', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][1].set(ylabel='v_y [m/s]')
                self.states_plot_handles[state]['axs_handles'][2].plot(np.arange(self.episode_length), self.states_dict[state]['z'], 'k-', linewidth=1.)
                self.states_plot_handles[state]['axs_handles'][2].set(xlabel='episode length', ylabel='Z [m]')

            self.states_plot_handles[state]['fig_handles'].canvas.set_window_title("state: " + state)
            self.states_plot_handles[state]['fig_handles'].canvas.draw()
            self.states_plot_handles[state]['fig_handles'].canvas.flush_events()

        for reward in self.log_rewards:
            self.rewards_plot_handles[reward]['axs_handles'].plot(np.arange(self.episode_length), self.rewards_dict[reward], 'b-', linewidth=1.)
            self.rewards_plot_handles[reward]['axs_handles'].set(xlabel='episode length', ylabel='reward')
        
            self.rewards_plot_handles[reward]['fig_handles'].canvas.set_window_title("reward: " + reward)
            self.rewards_plot_handles[reward]['fig_handles'].canvas.draw()
            self.rewards_plot_handles[reward]['fig_handles'].canvas.flush_events()
