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

from gym import LEGGED_GYM_ROOT_DIR
import os
import time
from colorama import Fore

import isaacgym
from gym.envs import *
from gym.utils import get_args, task_registry, Logger, VisualizationRecorder, ScreenShotter, AnalysisRecorder, CSVLogger, DictLogger, SuccessRater
from gym.scripts.plotting import LivePlotter

import numpy as np
import torch

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 9)
    env_cfg.env.env_spacing = 2.
    env_cfg.env.episode_length_s = int(1e7) # 5, int(1e7)
    env_cfg.seed = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.mesh_type = 'plane' # 'plane', 'trimesh'
    env_cfg.terrain.terrain_kwargs = {'type': 'random_uniform'} # stepping_stones, random_uniform, gap
    env_cfg.terrain.measure_heights = False # True, False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False # True
    env_cfg.init_state.reset_mode = 'reset_to_basic' # 'reset_to_basic', 'reset_to_range'
    env_cfg.commands.resampling_time = -1 # -1, 3, 5, 15
    env_cfg.commands.curriculum = False # True, False
    # env_cfg.viewer.pos = [0, -1.5, 1] # [0, -3.5, 3]
    env_cfg.viewer.pos = [0, -2.5, .5] # [0, -3.5, 3]
    # env_cfg.viewer.lookat = [0, 0, 0] # [1, 1.5, 0]
    env_cfg.viewer.lookat = [0, 0.5, 0] # [1, 1.5, 0]
    env_cfg.commands.ranges.lin_vel_x = [0., 0.] # [0., 0.], [-2., 2.]
    env_cfg.commands.ranges.lin_vel_y = 0. # 0., 1.
    env_cfg.commands.ranges.yaw_vel = 0. # 0., 1.

    if env_cfg.__class__.__name__ == 'HumanoidControllerCfg':
        env_cfg.commands.adjust_step_command = False # True, False
        env_cfg.commands.adjust_prob = 0.05
        env_cfg.commands.sample_angle_offset = 20
        env_cfg.commands.sample_radius_offset = 0.05 # 0.05

        env_cfg.commands.ranges.sample_period = [35, 36] # [20, 21], [35, 36]
        env_cfg.commands.ranges.dstep_width = [0.3, 0.3]

    # * prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # * load policy
    train_cfg.runner.resume = True
    policy_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy_runner.alg.actor_critic.eval()

    # * export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported')
        policy_runner.export(path)
        print('Exported policy model to: ', path)
    
    start = time.time()
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    default_camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)

    # log_states = [ "root_states", "step_length", "step_width" ] # "base_lin_vel", "base_pos", "dof_pos", 
    # log_commands = [ "commands", "dstep_length", "dstep_width"] # "step_commands" 
    log_states = [ "contact_forces" ] # "base_lin_vel", "base_pos", "dof_pos", 
    log_commands = [] # "step_commands" 
    log_rewards = [] #[ "wrong_contact", "step_location", ]
    max_it = int(1e6)
    screenshotter = ScreenShotter(env, train_cfg.runner.experiment_name, policy_runner.log_dir)

    if LIVE_PLOT:
        live_plotter = LivePlotter(log_states, log_commands, log_rewards, robot_index, joint_index, env.dt)
    if RECORD_FRAMES:
        recorder = AnalysisRecorder(env, train_cfg.runner.experiment_name, policy_runner.log_dir)
    if SAVE_CSV:
        csv_logger = CSVLogger(env, train_cfg.runner.experiment_name, policy_runner.log_dir, max_it)
    if SAVE_DICT:
        dict_logger = DictLogger(env, train_cfg.runner.experiment_name, policy_runner.log_dir, max_it)
    if CHECK_SUCCESS_RATE:
        success_rater = SuccessRater(env, train_cfg.runner.experiment_name, policy_runner.log_dir)
        test_episodes = 10000

    for i in range(max_it):
        actions = policy_runner.get_inference_actions()
        policy_runner.set_actions(actions)
        env.step()
        policy_runner.reset_envs()
        
        if env.screenshot:
            image = env.gym.get_camera_image(env.sim, env.envs[0], env.camera_handle, isaacgym.gymapi.IMAGE_COLOR)
            image = image.reshape(image.shape[0], -1, 4)[..., :3]
            screenshotter.screenshot(image)
            env.screenshot = False

        if CUSTOM_COMMANDS:
            # * Scenario 1 (For flat terrain)
            if (i+1) == 500:
                env.commands[:, 0] = 0.5
                env.commands[:, 1] = 0.
            elif (i+1) == 1000: 
                env.commands[:, 0] = 1.0
                env.commands[:, 1] = 0.

        if MOVE_CAMERA and not env.headless:
            camera_position += camera_vel * env.dt
            camera_position = default_camera_position + env.base_pos[robot_index].cpu().numpy()
            env.set_camera(camera_position, camera_position + camera_direction)

        if LIVE_PLOT:
            live_plotter.log(
                env.reset_buf[robot_index].item(),
                env.manual_reset_flag,
                states_dict = {
                    state: eval(f"env.{state}[{robot_index}]", {"env": env}) for state in log_states
                },
                commands_dict = {
                    command: eval(f"env.{command}[{robot_index}]", {"env": env}) for command in log_commands
                },
                rewards_dict = {
                    reward: eval(f"env.episode_sums['{reward}'][{robot_index}]", {"env": env}) for reward in log_rewards
                },
                extras_dict = {
                    'update_count': env.update_count[robot_index].item(),
                    'update_commands_ids': env.update_commands_ids[robot_index].item(),
                }
            )
            live_plotter.plot()
        
        if RECORD_FRAMES:
            image = env.gym.get_camera_image(env.sim, env.envs[0], env.camera_handle, isaacgym.gymapi.IMAGE_COLOR)
            image = image.reshape(image.shape[0], -1, 4)[..., :3]
            recorder.log(
                image,
                states_dict = {
                    state: eval(f"env.{state}[{robot_index}]", {"env": env}) for state in log_states
                },
                commands_dict = {
                    command: eval(f"env.{command}[{robot_index}]", {"env": env}) for command in log_commands
                }
            )
            if env.record_done:
                recorder.save_and_exit()

        if SAVE_CSV:
            csv_logger.log()
            if env.record_done:
                csv_logger.save_and_exit()
        
        if SAVE_DICT:
            dict_logger.log()
            if env.record_done:
                dict_logger.save_and_exit()
        
        if CHECK_SUCCESS_RATE:
            if env.record_done:
                stop = time.time()
                print("collection_time:", stop - start)
                success_rater.save_and_exit()
            success_rater.log(i, test_episodes, env.reset_buf.sum().item(), env.timed_out.sum().item())



if __name__ == '__main__':
    EXPORT_POLICY = True # True, False
    CUSTOM_COMMANDS = False # True, False
    MOVE_CAMERA = True # True, False
    LIVE_PLOT = False # True, False
    RECORD_FRAMES = False # True, False
    SAVE_CSV = False # True, False
    SAVE_DICT = False # True, False
    CHECK_SUCCESS_RATE = False # True, False
    args = get_args()

    # # * custom loading
    # args.load_files = True # True, False
    # args.load_run = 'Feb06_00-27-24_sf' # load run name
    # args.checkpoint = '1000'

    with torch.inference_mode():
        play(args)