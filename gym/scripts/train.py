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

import numpy as np
import os
from datetime import datetime

import isaacgym
from gym.envs import *
from gym.utils import get_args, task_registry, wandb_helper
from gym import LEGGED_GYM_ROOT_DIR
import torch
from gym.utils.logging_and_saving import local_code_save_helper, wandb_singleton

import wandb

def train(args):
    # * Setup environment and policy_runner
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    policy_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    # * Setup wandb
    wandb_helper = wandb_singleton.WandbSingleton()
    wandb_helper.setup_wandb(env_cfg=env_cfg, train_cfg=train_cfg, args=args, log_dir=policy_runner.log_dir)
    local_code_save_helper.log_and_save(
        env, env_cfg, train_cfg, policy_runner)
    wandb_helper.attach_runner(policy_runner=policy_runner)

    # * Train
    policy_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, 
                        init_at_random_ep_len=True)
    
    # * Close wandb
    wandb_helper.close_wandb()

if __name__ == '__main__':
    args = get_args()
    train(args)
