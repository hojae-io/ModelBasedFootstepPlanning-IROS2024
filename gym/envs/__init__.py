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

from gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from .pendulum.pendulum import Pendulum
from .pendulum.pendulum_config import PendulumCfg, PendulumRunnerCfg

from .cartpole.cartpole import Cartpole
from .cartpole.cartpole_config import CartpoleCfg, CartpoleRunnerCfg

from .humanoid.humanoid_vanilla import HumanoidVanilla
from .humanoid.humanoid_vanilla_config import HumanoidVanillaCfg, HumanoidVanillaRunnerCfg
from .humanoid.humanoid_controller import HumanoidController
from .humanoid.humanoid_controller_config import HumanoidControllerCfg, HumanoidControllerRunnerCfg

from gym.utils.task_registry import task_registry

task_registry.register("pendulum", Pendulum, PendulumCfg, PendulumRunnerCfg)
task_registry.register("cartpole", Cartpole, CartpoleCfg, CartpoleRunnerCfg)

task_registry.register("humanoid_vanilla", HumanoidVanilla, HumanoidVanillaCfg, HumanoidVanillaRunnerCfg)
task_registry.register("humanoid_controller", HumanoidController, HumanoidControllerCfg, HumanoidControllerRunnerCfg)
                      

