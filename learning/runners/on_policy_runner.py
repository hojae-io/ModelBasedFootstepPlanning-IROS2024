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

import time
import os
from collections import deque
import statistics
import wandb

import torch

from learning.algorithms import PPO
from learning.modules import ActorCritic
from learning.utils import Logger
from learning.env import VecEnv
class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"][self.cfg["algorithm_class_name"]]
        self.policy_cfg = train_cfg["policy"]
        self.logging_cfg = train_cfg["logging"]
        self.device = device
        self.env = env

        self.num_actor_obs = self.get_obs_size(self.policy_cfg["actor_obs"])
        self.num_critic_obs = self.get_obs_size(self.policy_cfg["critic_obs"])
        self.num_actions = self.get_action_size(self.policy_cfg["actions"])
        self.obs_noise_vec = self.get_obs_noise_vec(self.policy_cfg["actor_obs"],
                                                    self.policy_cfg["noise"])
        actor_critic = ActorCritic(self.num_actor_obs,
                                   self.num_critic_obs,
                                   self.num_actions,
                                   **self.policy_cfg).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        
        # * init storage and model
        self.alg.init_storage(self.env.num_envs, 
                              self.num_steps_per_env, 
                              self.num_actor_obs, 
                              self.num_critic_obs, 
                              self.num_actions)
        
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # * Log
        self.log_dir = log_dir
        self.logger = Logger(log_dir, self.env.max_episode_length_s, self.device)
        reward_keys_to_log = list(self.env.reward_weights.keys())
        self.logger.initialize_buffers(self.env.num_envs, reward_keys_to_log)

        self.env.reset()

    def attach_to_wandb(self, wandb, log_freq=100, log_graph=True):
        wandb.watch((self.alg.actor_critic.actor,
                    self.alg.actor_critic.critic),
                    log_freq=log_freq,
                    log_graph=log_graph)

    def learn(self, num_learning_iterations=None, init_at_random_ep_len=False):
       
        if self.logging_cfg['enable_local_saving']:
            self.logger.make_log_dir()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        actor_obs = self.get_noisy_obs(self.policy_cfg["actor_obs"])
        # critic_obs = self.get_noisy_obs(self.policy_cfg["critic_obs"])
        critic_obs = self.get_obs(self.policy_cfg["critic_obs"])

        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.num_learning_iterations = num_learning_iterations
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # * Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(actor_obs, critic_obs)
                    self.set_actions(actions)
                    self.env.step()

                    dones = self.get_dones()
                    timed_out = self.get_timed_out()
                    rewards = self.compute_and_get_rewards()
                    self.reset_envs()
                    infos = self.get_infos()
                    
                    actor_obs = self.get_noisy_obs(self.policy_cfg["actor_obs"])
                    # critic_obs = self.get_noisy_obs(self.policy_cfg["critic_obs"])
                    critic_obs = self.get_obs(self.policy_cfg["critic_obs"])

                    self.alg.process_env_step(rewards, dones, timed_out)

                    # * Book keeping
                    ep_infos.append(infos['episode'])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # * Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            self.iteration_time = collection_time + learn_time
            self.tot_time += self.iteration_time

            self.log_wandb(locals())
            if (it % self.save_interval == 0) and self.logging_cfg['enable_local_saving']:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
            self.current_learning_iteration += 1

        if self.logging_cfg['enable_local_saving']:
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def get_obs(self, obs_list):
        self.env._set_obs_variables()
        observation = self.env.get_states(obs_list).to(self.device)
        return observation
    
    def get_noisy_obs(self, obs_list):
        observation = self.get_obs(obs_list)
        return observation + (2*torch.rand_like(observation) - 1) * self.obs_noise_vec

    def get_obs_noise_vec(self, obs_list, noise_dict):
        noise_vec = torch.zeros(self.get_obs_size(obs_list), device=self.device)
        obs_index = 0
        for obs in obs_list:
            obs_size = self.get_obs_size([obs])
            if obs in noise_dict.keys():
                noise_tensor = torch.ones(obs_size).to(self.device) * noise_dict[obs]
                if obs in self.env.scales.keys():
                    noise_tensor *= self.env.scales[obs]
                noise_vec[obs_index:obs_index+obs_size] = noise_tensor
            obs_index += obs_size
        return noise_vec

    def set_actions(self, actions):
        if hasattr(self.env.cfg.scaling, "clip_actions"):
            actions = torch.clip(actions, -self.env.cfg.scaling.clip_actions, self.env.cfg.scaling.clip_actions)
        self.env.set_states(self.policy_cfg["actions"], actions)

    def get_obs_size(self, obs_list):
        return self.env.get_states(obs_list)[0].shape[0]

    def get_action_size(self, action_list):
        return self.env.get_states(action_list)[0].shape[0]
    
    def get_timed_out(self):
        return self.env.get_state('timed_out').to(self.device)
    
    def get_dones(self):
        return self.env.reset_buf.to(self.device)

    def get_infos(self):
        return self.env.extras
    
    def compute_and_get_rewards(self):
        self.env.compute_reward()
        return self.env.rew_buf
    
    def reset_envs(self):
        env_ids = self.get_dones().nonzero(as_tuple=False).flatten()
        self.env.reset_envs(env_ids)

    def log_wandb(self, locs, width=100, pad=45):
        # * Logging to wandb
        ep_string = f''
        for key in locs['ep_infos'][0]:
            infotensor = torch.tensor([], device=self.device)
            for ep_info in locs['ep_infos']:
                if len(ep_info[key].shape) == 0:
                    ep_info[key] = ep_info[key].unsqueeze(0)
                infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
            value = torch.mean(infotensor)
            self.logger.add_log({"Episode/" + key: value})
            ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.logger.add_log({
            "Loss/value_function": locs['mean_value_loss'],
            "Loss/surrogate": locs['mean_surrogate_loss'],
            "Loss/learning_rate": self.alg.learning_rate,
            "Policy/mean_noise_std": mean_std.item(),
            "Perf/total_fps": fps,
            "Perf/collection time": locs['collection_time'],
            "Perf/learning_time": locs['learn_time'],
        })
        if len(locs['rewbuffer']) > 0:
            self.logger.add_log({
                "Train/mean_reward": statistics.mean(locs['rewbuffer']),
                "Train/mean_episode_length": statistics.mean(locs['lenbuffer']),
            })
        if wandb.run is not None:
            self.logger.log_to_wandb()

        # * Print logging info
        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {self.iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'num_actor_obs' : self.num_actor_obs,
            'actor_hidden_dims' : self.policy_cfg["actor_hidden_dims"],
            'num_actions' : self.num_actions,
            'num_critic_obs' : self.num_critic_obs,
            'critic_hidden_dims' : self.policy_cfg["critic_hidden_dims"],
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            'rollout': self.alg.storage,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_actions(self):
        obs = self.get_obs(self.policy_cfg["actor_obs"])
        return self.alg.actor_critic.act_inference(obs)

    def export(self, path):
        self.alg.actor_critic.export_policy(path)