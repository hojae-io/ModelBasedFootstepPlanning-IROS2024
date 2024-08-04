import wandb
import torch
from collections import deque
from statistics import mean
import os
import shutil
import fnmatch

class Logger:
    def __init__(self, log_dir, max_episode_length_s, device):
        self.log_dir = log_dir
        self.device = device
        self.avg_window = 100
        self.log = {}
        self.loss_dict = {}
        self.it = 0
        self.tot_iter = 0
        self.learning_iter = 0
        self.mean_episode_length = 0.
        self.total_mean_reward = 0.
        self.max_episode_length_s = max_episode_length_s

    def make_log_dir(self):
        os.makedirs(self.log_dir, exist_ok=True)

    def initialize_buffers(self, num_envs, reward_keys):
        self.current_episode_return = \
            {name: torch.zeros(num_envs, dtype=torch.float, device=self.device,
                                requires_grad=False) for name in reward_keys}
        self.current_episode_length = torch.zeros(num_envs,
                                         dtype=torch.float, device=self.device)
        self.avg_return_buffer = {name:  deque(maxlen=self.avg_window) 
                                    for name in reward_keys}   
        self.avg_length_buffer = deque(maxlen=self.avg_window)
        self.mean_rewards = {"Episode/"+name:  0. for name in reward_keys} 

    def log_to_wandb(self):
        wandb.log(self.log)

    def add_log(self, log_dict):
        self.log.update(log_dict)

    def add_loss(self, loss_dict):
        self.loss_dict.update(loss_dict)

    def update_iterations(self, it, tot_iter, learning_iter):
        self.it = it
        self.tot_iter = tot_iter
        self.learning_iter = learning_iter

    def log_current_reward(self, name, reward):
        if name in self.current_episode_return.keys():
            self.current_episode_return[name] += reward  
    
    def update_episode_buffer(self, dones):
        self.current_episode_length += 1
        terminated_ids = torch.where(dones == True)[0]
        for name in self.current_episode_return.keys():
            self.avg_return_buffer[name].extend(self.current_episode_return[name]
                                        [terminated_ids].cpu().numpy().tolist())
            self.current_episode_return[name][terminated_ids] = 0.
            
        self.avg_length_buffer.extend(self.current_episode_length[terminated_ids]
                                        .cpu().numpy().tolist())
        self.current_episode_length[terminated_ids] = 0
        if (len(self.avg_length_buffer) > 0):
            self.calculate_reward_avg()

    def calculate_reward_avg(self):
        self.mean_episode_length = mean(self.avg_length_buffer)
        self.mean_rewards = {"Episode/"+name:  mean(self.avg_return_buffer[name]) / self.max_episode_length_s
                        for name in  self.current_episode_return.keys()} 
        self.total_mean_reward = sum(list(self.mean_rewards.values()))

    def print_to_terminal(self):
        width=80
        pad=50
        str = f" \033[1m Learning iteration {self.it}/{self.tot_iter} \033[0m "

        log_string = (f"""{'#' * width}\n"""
                      f"""{str.center(width, ' ')}\n\n"""
                      f"""{'Computation:':>{pad}} {self.log['Perf/total_fps']:.0f} steps/s (collection: {self.log[
                      'Perf/collection_time']:.3f}s, learning {self.log['Perf/learning_time']:.3f}s)\n""")
        
        for key, value in self.loss_dict.items():
            log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
        
        log_string += (f"""{'Mean action noise std:':>{pad}} {self.log['Policy/mean_noise_std']:.2f}\n"""
                       f"""{'Mean reward:':>{pad}} {self.total_mean_reward:.2f}\n"""
                       f"""{'Mean episode length:':>{pad}} {self.log['Train/mean_episode_length']:.2f}\n""")

        for key, value in self.mean_rewards.items():
            log_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
                
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.log['Train/total_timesteps']}\n"""
                       f"""{'Iteration time:':>{pad}} {self.log['Train/iteration_time']:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.log['Train/time']:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.log['Train/time'] / (self.it + 1) * (
                               self.learning_iter - self.it):.1f}s\n""")

        print(log_string)

    def configure_local_files(self, save_paths):
        # create ignore patterns dynamically based on include patterns
        def create_ignored_pattern_except(*patterns):
            def _ignore_patterns(path, names):
                keep = set(name for pattern in patterns for name in
                           fnmatch.filter(names, pattern))
                ignore = set(name for name in names if name not in keep and
                             not os.path.isdir(os.path.join(path, name)))
                return ignore
            return _ignore_patterns

        def remove_empty_folders(path, removeRoot=True):
            if not os.path.isdir(path):
                return
            # remove empty subfolders
            files = os.listdir(path)
            if len(files):
                for f in files:
                    fullpath = os.path.join(path, f)
                    if os.path.isdir(fullpath):
                        remove_empty_folders(fullpath)
            # if folder empty, delete it
            files = os.listdir(path)
            if len(files) == 0 and removeRoot:
                os.rmdir(path)

        # copy the relevant source files to the local logs for records
        save_dir = self.log_dir+'/files/'
        for save_path in save_paths:
            if save_path['type'] == 'file':
                os.makedirs(save_dir+save_path['target_dir'],
                            exist_ok=True)
                shutil.copy2(save_path['source_file'],
                             save_dir+save_path['target_dir'])
            elif save_path['type'] == 'dir':
                shutil.copytree(
                    save_path['source_dir'],
                    save_dir+save_path['target_dir'],
                    ignore=create_ignored_pattern_except(
                        *save_path['include_patterns']))
            else:
                print('WARNING: uncaught save path type:', save_path['type'])
                
        remove_empty_folders(save_dir)

