import os
import json
import wandb

from gym import LEGGED_GYM_ROOT_DIR
from gym.scripts.train import train, setup
from gym.utils import get_args, task_registry
from gym.utils.logging_and_saving import wandb_singleton

# torch needs to be imported after isaacgym imports in local source
from torch.multiprocessing import Process
from torch.multiprocessing import set_start_method


def load_sweep_config(file_name):
    return json.load(open(os.path.join(
        LEGGED_GYM_ROOT_DIR, 'gym', 'scripts',
        'sweep_configs', file_name)))


def train_with_sweep_cfg():
    train_cfg, policy_runner = setup()
    train(train_cfg=train_cfg, policy_runner=policy_runner)


def sweep_wandb_mp():
    ''' start a new process for each train function '''

    p = Process(target=train_with_sweep_cfg, args=())
    p.start()
    p.join()
    p.kill()


def start_sweeps(args):
    # * required for multiprocessing CUDA workloads
    set_start_method('spawn')

    # * load sweep_config from JSON file
    if args.wandb_sweep_config is not None:
        sweep_config = load_sweep_config(args.wandb_sweep_config)
    else:
        sweep_config = load_sweep_config('sweep_config_cartpole.json')
    # * set sweep_id if you have a previous id to use
    sweep_id = None
    if args.wandb_sweep_id is not None:
        sweep_id = args.wandb_sweep_id

    wandb_helper = wandb_singleton.WandbSingleton()
    train_cfg = task_registry.train_cfgs[args.task]
    wandb_helper.set_wandb_values(train_cfg, args)

    if wandb_helper.is_wandb_enabled():
        if sweep_id is None:
            sweep_id = wandb.sweep(
                sweep_config,
                entity=wandb_helper.get_entity_name(),
                project=wandb_helper.get_project_name())
        wandb.agent(
            sweep_id,
            sweep_wandb_mp,
            entity=wandb_helper.get_entity_name(),
            project=wandb_helper.get_project_name(),
            count=sweep_config['run_cap'])
    else:
        print('ERROR: No WandB project and entity provided for sweeping')


if __name__ == '__main__':
    args = get_args()
    start_sweeps(args)
