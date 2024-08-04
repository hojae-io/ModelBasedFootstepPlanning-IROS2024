import os
from gym import LEGGED_GYM_ROOT_DIR


def log_and_save(env, env_cfg, train_cfg, runner):
    """Configure local and cloud code logging"""

    # setup local code saving if enabled
    if check_local_saving_flag(train_cfg):
        save_paths = get_local_save_paths(env, env_cfg)
        runner.logger.configure_local_files(save_paths)


def check_local_saving_flag(train_cfg):
    """Check if enable_local_saving is set to true in the training_config"""

    if hasattr(train_cfg, 'logging') and \
       hasattr(train_cfg.logging, 'enable_local_saving'):
        enable_local_saving = train_cfg.logging.enable_local_saving
    else:
        enable_local_saving = False
    return enable_local_saving


def get_local_save_paths(env, env_cfg):
    """Create a save_paths object for saving code locally"""

    learning_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'learning')
    learning_target = os.path.join('learning')

    gym_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'gym')
    gym_target = os.path.join('gym')

    # list of things to copy
    # source paths need the full path and target are relative to log_dir
    save_paths = [
        {'type': 'dir', 'source_dir': learning_dir,
                        'target_dir': learning_target,
            'include_patterns': ('*.py', '*.json')},
        {'type': 'dir', 'source_dir': gym_dir,
                        'target_dir': gym_target,
            'include_patterns': ('*.py', '*.json')}
    ]

    return save_paths
