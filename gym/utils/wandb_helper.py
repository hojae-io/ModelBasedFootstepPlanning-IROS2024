
def recursive_value_find(cfg, location):
    if len(location) == 1:
        return getattr(cfg, location[0])

    if hasattr(cfg, location[0]):
        return recursive_value_find(getattr(cfg, location[0]), location[1:])
    else:
        raise Exception(f"I couldn't find the value {location[0]} that you specified")


def craft_log_config(env_cfg, train_cfg, wandb_cfg, what_to_log):
    for log_key in what_to_log:
        location = what_to_log[log_key]
        if location[0] == 'train_cfg':
            wandb_cfg[log_key] = recursive_value_find(train_cfg, location[1:])
        elif location[0] == 'env_cfg':
            wandb_cfg[log_key] = recursive_value_find(env_cfg, location[1:])
        else:
            raise Exception(f"You didn't specify a valid cfg file in location: {location}")
