import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .utils import create_MLP
from .actor import Actor
from .critic import Critic

class ActorCritic(nn.Module):
    def __init__(self, num_actor_obs,
                       num_critic_obs,
                       num_actions,
                       actor_hidden_dims=[256, 256, 256],
                       critic_hidden_dims=[256, 256, 256],
                       activation="elu",
                       init_noise_std=1.0,
                       normalize_obs=False,
                       **kwargs):

        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        self.actor = Actor(num_actor_obs,
                           num_actions,
                           actor_hidden_dims,
                           activation,
                           init_noise_std,
                           normalize_obs)

        self.critic = Critic(num_critic_obs,
                             critic_hidden_dims,
                             activation,
                             normalize_obs)

        print(f"Actor MLP: {self.actor.mean_NN}")
        print(f"Critic MLP: {self.critic.NN}")

    @property
    def action_mean(self):
        return self.actor.action_mean

    @property
    def action_std(self):
        return self.actor.action_std

    @property
    def entropy(self):
        return self.actor.entropy
    
    @property
    def std(self):
        return self.actor.std

    def update_distribution(self, observations):
        self.actor.update_distribution(observations)

    def act(self, observations, **kwargs):
        return self.actor.act(observations)

    def get_actions_log_prob(self, actions):
        return self.actor.get_actions_log_prob(actions)

    def act_inference(self, observations):
        return self.actor.act_inference(observations)

    def evaluate(self, critic_observations, actions=None, **kwargs):
        return self.critic.evaluate(critic_observations, actions)

    def export_policy(self, path):
        self.actor.export(path)