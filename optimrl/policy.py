import abc
from typing import Any, Dict

import gym
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
from torch.distributions.normal import Normal


def default_env_creation(env_name: str) -> gym.Env:
    return gym.make(env_name)


class GymPolicy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        self._dummy_param_for_device_ = nn.Parameter(torch.empty(0))

    @abc.abstractmethod
    def forward(self, obs: Any, *args, **kwargs) -> Dict[str, Any]:
        pass

    def act(self, obs: Any, *args, **kwargs):
        out = self.forward(obs, *args, **kwargs)
        return out["action"], out

    @property
    def device(self) -> torch.device:
        return self._dummy_param_for_device_.device

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))


class ACPolicy(GymPolicy, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def critic(self, obs, *args, **kwargs) -> torch.Tensor:
        pass

    def log_prob(self, dist: td.Distribution, actions: torch.Tensor):
        if isinstance(dist, td.Categorical):
            return dist.log_prob(actions)
        return dist.log_prob(actions).sum(axis=-1)


class LinearACPolicy(ACPolicy):
    def __init__(
        self,
        obs_dims: int,
        action_dims: int,
        continuous_actions: bool = False,
        action_std_init: float = 0.6,
    ):
        super().__init__()
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.continuous_actions = continuous_actions
        self.action_std_init = action_std_init

        if self.continuous_actions:
            # self.action_var = nn.Parameter(
            #     torch.full((action_dims,), action_std_init * action_std_init),
            #     requires_grad=False,
            # )
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dims))

        self.policy_net = nn.Sequential(
            nn.Linear(obs_dims, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dims, bias=False),
        )

        self.critic_net = nn.Sequential(
            nn.Linear(obs_dims, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1, bias=False),
        )

    def log_prob(self, dist: td.Distribution, actions: torch.Tensor):
        if isinstance(dist, td.Categorical):
            return dist.log_prob(actions)
        return dist.log_prob(actions).sum(1)

    def forward(self, obs: Any) -> Dict[str, Any]:
        mu = self.policy_net(obs)
        # print(mu.shape)
        # print(self.actor_logstd.shape)
        actor_logstd = self.actor_logstd.expand_as(mu)
        action_std = torch.exp(actor_logstd)
        dist = Normal(mu, action_std)
        action = dist.sample()
        log_probs = self.log_prob(dist, action)
        return {"action": action, "dist": dist, "log_probs": log_probs}

    def critic(self, obs: Any):
        return self.critic_net(obs).squeeze()

    def act(self, obs: Any):
        out = self.forward(obs)
        return np.squeeze(out["action"].detach().cpu().numpy()), out
