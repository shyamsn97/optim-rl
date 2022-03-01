import abc
from typing import Any, Dict

import gym
import torch
import torch.nn as nn


def default_env_creation(env_name: str) -> gym.Env:
    return gym.make(env_name)


class GymPolicy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        self._dummy_param = nn.Parameter(torch.empty(0))

    @abc.abstractmethod
    def forward(self, obs: Any, *args, **kwargs) -> Dict[str, Any]:
        pass

    def act(self, obs: Any, *args, **kwargs):
        out = self.forward(obs, *args, **kwargs)
        return out["action"], out

    @property
    def device(self) -> torch.device:
        return self._dummy_param.device

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
