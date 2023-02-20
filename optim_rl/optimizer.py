from __future__ import annotations

import abc
from typing import Any, Union

import torch


class Loss(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def backward(self):
        # to be called like loss in typical torch fashion
        pass


class RLOptimizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def zero_grad(self):
        """
        zeros grad
        """

    @abc.abstractmethod
    def loss_fn(self, *args, **kwargs) -> Union[Loss, torch.Tensor, Any]:
        """
        Calculate loss, passed into step function
        """

    @abc.abstractmethod
    def step(self) -> Any:
        """
        Updates parameters
        """

    def rollout(
        self, rollout_fn, pool=None, num_rollouts: int = 1, *args, **kwargs
    ) -> Any:
        """
        Optional default rollout_fn for the algorithm.
        """
        return rollout_fn(*args, **kwargs)

    def update(self, *args, **kwargs) -> Any:
        self.zero_grad()
        loss = self.loss_fn(*args, **kwargs)
        loss.backward()
        self.step()
