from __future__ import annotations

import abc
from typing import Any, Dict, Tuple, Union

import torch


class Loss(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def backward(self):
        # to be called like loss in typical torch fashion
        pass


class LossFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> Union[Loss, torch.Tensor, Any]:
        """Loss

        Returns:
            _type_: _description_
        """


class RLOptimizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def zero_grad(self):
        """
        zeros grad
        """

    @abc.abstractmethod
    def loss(
        self, data: Dict[str, Any]
    ) -> Tuple[Union[torch.Tensor, Loss], Dict[str, Any]]:
        """
            Calculates loss on rollout

        Args:
            rollout (Rollout): _description_

        Returns:
            _type_: _description_
        """

    @abc.abstractmethod
    def step(self) -> None:
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

    def update(self, *args, **kwargs) -> Dict[str, Any]:
        self.zero_grad()
        loss, metrics = self.loss(*args, **kwargs)
        loss.backward()
        self.step()
        return metrics

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        return self.update(*args, **kwargs)
