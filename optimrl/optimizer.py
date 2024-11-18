from __future__ import annotations

import abc
from typing import Any, Dict, Union

import torch

from optimrl.rollout import Rollout


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
    def __init__(self, policy: Any, loss_fn: LossFunction):
        self.policy = policy
        self.loss_fn = loss_fn

    @property
    def device(self):
        if getattr(self, "_device", None) is None:
            self._device = torch.device("cpu")
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    @device.deleter
    def device(self, value):
        del self._device

    def to(self, device) -> RLOptimizer:
        self.device = device
        return self

    @abc.abstractmethod
    def step(self, rollout: Rollout, *args, **kwargs) -> None:
        """
        Updates parameters
        """

    def rollout(self, *args, **kwargs) -> Rollout:
        """
        Rollout fn
        """
        return Rollout.rollout(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        return self.step(*args, **kwargs)
