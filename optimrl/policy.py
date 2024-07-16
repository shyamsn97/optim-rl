import abc
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class Policy:
    @abc.abstractmethod
    def forward(self, obs: Any, prev_output: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Forward method

        Args:
            obs (torch.Tensor): _description_
            prev_output (Dict[str, Any], optional): _description_. Defaults to {}.

        Returns:
            Dict[str, Any]: _description_
        """

    @abc.abstractmethod
    def act(
        self, obs: Any, prev_output: Dict[str, Any] = {}
    ) -> Tuple[Any, Dict[str, Any]]:
        """Main act method

        Args:
            obs (torch.Tensor): _description_
            prev_output (Dict[str, Any], optional): _description_. Defaults to {}.
        """

    @abc.abstractmethod
    def train_step(
        self, obs: torch.Tensor, prev_output: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """Function that is used for train step

        Args:
            obs (torch.Tensor): _description_
            prev_output (Dict[str, Any], optional): _description_. Defaults to {}.

        Returns:
            Dict[str, Any]: _description_
        """


class GymPolicy(nn.Module, Policy):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(
        self, obs: torch.Tensor, prev_output: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """Forward method of policy

        Args:
            obs (torch.Tensor): _description_
            policy (Dict[str, Any]): _description_

        Returns:
            Dict[str, Any]: _description_
        """

    @abc.abstractmethod
    def act(
        self, obs: torch.Tensor, prev_output: Dict[str, Any] = {}
    ) -> Tuple[Any, Dict[str, Any]]:
        """Main act method

        Args:
            obs (torch.Tensor): _description_
            prev_output (Dict[str, Any], optional): _description_. Defaults to {}.
        """

    def train_forward(
        self, obs: torch.Tensor, prev_output: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        return self.forward(obs=obs, prev_output=prev_output)
