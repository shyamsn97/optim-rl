import abc
from typing import Optional, Tuple

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F


def get_convs(
    initial_size,
    output_size,
    initial_channels,
    out_channels,
    channel_dimension=32,
    bias=True,
):
    size = initial_size
    convs = []
    if size == output_size:
        convs.append(torch.nn.Conv2d(initial_channels, out_channels, 1, bias=bias))
    in_channels = initial_channels
    while size > output_size:
        if (size // 2) == output_size:
            convs.append(
                torch.nn.Conv2d(in_channels, out_channels, 3, 2, padding=1, bias=bias)
            )
        else:
            convs.append(
                torch.nn.Conv2d(
                    in_channels, channel_dimension, 3, 2, padding=1, bias=bias
                )
            )
            convs.append(torch.nn.ReLU())
            in_channels = channel_dimension
        size = size // 2
    return torch.nn.Sequential(*convs)


def get_deconvs(
    initial_size,
    output_size,
    initial_channels,
    out_channels,
    channel_dimension=32,
    bias=True,
):
    size = initial_size
    deconvs = []
    if size == output_size:
        deconvs.append(torch.nn.Conv2d(initial_channels, out_channels, 1, bias=bias))
    in_channels = initial_channels
    while size < output_size:
        if (size * 2) == output_size:
            deconvs.append(
                torch.nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    3,
                    2,
                    padding=1,
                    output_padding=1,
                    bias=bias,
                )
            )
        else:
            deconvs.append(
                torch.nn.ConvTranspose2d(
                    in_channels,
                    channel_dimension,
                    3,
                    2,
                    padding=1,
                    output_padding=1,
                    bias=bias,
                )
            )
            deconvs.append(torch.nn.ReLU())
            in_channels = channel_dimension
        size = size * 2
    return torch.nn.Sequential(*deconvs)


class Distribution(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = logits

    @abc.abstractmethod
    def dist(self, logit_dim: int = -1) -> td.Distribution:
        pass

    def sample(self, logit_dim: int = -1) -> torch.Tensor:
        dist = self.dist(logit_dim)
        return dist.sample()

    def forward(
        self, logit_dim: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor, td.Distribution]:
        sample, dist = self.sample(logit_dim)
        return self.logits, sample, dist


class NormalDistribution(Distribution):
    def __init__(self, logits: torch.Tensor):
        super().__init__(logits)
        self.logits = logits

    def dist(
        self, logit_dim: int = -1, logits: Optional[torch.Tensor] = None, std=None
    ) -> td.Distribution:
        if logits is None:
            logits = self.logits
        if std is None:
            mean, std = torch.chunk(logits, 2, dim=logit_dim)
            std = F.softplus(std) + 0.1
        else:
            mean = logits
            std = 1.0
        dist = td.Normal(mean, std)
        return dist


class BackendModule(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, logits: torch.Tensor):
        """
        Sample from a distribution
        """


class LinearBackendModule(BackendModule):
    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.net = nn.Sequential(
            nn.Linear(input_dims, output_dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.size(0), -1)
