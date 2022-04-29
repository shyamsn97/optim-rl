from typing import Tuple

import torch
import torch.distributions as td
import torch.nn as nn

from ezrl.algorithms.dreamer.utils import (
    BackendModule,
    DistributionModel,
    LinearBackendModule,
    NormalDistributionModel,
)


class RecurrentModel(nn.Module):
    """
    Defined as:
        h_t = f(h_t-1, z_t-1, a_t-1)

        h_t: output hidden state at timestep t

        f: rnn model
        h_t-1: previous hidden state
        z_t-1: previous latent state
        a_t-1: previous action
    """

    def __init__(self, hidden_dim: int, latent_dim: int, action_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.recurrent_input_dim = self.action_dim + self.latent_dim
        self.rnn = nn.GRUCell(self.recurrent_input_dim, self.hidden_dim)

    def initialize_state(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim)

    def forward(
        self,
        prev_hidden: torch.Tensor,
        prev_latent_state: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        inp = torch.cat([prev_action, prev_latent_state], dim=-1)
        return self.rnn(inp, prev_hidden)


class RepresentationModel(nn.Module):
    """
    Posterior Model

    Defined as:
        z_t ~ q(z_t | h_t, x_t)

        z_t: posterior latent state at timestep t

        q: posterior distribution to sample latent state from
        h_t: current hidden state
        x_t: current observation encoding
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        obs_encoding_dim: int,
        backend_module: BackendModule = LinearBackendModule,
        distribution_model: DistributionModel = NormalDistributionModel,
    ):
        super().__init__()
        self.obs_encoding_dim = obs_encoding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.backend_module = backend_module
        self.distribution_model = distribution_model
        self.net = distribution_model(
            nn.Sequential(
                backend_module(obs_encoding_dim + hidden_dim, 32),
                nn.Tanh(),
                nn.Linear(32, self.latent_dim * 2),
            )
        )

    def forward(
        self, hidden_state: torch.Tensor, obs_encoding: torch.Tensor
    ) -> Tuple[torch.Tensor, td.Distribution]:
        inp = torch.cat([hidden_state, obs_encoding], dim=-1)
        out = self.net(inp)
        return out


class TransitionPredictor(nn.Module):
    """
    Prior Model

    Defined as:
        zhat_t ~ p(zhat_t | h_t)

        zhat_t: prior latent state at timestep t

        p: prior distribution to sample latent state from
        h_t: current hidden state
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        backend_module: BackendModule = LinearBackendModule,
        distribution_model: DistributionModel = NormalDistributionModel,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.backend_module = backend_module
        self.distribution_model = distribution_model
        self.net = distribution_model(
            nn.Sequential(
                backend_module(self.hidden_dim, 32),
                nn.Tanh(),
                nn.Linear(32, self.latent_dim * 2),
            )
        )

    def forward(
        self, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, td.Distribution]:
        return self.net(hidden_state)


class RSSM(nn.Module):
    def __init__(
        self,
        recurrent_model: RecurrentModel,
        representation_model: RepresentationModel,
        transition_predictor: TransitionPredictor,
    ):
        super().__init__()

        self.action_dim = recurrent_model.action_dim
        self.latent_dim = recurrent_model.latent_dim
        self.hidden_dim = recurrent_model.hidden_dim
        self.obs_encoding_dim = representation_model.obs_encoding_dim

        # h_t = f(h_t-1, z_t-1, a_t-1)
        self.recurrent_model: RecurrentModel = recurrent_model

        # posterior, z_t ~ q(z_t | h_t, x_t)
        self.representation_model: RepresentationModel = representation_model
        # prior, zhat_t ~ p(zhat_t | h_t)
        self.transition_predictor: TransitionPredictor = transition_predictor

    def initialize_hidden_state(self, batch_size: int = 1) -> torch.Tensor:
        return self.recurrent_model.initialize_state(batch_size)

    def prior(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, td.Distribution]:
        return self.transition_predictor(hidden_state)

    def posterior(
        self, hidden_state: torch.Tensor, obs_encoding: torch.Tensor
    ) -> Tuple[torch.Tensor, td.Distribution]:
        return self.representation_model(hidden_state, obs_encoding)

    def recurrent(
        self,
        prev_hidden_state: torch.Tensor,
        prev_latent_state: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        return self.recurrent_model(prev_hidden_state, prev_latent_state, prev_action)


class ObsDecoder(nn.Module):
    """
    Observation Decoder

    Defined as:
        xhat_t ~ p(xhat_t | h_t, z_t)

        xhat_t: posterior latent state at timestep t

        p: prior distribution to decode obs from latent state
        h_t: current hidden state
        z_t: current latent state
    """

    def __init__(self, obs_decoder_model: nn.Module, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.net = NormalDistributionModel(obs_decoder_model)

    def forward(
        self, hidden_state: torch.Tensor, latent_state: torch.Tensor
    ) -> Tuple[torch.Tensor, td.Distribution]:
        inp = torch.cat([hidden_state, latent_state], dim=-1)
        return self.net(inp)


class RewardPredictor(nn.Module):
    """
    Observation Decoder

    Defined as:
        rhat_t ~ p(rhat_t | h_t, z_t)

        rhat_t: prior reward prediction at timestep t

        p: prior distribution to predict rewards
        h_t: current hidden state
        z_t: current latent state
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        backend_module: BackendModule = LinearBackendModule,
        distribution_model: DistributionModel = NormalDistributionModel,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.backend_module = backend_module
        self.distribution_model = distribution_model
        self.net = distribution_model(
            nn.Sequential(
                backend_module(latent_dim + hidden_dim, 32), nn.Tanh(), nn.Linear(32, 2)
            )
        )

    def forward(
        self, hidden_state: torch.Tensor, latent_state: torch.Tensor
    ) -> Tuple[torch.Tensor, td.Distribution]:
        inp = torch.cat([hidden_state, latent_state], dim=-1)
        return self.net(inp)


class GammaPredictor(nn.Module):
    """
    Observation Decoder

    Defined as:
        gammahat_t ~ p(rhat_t | h_t, z_t)

        gammahat_t: prior reward prediction at timestep t

        p: prior distribution to predict rewards
        h_t: current hidden state
        z_t: current latent state
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        backend_module: BackendModule = LinearBackendModule,
        distribution_model: DistributionModel = NormalDistributionModel,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.backend_module = backend_module
        self.distribution_model = distribution_model
        self.net = distribution_model(
            nn.Sequential(
                backend_module(latent_dim + hidden_dim, 32), nn.Tanh(), nn.Linear(32, 2)
            )
        )

    def forward(
        self, hidden_state: torch.Tensor, latent_state: torch.Tensor
    ) -> Tuple[torch.Tensor, td.Distribution]:
        inp = torch.cat([hidden_state, latent_state], dim=-1)
        return self.net(inp)
