from typing import Tuple

import torch
import torch.distributions as td
import torch.nn as nn

from ezrl.algorithms.dreamer.components import RSSM, GammaPredictor, RewardPredictor


class WorldModel(nn.Module):
    def __init__(
        self,
        rssm: RSSM,
        obs_encoder: nn.Module,
        obs_decoder: nn.Module,
        reward_predictor: RewardPredictor,
        gamma_predictor: GammaPredictor,
    ):
        super().__init__()
        self.rssm = rssm
        self.action_dim = rssm.action_dim
        self.latent_dim = rssm.latent_dim
        self.hidden_dim = rssm.hidden_dim
        self.obs_encoding_dim = rssm.obs_encoding_dim

        self.obs_encoder = obs_encoder

        # xhat_t ~ p(xhat_t | h_t, z_t)
        self.obs_decoder = obs_decoder

        # rhat_t ~ p(rhat_t | h_t, z_t)
        self.reward_predictor = reward_predictor

        # gammahat_t ~ p(rhat_t | h_t, z_t)
        self.gamma_predictor = gamma_predictor

    def recurrent(
        self,
        prev_hidden_state: torch.Tensor,
        prev_latent_state: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        h_t = f(h_t-1, z_t-1, a_t-1)

        Args:
            prev_hidden_state (torch.Tensor): _description_
            prev_latent_state (torch.Tensor): _description_
            prev_action (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return self.rssm.recurrent(prev_hidden_state, prev_latent_state, prev_action)

    def posterior(
        self, hidden_state: torch.Tensor, obs_encoding: torch.Tensor
    ) -> Tuple[torch.Tensor, td.Distribution]:
        """
        z_t ~ q(z_t | h_t, x_t)

        Args:
            obs_encoding (torch.Tensor): _description_
            hidden_state (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, td.Distribution]: _description_
        """
        return self.rssm.posterior(obs_encoding, hidden_state)

    def prior(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, td.Distribution]:
        """
        zhat_t ~ p(zhat_t | h_t)

        Args:
            hidden_state (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, td.Distribution]: _description_
        """
        return self.rssm.prior(hidden_state)

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        encoded_obs = self.obs_encoder(obs)
        return encoded_obs

    def decode_obs(
        self, hidden_state: torch.Tensor, latent_state: torch.Tensor
    ) -> Tuple[torch.Tensor, td.Distribution]:
        """
        xhat_t ~ p(xhat_t | h_t, z_t)

        Args:
            hidden_state (torch.Tensor): _description_
            latent_state (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, td.Distribution]: _description_
        """
        return self.obs_decoder(hidden_state, latent_state)

    def predict_reward(
        self, hidden_state: torch.Tensor, latent_state: torch.Tensor
    ) -> Tuple[torch.Tensor, td.Distribution]:
        """
        rhat_t ~ p(rhat_t | h_t, z_t)

        Args:
            hidden_state (torch.Tensor): _description_
            latent_state (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, td.Distribution]: _description_
        """
        return self.reward_predictor(hidden_state, latent_state)

    def predict_gamma(
        self, hidden_state: torch.Tensor, latent_state: torch.Tensor
    ) -> Tuple[torch.Tensor, td.Distribution]:
        """
        gammahat_t ~ p(rhat_t | h_t, z_t)

        Args:
            hidden_state (torch.Tensor): _description_
            latent_state (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, td.Distribution]: _description_
        """
        return self.gamma_predictor(latent_state, hidden_state)
