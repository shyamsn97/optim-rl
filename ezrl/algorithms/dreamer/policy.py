from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn

from ezrl.algorithms.dreamer.utils import Distribution
from ezrl.algorithms.dreamer.world_model import WorldModel
from ezrl.policy import ACPolicy


class DreamerPolicy(ACPolicy):
    def __init__(
        self,
        world_model: WorldModel,
    ):
        super().__init__()
        self.world_model = world_model

        self.action_dim = world_model.action_dim
        self.latent_dim = world_model.latent_dim
        self.hidden_dim = world_model.hidden_dim
        self.obs_encoding_dim = world_model.obs_encoding_dim

        self.policy_net = nn.Sequential(
            nn.Linear(world_model.latent_dim, self.action_dim), nn.Softmax(dim=-1)
        )
        self.critic_net = nn.Linear(world_model.latent_dim, 1)

        log_std = -0.5 * np.ones(self.action_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.__device_param_dummy__ = nn.Parameter(
            torch.empty(0)
        )  # to keep track of device

    @property
    def device(self) -> torch.device:
        return self.__device_param_dummy__.device

    def initialize_hidden_state(self, batch_size: int = 1) -> torch.Tensor:
        return self.world_model.rssm.initialize_hidden_state(batch_size).to(self.device)

    def log_prob(self, dist: td.Distribution, actions: torch.Tensor):
        if isinstance(dist, td.Categorical):
            return dist.log_prob(probs=actions)
        return dist.log_prob(actions).sum(axis=-1)

    def dist(self, action_logits: torch.Tensor) -> td.Distribution:
        std = torch.exp(self.log_std)
        return td.normal.Normal(action_logits, std)

    def forward(self, latent_state: Any) -> Dict[str, Any]:
        mu = self.policy_net(latent_state)
        dist = self.dist(mu)
        action = dist.sample()
        # log_probs = self.log_prob(dist, action)
        value = self.critic_net(latent_state).squeeze()
        return {
            "action": action,
            "dist": dist,
            "value": value,
            "logits": mu,
        }

    def critic(self, latent_state: Any):
        return self.critic_net(latent_state).squeeze()

    def act(self, latent_state: Any):
        out = self.forward(latent_state)
        return np.squeeze(out["action"].detach().cpu().numpy()), out

    def unroll(
        self, initial_observation: torch.Tensor, num_steps: int = 15
    ) -> Tuple[torch.Tensor, Distribution]:
        # initial_observation size -- B x C x H x W
        batch_size = initial_observation.size(0)

        hidden_states = []
        latent_state_logits = []
        reward_logits = []
        discount_logits = []

        actions = []
        values = []

        hidden_state = self.initialize_hidden_state(batch_size)
        encoded = self.world_model.encode_obs(initial_observation)
        latent_state = self.world_model.posterior(hidden_state, encoded)

        for i in range(num_steps):
            latent_state_logits.append(latent_state.logits)
            latent_state_sample = latent_state.sample()

            out = self.forward(latent_state_sample)

            # action_logits = out["logits"]
            rewards = self.world_model.predict_reward(hidden_state, latent_state_sample)
            discounts = self.world_model.predict_discount(
                hidden_state, latent_state_sample
            )

            action_logits = out["logits"]
            value = out["value"]

            hidden_state = self.world_model.recurrent(
                hidden_state, latent_state_sample, action_logits
            )
            latent_state = self.world_model.prior(hidden_state)

            hidden_states.append(hidden_state)
            reward_logits.append(rewards.logits)
            discount_logits.append(discounts.logits)
            actions.append(action_logits)
            values.append(value)

        hidden_states = torch.stack(hidden_states)
        latent_states = self.world_model.prior_model.distribution(
            torch.stack(latent_state_logits)
        )
        rewards = self.world_model.reward_predictor.distribution(
            torch.stack(reward_logits)
        )
        discounts = self.world_model.discount_predictor.distribution(
            torch.stack(discount_logits)
        )
        actions = torch.stack(actions)
        values = torch.stack(values)

        return hidden_states, latent_states, rewards, discounts, actions, values

    def unroll_with_posteriors(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, Distribution]:
        # observations size -- T x B x C x H x W
        # actions size -- T x B x L
        num_steps = observations.size(0)
        batch_size = observations.size(1)

        hidden_states = []
        posterior_logits = []
        prior_logits = []
        decoded_logits = []
        reward_logits = []
        discount_logits = []

        # h_0
        hidden_state = self.initialize_hidden_state(batch_size)

        for i in range(num_steps):
            hidden_states.append(hidden_state)

            # t
            encoded = self.world_model.encode_obs(observations[i])
            posterior = self.world_model.posterior(hidden_state, encoded)
            sampled_posterior = posterior.sample()
            prior = self.world_model.prior(hidden_state)

            decoded = self.world_model.decode_obs(hidden_state, sampled_posterior)
            reward = self.world_model.predict_reward(hidden_state, sampled_posterior)
            discount = self.world_model.predict_discount(
                hidden_state, sampled_posterior
            )

            # h_t+1
            hidden_state = self.world_model.recurrent(
                hidden_state, sampled_posterior, actions[i]
            )

            posterior_logits.append(posterior.logits)
            prior_logits.append(prior.logits)
            decoded_logits.append(decoded.logits)
            reward_logits.append(reward.logits)
            discount_logits.append(discount.logits)

        hidden_states = torch.stack(hidden_states)
        posterior_logits = torch.stack(posterior_logits)
        prior_logits = torch.stack(prior_logits)
        decoded_logits = torch.stack(decoded_logits)
        reward_logits = torch.stack(reward_logits)
        discount_logits = torch.stack(discount_logits)

        return (
            hidden_states,
            self.world_model.posterior_model.distribution(posterior_logits),
            self.world_model.prior_model.distribution(prior_logits),
            self.world_model.obs_decoder.distribution(decoded_logits),
            self.world_model.reward_predictor.distribution(reward_logits),
            self.world_model.discount_predictor.distribution(discount_logits),
        )
