import torch
import torch.distributions as td
import torch.optim as optim
from torch.distributions.kl import kl_divergence

from ezrl.algorithms.dreamer.policy import DreamerPolicy
from ezrl.optimizer import RLOptimizer


class DreamerOptimizer(RLOptimizer):
    def __init__(
        self,
        policy: DreamerPolicy,
        horizon: int = 15,
        kl_beta: float = 0.1,
        kl_alpha: float = 0.8,
        world_model_lr: float = 2e-4,
        actor_lr: float = 2e-5,
        critic_lr: float = 1e-4,
        grad_clip: float = 100.0,
    ):
        self.policy = policy
        self.world_model = policy.world_model
        self.horizon = horizon
        self.kl_beta = kl_beta
        self.kl_alpha = kl_alpha

        self.world_model_lr = world_model_lr
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.grad_clip = grad_clip

        self.setup_optimizer()

    def setup_optimizer(self):
        self.world_model_optimizer = optim.Adam(
            self.policy.world_model.parameters(), lr=self.world_model_lr
        )
        self.actor_optimizer = optim.Adam(
            self.policy.policy_net.parameters(), lr=self.actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.policy.critic_net.parameters(), lr=self.critic_lr
        )

    def representation_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        # nonterminals: torch.Tensor,
    ):
        # shape should be T x B x shape, where T = timesteps, B = batch

        (
            _,
            posteriors,
            priors,
            decoded_observations,
            predicted_rewards,
            predicted_discounts,
        ) = self.policy.unroll_with_posteriors(observations, actions)

        # image loss
        # - lnp(x_t | h_t, z_t)
        image_dist = td.Independent(decoded_observations.dist(logit_dim=2), 3)
        image_loss = -torch.mean(image_dist.log_prob(observations))

        # reward loss
        reward_dist = td.Independent(predicted_rewards.dist(), 1)
        reward_loss = -torch.mean(reward_dist.log_prob(rewards))

        # discount loss
        # discount_dist = td.Independent(predicted_discounts.dist(), 1)
        # discount_loss = -torch.mean(discount_dist.log_prob(nonterminals))  # noqa

        # KL loss
        kl_prior = kl_divergence(
            td.Independent(posteriors.dist(logits=posteriors.logits.detach()), 1),
            td.Independent(priors.dist(), 1),
        )
        kl_posterior = kl_divergence(
            td.Independent(posteriors.dist(), 1),
            td.Independent(priors.dist(logits=priors.logits.detach()), 1),
        )
        kl_loss = self.kl_alpha * torch.mean(kl_prior) + (
            1.0 - self.kl_alpha
        ) * torch.mean(kl_posterior)

        return (
            image_loss + reward_loss + self.kl_beta * kl_loss,
            image_loss,
            reward_loss,
            kl_loss,
            image_dist,
        )

    def zero_grad(self) -> None:
        self.world_model_optimizer.zero_grad()

    def loss_fn(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        # nonterminals: torch.Tensor
    ):
        return self.representation_loss(observations, actions, rewards)

    def step(self):
        self.world_model_optimizer.step()

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ):
        loss, image_loss, reward_loss, kl_loss, image_dist = self.loss_fn(
            observations, actions, rewards
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.world_model.parameters(), self.grad_clip
        )
        self.step()
        return loss, image_loss, reward_loss, kl_loss, image_dist
