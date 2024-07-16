from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from optimrl.optimizer import LossFunction, RLOptimizer


def get_returns_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    normalize_returns: bool = False,
    normalize_advantages: bool = True,
):
    with torch.no_grad():
        returns = torch.zeros_like(rewards)
        num_steps = returns.shape[0]

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                R = torch.zeros_like(rewards[t])
            else:
                R = returns[t + 1]
            returns[t] = rewards[t] + (1.0 - dones[t]) * R * gamma

        if normalize_returns:
            # normalize over num_steps
            returns = (returns - returns.mean(dim=0, keepdim=True)) / returns.std(
                dim=0, keepdim=True
            )

        advantages = returns - values.detach()
        if normalize_advantages:
            advantages = (
                advantages - advantages.mean(dim=0, keepdim=True)
            ) / advantages.std(dim=0, keepdim=True)
        return returns, advantages


class PPOLossFunction(LossFunction):
    def __init__(
        self,
        vf_coef: float = 0.5,
        ent_coef: float = 0.001,
        clip_ratio: float = 0.2,
        clip_vloss: bool = True,
    ):
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_ratio = clip_ratio
        self.clip_vloss = clip_vloss

    def __call__(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        values: Optional[torch.Tensor],
        returns: Optional[torch.Tensor],
        advantages: Optional[torch.Tensor],
        old_values: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
    ):
        logratio = log_probs - old_log_probs
        ratio = logratio.exp()

        # pgloss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # value loss
        if self.clip_vloss and old_values is not None:
            v_loss_unclipped = (values - returns) ** 2
            v_clipped = old_values.detach() + torch.clamp(
                values - old_values.detach(),
                -self.clip_ratio,
                self.clip_ratio,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((values - returns) ** 2).mean()

        if entropy is not None:
            entropy_loss = entropy.mean()
        else:
            entropy = 0.0
        loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss
        return loss, {
            "pg_loss": pg_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "v_loss": v_loss.item(),
        }


class RolloutGenerator:
    def __init__(
        self,
        rollouts,
        num_minibatches: int,
        gamma: float,
        norm_returns: bool,
        norm_advantages: bool,
    ):
        self.rollouts = rollouts
        self.num_steps = rollouts["obs"].shape[0]
        self.num_envs = rollouts["obs"].shape[1]
        self.batch_size = self.num_envs * self.num_steps
        self.num_minibatches = num_minibatches
        self.gamma = gamma
        self.norm_returns = norm_returns
        self.norm_advantages = norm_advantages
        self.minibatch_size = int(self.batch_size // self.num_minibatches)

        self.rewards = rollouts["rewards"].view(self.num_steps, self.num_envs).detach()
        self.np_rewards = self.rewards.detach().cpu().numpy()

        self.dones = rollouts["dones"].view(self.num_steps, self.num_envs).detach()
        self.old_values = (
            torch.stack(rollouts["policy_outs"]["values"])
            .detach()
            .view(self.num_steps, self.num_envs)
        )
        self.old_log_probs = torch.stack(rollouts["policy_outs"]["log_probs"]).detach()
        self.observations = rollouts["obs"].detach()
        self.actions = rollouts["actions"].detach()

        with torch.no_grad():
            self.returns, self.advantages = get_returns_advantages(
                rewards=self.rewards,
                values=self.old_values,
                dones=self.dones,
                gamma=self.gamma,
                normalize_returns=self.norm_returns,
                normalize_advantages=self.norm_advantages,
            )
            # flatten stuff

        self.rewards = self.rewards.view(-1)
        self.dones = self.dones.view(-1)
        self.old_values = self.old_values.view(-1)

        self.observations = torch.flatten(self.observations, 0, 1)
        self.old_log_probs = torch.flatten(self.old_log_probs, 0, 1)
        self.actions = torch.flatten(self.actions, 0, 1)

        self.returns = self.returns.view(
            self.batch_size,
        )
        self.advantages = self.advantages.view(
            self.batch_size,
        )

    def create(self):
        b_inds = np.arange(self.batch_size)
        np.random.shuffle(b_inds)
        for start in range(0, self.batch_size, self.minibatch_size):
            end = start + self.minibatch_size
            mb_inds = b_inds[start:end]
            yield (
                self.observations[mb_inds],
                self.actions[mb_inds],
                self.old_log_probs[mb_inds],
                self.old_values[mb_inds],
                self.returns[mb_inds],
                self.advantages[mb_inds],
            )


class RecurrentRolloutGenerator:
    def __init__(
        self,
        rollouts,
        num_minibatches: int,
        gamma: float,
        norm_returns: bool,
        norm_advantages: bool,
    ):
        self.rollouts = rollouts
        self.num_steps = rollouts["obs"].shape[0]
        self.num_envs = rollouts["obs"].shape[1]
        self.num_minibatches = num_minibatches
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.envsperbatch = self.num_envs // self.num_minibatches

        self.gamma = gamma
        self.norm_returns = norm_returns
        self.norm_advantages = norm_advantages

        self.rewards = rollouts["rewards"].view(self.num_steps, self.num_envs).detach()
        self.np_rewards = self.rewards.detach().cpu().numpy()

        self.dones = rollouts["dones"].view(self.num_steps, self.num_envs).detach()
        self.old_values = (
            torch.stack(rollouts["policy_outs"]["values"])
            .detach()
            .view(self.num_steps, self.num_envs)
        )
        self.old_log_probs = torch.stack(rollouts["policy_outs"]["log_probs"]).detach()
        self.observations = rollouts["obs"].detach()
        self.actions = rollouts["actions"].detach()

        with torch.no_grad():
            self.returns, self.advantages = get_returns_advantages(
                rewards=self.rewards,
                values=self.old_values,
                dones=self.dones,
                gamma=self.gamma,
                normalize_returns=self.norm_returns,
                normalize_advantages=self.norm_advantages,
            )
            # flatten stuff

        # self.rewards = self.rewards.view(-1)
        # self.dones = self.dones.view(-1)
        # self.old_values = self.old_values.view(-1)

        # self.observations = torch.flatten(self.observations, 0, 1)
        # self.old_log_probs = torch.flatten(self.old_log_probs, 0, 1)
        # self.actions = torch.flatten(self.actions, 0, 1)

        # self.returns = self.returns.view(
        #     self.batch_size,
        # )
        # self.advantages = self.advantages.view(
        #     self.batch_size,
        # )

    def create(self):
        envids = np.arange(self.num_envs)
        np.random.shuffle(envids)
        for start in range(0, self.num_envs, self.envsperbatch):
            end = start + self.envsperbatch
            mb_inds = envids[start:end]
            yield (
                self.observations[:, mb_inds],  # num_steps x num_envs x ...
                torch.flatten(self.actions[:, mb_inds], 0, 1),
                torch.flatten(self.old_log_probs[:, mb_inds], 0, 1),
                self.old_values[:, mb_inds].view(
                    self.minibatch_size,
                ),
                self.returns[:, mb_inds].view(
                    self.minibatch_size,
                ),
                self.advantages[:, mb_inds].view(
                    self.minibatch_size,
                ),
            )


class PPOOptimizer(RLOptimizer):
    def __init__(
        self,
        policy,
        loss_fn,
        num_minibatches: int = 4,
        pi_lr: float = 0.0002,
        n_updates: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 0.5,
        norm_returns: bool = False,
        norm_advantages: bool = True,
        recurrent: bool = False,
    ):
        super().__init__(policy, loss_fn)
        self.num_minibatches = num_minibatches
        self.pi_lr = pi_lr
        self.n_updates = n_updates
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.norm_returns = norm_returns
        self.norm_advantages = norm_advantages
        self.recurrent = recurrent

        # setup optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.pi_lr, eps=1e-5
        )

    def step(
        self,
        rollouts,
    ):
        if self.recurrent:
            rollout_generator = RecurrentRolloutGenerator(
                rollouts,
                num_minibatches=self.num_minibatches,
                gamma=self.gamma,
                norm_returns=self.norm_returns,
                norm_advantages=self.norm_advantages,
            )
        else:
            rollout_generator = RolloutGenerator(
                rollouts,
                num_minibatches=self.num_minibatches,
                gamma=self.gamma,
                norm_returns=self.norm_returns,
                norm_advantages=self.norm_advantages,
            )

        for _ in range(self.n_updates):
            gen = rollout_generator.create()
            for (
                b_obs,
                b_actions,
                b_old_log_probs,
                b_old_values,
                b_returns,
                b_advantages,
            ) in gen:
                self.optimizer.zero_grad()
                out = self.policy.train_forward(b_obs)
                assert "dist" in out
                log_probs = (
                    out["dist"]
                    .log_prob(b_actions)
                    .view(rollout_generator.minibatch_size, -1)
                    .squeeze(dim=-1)
                )
                entropy = out["entropy"].view(
                    rollout_generator.minibatch_size,
                )
                values = out["values"].view(
                    rollout_generator.minibatch_size,
                )
                loss, stats = self.loss_fn(
                    log_probs=log_probs,
                    old_log_probs=b_old_log_probs,
                    values=values,
                    old_values=b_old_values,
                    returns=b_returns,
                    advantages=b_advantages,
                    entropy=entropy,
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        return loss.item(), rollout_generator.np_rewards, stats
