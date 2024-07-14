from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from optimrl.optimizer import LossFunction, RLOptimizer


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

    def get_returns_advantages(
        self,
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

        # setup optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.pi_lr, eps=1e-5
        )

    def step(
        self,
        rollouts,
    ):
        num_steps = rollouts["obs"].shape[0]
        num_envs = rollouts["obs"].shape[1]
        batch_size = num_envs * num_steps
        minibatch_size = int(batch_size // self.num_minibatches)

        rewards = rollouts["rewards"].view(num_steps, num_envs).detach()
        np_rewards = rewards.detach().cpu().numpy()

        dones = rollouts["dones"].view(num_steps, num_envs).detach()
        old_values = (
            torch.stack(rollouts["policy_outs"]["values"])
            .detach()
            .view(num_steps, num_envs)
        )
        old_log_probs = torch.stack(rollouts["policy_outs"]["log_probs"]).detach()
        observations = rollouts["obs"].detach()
        actions = rollouts["actions"].detach()

        with torch.no_grad():
            returns, advantages = self.loss_fn.get_returns_advantages(
                rewards=rewards,
                values=old_values,
                dones=dones,
                gamma=self.gamma,
                normalize_returns=self.norm_returns,
                normalize_advantages=self.norm_advantages,
            )
            # flatten stuff

        rewards = rewards.view(-1)
        dones = dones.view(-1)
        old_values = old_values.view(-1)

        observations = torch.flatten(observations, 0, 1)
        old_log_probs = torch.flatten(old_log_probs, 0, 1)
        actions = torch.flatten(actions, 0, 1)

        returns = returns.view(
            batch_size,
        )
        advantages = advantages.view(
            batch_size,
        )

        b_inds = np.arange(batch_size)
        for _ in range(self.n_updates):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                self.optimizer.zero_grad()
                out = self.policy(observations[mb_inds], actions[mb_inds])
                log_probs = out["log_probs"].view(minibatch_size, -1).squeeze(dim=-1)
                entropy = out["entropy"].view(
                    minibatch_size,
                )
                values = out["values"].view(
                    minibatch_size,
                )
                loss, stats = self.loss_fn(
                    log_probs=log_probs,
                    old_log_probs=old_log_probs[mb_inds],
                    values=values,
                    old_values=old_values[mb_inds],
                    returns=returns[mb_inds],
                    advantages=advantages[mb_inds],
                    entropy=entropy,
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        return loss.item(), np_rewards, stats
