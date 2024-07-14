from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def convert_to_torch(rollout, device):
    steps = rollout.steps
    num_envs = rollout.num_envs
    num_steps = len(rollout)
    # num_envs x ...
    obs_shape = steps[0].observation.shape[1:]

    if len(steps[0].action.shape) <= 1:
        action_shape = ()
    else:
        action_shape = steps[0].action.shape[1:]

    obs = torch.zeros((num_steps, num_envs) + obs_shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + action_shape).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)

    infos = []
    policy_outs = {}

    for i, step in enumerate(steps):
        obs[i, :] = torch.from_numpy(step.observation).to(device)
        actions[i] = torch.Tensor(step.action).to(device)
        rewards[i] = torch.from_numpy(step.reward).to(device)
        dones[i] = torch.from_numpy(step.done).to(device)
        infos.append(step.info)

        for k in step.policy_output:
            if k not in policy_outs:
                policy_outs[k] = []
            policy_outs[k].append(step.policy_output[k])

    for k in policy_outs:
        if isinstance(policy_outs[k], torch.Tensor):
            policy_outs[k] = torch.stack(policy_outs[k]).to(device)

    last_obs = torch.Tensor(rollout.last_obs).to(device)
    last_done = torch.Tensor(rollout.last_done).to(device)

    return {
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "infos": infos,
        "policy_outs": policy_outs,
        "last_obs": last_obs,
        "last_done": last_done,
    }


def get_returns_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    last_done: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
):
    with torch.no_grad():
        num_steps = rewards.shape[0]
        device = rewards.device
        last_value = last_value.view(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - last_done
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values
        return returns, advantages


class PPOLossFunction:
    def __init__(
        self,
        vf_coef: float = 0.5,
        ent_coef: float = 0.001,
        clip_ratio: float = 0.2,
        clip_vloss: bool = True,
        norm_advantages: bool = True,
    ):
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_ratio = clip_ratio
        self.clip_vloss = clip_vloss
        self.norm_advantages = norm_advantages

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
        if self.norm_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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

        entropy_loss = entropy.mean()
        loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss
        return loss, {
            "pg_loss": pg_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "v_loss": v_loss.item(),
        }


class PPOOptimizer:
    def __init__(
        self,
        policy,
        loss_fn,
        num_minibatches: int = 4,
        pi_lr: float = 2.5e-4,
        n_updates: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 0.5,
    ):
        self.policy = policy
        self.loss_fn = loss_fn
        self.num_minibatches = num_minibatches
        self.pi_lr = pi_lr
        self.n_updates = n_updates
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm

        # setup optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.pi_lr, eps=1e-5
        )

    def step(
        self,
        rollouts,
        device,
    ):
        num_envs = rollouts.num_envs
        num_steps = len(rollouts)
        batch_size = num_envs * num_steps
        minibatch_size = int(batch_size // self.num_minibatches)

        rollouts = convert_to_torch(rollouts, device)
        rewards = rollouts["rewards"].view(num_steps, num_envs)
        np_rewards = rewards.detach().cpu().numpy()

        dones = rollouts["dones"]
        old_values = (
            torch.stack(rollouts["policy_outs"]["values"])
            .detach()
            .view(num_steps, num_envs)
        )

        old_log_probs = torch.stack(rollouts["policy_outs"]["log_probs"]).detach()
        observations = rollouts["obs"]
        actions = rollouts["actions"]

        with torch.no_grad():
            last_value = self.policy(rollouts["last_obs"])["values"]
            last_done = rollouts["last_done"]
            returns, advantages = get_returns_advantages(
                rewards=rewards,
                values=old_values,
                dones=dones,
                last_value=last_value,
                last_done=last_done,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
            # flatten stuff

        rewards = rewards.view(-1)
        dones = dones.view(-1)
        old_values = old_values.view(-1)
        last_value = last_value.view(-1)
        last_done = last_done.view(-1)

        observations = torch.flatten(observations, 0, 1)
        old_log_probs = torch.flatten(old_log_probs, 0, 1)
        actions = torch.flatten(actions, 0, 1)

        returns = returns.view(
            batch_size,
        )
        advantages = advantages.view(
            batch_size,
        )

        #         print("observations", observations.shape)
        #         print("old_log_probs", old_log_probs.shape)
        #         print("actions", actions.shape)
        #         print("returns", returns.shape)
        #         print("advantages", advantages.shape)
        #         print("rewards", rewards.shape)
        #         print("dones", dones.shape)

        b_inds = np.arange(batch_size)
        for _ in range(self.n_updates):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                self.optimizer.zero_grad()
                out = self.policy(observations[mb_inds], actions[mb_inds])
                log_probs = out["log_probs"].view(minibatch_size, -1)
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
