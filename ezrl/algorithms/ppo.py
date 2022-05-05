from typing import Any, Dict, Optional  # noqa

import gym
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
import torch.optim as optim

from ezrl.optimizer import RLOptimizer
from ezrl.policy import ACPolicy


def ppo_rollout(
    policy: ACPolicy, env_name: str = None, env=None, env_creation_fn=None
) -> Dict[str, np.array]:
    if env_name is None and env is None:
        raise ValueError("env_name or env must be provided!")
    if env is None:
        if env_creation_fn is None:
            env_creation_fn = gym.make
        env = env_creation_fn(env_name)
    done = False
    observations, actions, rewards, log_probs, values = ([], [], [], [], [])
    observation = env.reset()
    with torch.no_grad():
        while not done:
            obs = torch.from_numpy(observation).to(policy.device)
            action, out = policy.act(obs)
            values = out["values"]
            next_observation, reward, done, info = env.step(action)

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(out["log_probs"].detach().cpu().numpy())
            values.append(values.detach().cpu().numpy())

            observation = next_observation
    env.close()
    return {
        "observations": np.array(observations),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "log_probs": np.array(log_probs),
        "values": np.array(values),
    }


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOOptimizer(RLOptimizer):
    def __init__(
        self,
        policy: ACPolicy,
        pi_lr: float = 0.0005,
        vf_coef: float = 0.5,
        entropy_weight: float = 0.001,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        num_rollouts: int = 1,
    ):
        self.policy = policy
        self.pi_lr = pi_lr
        self.vf_coef = vf_coef
        self.entropy_weight = entropy_weight
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.num_rollouts = num_rollouts
        self.setup_optimizer()

    def discount_rewards(self, rews: torch.Tensor) -> torch.Tensor:
        n = len(rews)
        rtgs = torch.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + self.gamma * (rtgs[i + 1] if i + 1 < n else 0)
        return rtgs

    def setup_optimizer(self):
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.pi_lr)

    def calculate_advantages(
        self,
        returns: torch.Tensor,
        values: torch.Tensor,
        discount_factor: float,
        normalize: bool = False,
    ):

        adv = returns - values

        adv = adv.squeeze()

        if normalize:
            adv = (adv - adv.squeeze()) / adv.std()
        return adv

    def calculate_returns(
        self, rewards: torch.Tensor, discount_factor: float, normalize: bool = False
    ):

        returns = []
        R = 0

        for r in reversed(rewards):
            R = r + R * discount_factor
            returns.insert(0, R)

        returns = torch.tensor(returns).squeeze().to(rewards.device)

        if normalize:
            returns = (returns - returns.mean()) / returns.std()
        return returns

    def value_loss(self, values, returns):
        assert tuple(values.squeeze().size()) == tuple(returns.squeeze().size())
        return F.mse_loss(returns.squeeze(), values.squeeze()).mean()

    def actor_loss(self, log_probs, old_logprobs, advantages):
        ratio = torch.exp(log_probs.squeeze() - old_logprobs.squeeze())
        assert tuple(ratio.size()) == tuple(advantages.size())
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            * advantages
        )
        loss_pi = -(torch.min(surr1, surr2)).mean()
        return loss_pi

    def zero_grad(self):
        self.optimizer.zero_grad()

    def loss_fn(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        advantages: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):
        if returns is None:
            returns = self.calculate_returns(rewards, self.gamma, normalize=False)
        if advantages is None:
            advantages = self.calculate_advantages(
                returns, values, self.gamma, normalize=False
            )

        out = self.policy(observations, *args, **kwargs)
        dist = out["dist"]
        log_probs = self.policy.log_prob(dist, actions)
        actor_loss = self.actor_loss(log_probs, old_log_probs, advantages)
        values = self.policy.critic(observations)
        value_loss = self.value_loss(values, returns)
        loss = actor_loss + self.vf_coef * value_loss

        return loss, actor_loss, value_loss

    def step(self):
        self.optimizer.step()

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        advantages: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> Any:
        losses = []
        actor_losses = []
        value_losses = []
        for _ in range(self.train_pi_iters):
            self.zero_grad()
            loss, actor_loss, value_loss = self.loss_fn(
                observations,
                actions,
                log_probs,
                rewards,
                values,
                returns,
                advantages,
                *args,
                **kwargs
            )
            loss.backward()
            self.step()
            losses.append(loss.item())
            actor_losses.append(actor_loss.item())
            value_losses.append(value_loss.item())
        return np.array(losses), np.array(actor_losses), np.array(value_losses)

    def rollout_fn(self):
        return ppo_rollout

    def rollout(
        self, rollout_fn=None, pool=None, num_rollouts: int = 1, *args, **kwargs
    ) -> Dict[str, np.array]:
        """
        Optional default rollout_fn for the algorithm.
        """
        if rollout_fn is None:
            rollout_fn = ppo_rollout
        if pool is None:
            rollouts = [
                rollout_fn(self.policy, *args, **kwargs) for _ in range(num_rollouts)
            ]
        else:
            rollouts = list(
                pool.starmap(
                    rollout_fn, [tuple(self.policy, *args, **kwargs) for _ in range(num_rollouts)]
                )
            )
        return rollouts
