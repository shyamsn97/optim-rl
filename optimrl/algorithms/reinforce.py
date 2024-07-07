import gym
import numpy as np
import torch
import torch.optim as optim

from optimrl.optimizer import RLOptimizer
from optimrl.policy import GymPolicy


def reinforce_rollout(
    policy: GymPolicy, env_name: str = None, env=None, env_creation_fn=None
):
    if env_name is None and env is None:
        raise ValueError("env_name or env must be provided!")
    if env is None:
        if env_creation_fn is None:
            env_creation_fn = gym.make
        env = env_creation_fn(env_name)
    done = False
    observations, actions, rewards = (
        [],
        [],
        [],
    )
    observation = env.reset()
    with torch.no_grad():
        while not done:
            action, out = policy.act(
                torch.from_numpy(observation).unsqueeze(0).to(policy.device)
            )
            next_observation, reward, done, info = env.step(action)

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)

            observation = next_observation
    env.close()
    return np.array(observations), np.array(actions), np.array(rewards)


class ReinforceOptimizer(RLOptimizer):
    def __init__(self, policy: GymPolicy, lr: float = 0.01, gamma: float = 0.99):
        self.policy = policy
        self.lr = lr
        self.gamma = gamma
        self.setup_optimizer()

    def discount_rewards(self, rews: torch.Tensor) -> torch.Tensor:
        n = len(rews)
        rtgs = torch.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + self.gamma * (rtgs[i + 1] if i + 1 < n else 0)
        return rtgs

    def setup_optimizer(self):
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def loss_fn(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        discount_rewards: bool = True,
    ):
        with torch.no_grad():
            torch_rewards = (
                torch.zeros_like(rewards, device=rewards.device) + rewards
            )  # copy
            if discount_rewards:
                torch_rewards = self.discount_rewards(torch_rewards)
            torch_rewards = torch_rewards - torch.mean(torch_rewards)
            torch_rewards = torch_rewards / torch.std(torch_rewards) + 1e-10

        policy_out = self.policy(observations)
        dist = policy_out["dist"]
        log_probs = dist.log_prob(actions)
        loss = -1 * torch.sum(torch_rewards * log_probs)
        return loss

    def step(self):
        self.optimizer.step()

    def rollout_fn(self):
        return reinforce_rollout

    def rollout(
        self, rollout_fn=None, pool=None, num_rollouts: int = 1, *args, **kwargs
    ):
        """
        Optional default rollout_fn for the algorithm.
        """
        if rollout_fn is None:
            rollout_fn = reinforce_rollout
        return rollout_fn(self.policy, *args, **kwargs)
