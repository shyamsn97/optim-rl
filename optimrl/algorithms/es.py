import copy
from typing import Tuple

import gym
import numpy as np
import scipy.stats as ss
import torch
import torch.optim as optim
from einops import repeat
from torch.distributions.normal import Normal
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from optimrl.optimizer import Loss, RLOptimizer
from optimrl.policy import GymPolicy


def es_rollout(policy: GymPolicy, env_name: str = None, env=None, env_creation_fn=None):
    if env_name is None and env is None:
        raise ValueError("env_name or env must be provided!")
    if env is None:
        if env_creation_fn is None:
            env_creation_fn = gym.make
        env = env_creation_fn(env_name)
    done = False
    rewards = []
    observation = env.reset()
    with torch.no_grad():
        while not done:
            action, _ = policy.act(
                torch.from_numpy(observation).unsqueeze(0).to(policy.device)
            )
            next_observation, reward, done, info = env.step(action)

            rewards.append(reward)

            observation = next_observation
    env.close()
    return np.array(rewards)


def normalized_rank(rewards):
    """
    Rank the rewards and normalize them.
    """
    ranked = ss.rankdata(rewards)
    norm = (ranked - 1) / (len(ranked) - 1)
    norm -= 0.5
    return norm


class ESLoss(Loss):
    def __init__(
        self,
        rewards,
        epsilon: torch.Tensor,
        mean: torch.Tensor,
        sigma: float,
        l2_decay: float,
        policy: GymPolicy,
    ):
        self.rewards = rewards
        self.epsilon = epsilon
        self.mean = mean
        self.sigma = sigma
        self.l2_decay = l2_decay
        self.policy = policy
        self.centered_fitnesses = (
            torch.from_numpy(normalized_rank(rewards))
            .float()
            .unsqueeze(0)
            .to(epsilon.device)
        )

    def backward(self):
        with torch.no_grad():
            # (1 x pop_size) mm (pop_size x param_size)
            grad = (
                -1.0
                * (
                    torch.mm(self.centered_fitnesses, self.epsilon)
                    / (self.centered_fitnesses.shape[-1] * self.sigma)
                ).squeeze()
            )
            grad = (grad + self.mean * self.l2_decay).squeeze()  # L2 Decay
        index = 0
        for parameter in self.policy.parameters():
            size = np.prod(parameter.shape)
            parameter.grad = grad[index : index + size].view(parameter.shape)
            index += size


class ESOptimizer(RLOptimizer):
    def __init__(
        self,
        policy: GymPolicy,
        population_size: int = 10,
        sigma: float = 0.2,
        l2_decay: float = 0.005,
        lr: float = 0.02,
    ):
        self.policy = policy
        self.population_size = population_size
        self.lr = lr

        self.sigma = sigma
        self.noise_dist = Normal(0, sigma)

        self.l2_decay = l2_decay
        self.setup_optimizer()

        self.num_parameters = parameters_to_vector(policy.parameters()).size(-1)
        self.policies = [
            copy.deepcopy(policy).to(torch.device("cpu"))
            for _ in range(population_size)
        ]

        # to be populated later
        self.perturbed_params = None
        self.mean = None
        self.epsilon = None

    def setup_optimizer(self):
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def loss_fn(
        self,
        rewards: np.array,
        epsilon: torch.Tensor,
        mean: torch.Tensor,
    ):
        epsilon = epsilon.to(self.policy.device)
        mean = mean.to(epsilon.device)
        return ESLoss(rewards, epsilon, mean, self.sigma, self.l2_decay, self.policy)

    def step(self):
        self.optimizer.step()

    def get_perturbed_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            mean = parameters_to_vector(self.policy.parameters()).to(
                torch.device("cpu")
            )
            epsilon = self.noise_dist.sample(
                [int(self.population_size / 2), self.num_parameters]
            )
            epsilon = torch.cat([epsilon, -1.0 * epsilon], dim=0)
            params = repeat(mean, "w -> b w", b=self.population_size) + epsilon
        return params, epsilon, mean

    def rollout(self, rollout_fn=None, pool=None, *args, **kwargs):
        if rollout_fn is None:
            rollout_fn = es_rollout
        perturbed_params, epsilon, mean = self.get_perturbed_params()
        for i in range(len(self.policies)):
            vector_to_parameters(perturbed_params[i], self.policies[i].parameters())
        if pool is None:
            return (
                [
                    np.sum(rollout_fn(policy, *args, **kwargs)["rewards"])
                    for policy in self.policies
                ],
                epsilon,
                mean,
            )
        rewards = list(
            pool.starmap([tuple(policy, *args, **kwargs) for policy in self.policies])
        )
        return (
            np.array([np.sum(r["rewards"]) for r in rewards]),
            epsilon,
            mean,
        )
