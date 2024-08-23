import os
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List  # noqa

import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from optimrl.policy import GymPolicy


def render(env: gym.Env, policy: GymPolicy):
    observation = env.reset()
    done = False
    rendereds = []
    rewards = []
    with torch.no_grad():
        while not done:
            rendered = env.render(mode="rgb_array")
            time.sleep(0.01)

            action, _ = policy.act(
                torch.from_numpy(observation).unsqueeze(0).to(policy.device)
            )
            next_observation, reward, done, _ = env.step(action)

            observation = next_observation

            rendereds.append(rendered)
            rewards.append(reward)
    env.close()
    return rendereds, rewards


def get_tensorboard_logger(
    experiment_name: str, base_log_path: str = "tensorboard_logs"
):
    log_path = "{}/{}_{}".format(base_log_path, experiment_name, datetime.now())
    train_writer = SummaryWriter(log_path, flush_secs=10)
    full_log_path = os.path.join(os.getcwd(), log_path)
    print(
        "Follow tensorboard logs with: tensorboard --logdir '{}'".format(full_log_path)
    )
    return train_writer


def flatten_list_of_dicts(
    list_of_dicts: List[Dict[str, Iterable]]
) -> Dict[str, Iterable]:
    out = {}
    for d in list_of_dicts:
        for k in d:
            if k not in out:
                out[k] = list(d[k])
            else:
                out[k].extend(d[k])
    return out


def get_returns_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    normalize_returns: bool = False,
    normalize_advantages: bool = False,
):
    """Compute generalized advantage estimate.
    rewards: a list of rewards at each step.
    values: the value estimate of the state at each step.
    episode_ends: an array of the same shape as rewards, with a 1 if the
        episode ended at that step and a 0 otherwise.
    gamma: the discount factor.
    lam: the GAE lambda parameter.
    """
    with torch.no_grad():
        T = rewards.shape[0]
        N = rewards.shape[1]
        gae_step = torch.zeros((N,))
        advantages = torch.zeros((T, N))
        values = values.detach()

        for t in reversed(range(T)):
            if t == (T - 1):
                next_value = last_value
            else:
                next_value = values[t + 1, :]
            delta = (
                rewards[t, :] + (gamma * next_value * (1 - dones[t, :])) - values[t, :]
            )
            gae_step = delta + (gamma * lam * (1 - dones[t, :]) * gae_step)
            # And store it
            advantages[t, :] = gae_step

        returns = advantages + values
        if normalize_returns:
            # normalize over num_steps
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        return returns, advantages
