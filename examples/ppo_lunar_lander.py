from typing import Any, Dict, Optional  # noqa

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
from optimrl.algorithms.ppo import PPOLossFunction, PPOOptimizer
from optimrl.utils import get_tensorboard_logger
from tqdm import tqdm

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def ppo_rollout(
    policy,
    device
):
    SEED = None
    env = gym.make("LunarLander-v2")
    max_steps = 10000
    done = False
    observations, actions, rewards, logits, log_probs, values, terminals = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    count = 0
    observation, _ = env.reset(seed=SEED)
    with torch.no_grad():
        while not done:
            obs = torch.from_numpy(observation).unsqueeze(0).to(device)
            out = policy(obs)
            action = out["actions"].item()
            next_observation, reward, done, truncated, info = env.step(action)

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            logits.append(out["logits"].squeeze().detach().cpu().numpy())
            log_probs.append(out["log_probs"].squeeze().detach().cpu().numpy())
            values.append(out["values"].squeeze().detach().cpu().numpy())
            terminals.append(done)

            observation = next_observation
            if count == max_steps:
                done = True
            count += 1
    env.close()

    return {
        "observations": np.array(observations),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "log_probs": np.array(log_probs),
        "logits":np.array(logits),
        "values": np.array(values),
        "terminals": np.array(terminals),
    }



class PPOCategoricalPolicy(nn.Module):
    def __init__(
        self,
        obs_dims,
        act_dims
    ):
        super().__init__()
        self.obs_dims = obs_dims
        self.act_dims = act_dims

        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_dims, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.act_dims),  std=0.01)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_dims, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

    def forward(self, obs: torch.Tensor):
        logits = self.actor(obs)
        values = self.critic(obs)
        dist = td.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return {"actions":actions, "values":values, "log_probs":log_probs, "dist":dist, "logits":logits}

if __name__ == "__main__":
    writer = get_tensorboard_logger("logs/PPOLunarLander")

    bar = tqdm(np.arange(50000))
    device = torch.device("cpu")

    env = gym.make("LunarLander-v2")
    observation_dims = env.observation_space.shape[-1]
    action_dims = env.action_space.n

    policy = PPOCategoricalPolicy(observation_dims, action_dims)
    loss_fn = PPOLossFunction(entropy_weight=0.01, norm_advantages=False)
    optimizer = PPOOptimizer(
        policy=policy,
        loss_fn=loss_fn,
        pi_lr=0.002,
        n_updates=4
    )

    rewards = []
    for i in bar:
        with torch.no_grad():
            rollout = ppo_rollout(policy, device)
        sum_reward = np.sum(rollout["rewards"])
        loss = optimizer.update([rollout], device)
        rewards.append(sum_reward)
        # bar.set_description(f"Loss: {loss}, Sum reward: {np.mean(rewards[-20:])}")
        if i % 500 == 0:
            print(f"Loss: {loss}, Sum reward: {np.mean(rewards[-25:])}")