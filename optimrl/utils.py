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
