from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch


@dataclass
class EnvStep:
    observation: Any
    action: Any
    reward: Any
    done: Any
    info: Dict[str, Any]
    policy_output: Dict[str, Any]
    step: int


@dataclass
class Rollout:
    steps: List[EnvStep]
    last_obs: Any
    last_done: Any
    num_envs: int
    _stats: Any = None
    episodic_return: Any = None

    @property
    def stats(self):
        if self._stats is None:
            sum_rewards = np.mean(np.sum(np.array([s.reward for s in self.steps]), 0))
            self._stats = {"sum_rewards": sum_rewards}
        return self._stats

    def __len__(self) -> int:
        return len(self.steps)

    @classmethod
    def rollout(cls, envs, policy, num_steps: int, seed: int) -> Rollout:
        num_envs = envs.num_envs
        next_obs, infos = envs.reset(seed=seed)
        next_done = np.zeros((num_envs))
        env_steps = []
        policy_out = None
        episodic_return = None
        for step in range(num_steps):
            obs = next_obs
            done = next_done
            with torch.no_grad():
                action, policy_out = policy.act(obs, policy_out)
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = np.logical_or(terminations, truncations)
            env_step = EnvStep(
                observation=obs,
                action=action,
                reward=reward,
                done=done,
                info=infos,
                policy_output=policy_out,
                step=step,
            )
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episodic_return = info["episode"]["r"]
            env_steps.append(env_step)
        return cls(
            steps=env_steps,
            last_obs=next_obs,
            last_done=next_done,
            num_envs=num_envs,
            episodic_return=episodic_return,
        )
