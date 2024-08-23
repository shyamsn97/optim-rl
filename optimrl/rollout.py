from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
    renders: Optional[Any] = None


@dataclass
class Rollout:
    steps: List[EnvStep]
    last_obs: Any
    last_done: Any
    last_policy_out: Dict[str, Any]
    num_envs: int
    _stats: Any = None
    episodic_return: Any = None
    renders: Any = None

    @property
    def stats(self):
        if self._stats is None:
            sum_rewards = np.mean(np.sum(np.array([s.reward for s in self.steps]), 0))
            self._stats = {"sum_rewards": sum_rewards}
        return self._stats

    def __len__(self) -> int:
        return len(self.steps)

    @classmethod
    def rollout(
        cls,
        envs,
        policy,
        num_steps: int,
        seed: int,
        policy_out={},
        evaluate: bool = False,
        render: bool = False,
    ) -> Rollout:
        num_envs = envs.num_envs
        next_obs, infos = envs.reset(seed=seed)
        env_steps = []
        episodic_return = None
        renders = []
        action = None
        policy.eval()
        for step in range(num_steps):
            obs = next_obs
            with torch.no_grad():
                action, policy_out = policy.act(obs, policy_out=policy_out)
                if render:
                    renders.append(envs.envs[0].render())
                next_obs, reward, terminations, truncations, infos = envs.step(action)
                done = np.logical_or(terminations, truncations)
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
                if evaluate:
                    if done:
                        break
        with torch.no_grad():
            _, last_policy_out = policy.act(next_obs, policy_out=policy_out)
        return cls(
            steps=env_steps,
            last_obs=next_obs,
            last_done=done,
            last_policy_out=last_policy_out,
            num_envs=num_envs,
            episodic_return=episodic_return,
            renders=renders,
        )

    def to_torch(self, device) -> Dict[str, Any]:
        steps = self.steps
        num_envs = self.num_envs
        num_steps = len(self)
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

        last_policy_out = {}
        for k in self.last_policy_out:
            last_policy_out[k] = self.last_policy_out[k]
            if isinstance(last_policy_out[k], torch.Tensor):
                last_policy_out[k] = last_policy_out[k].to(device)

        last_obs = torch.Tensor(self.last_obs).to(device)
        last_done = torch.Tensor(self.last_done).to(device)

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "infos": infos,
            "policy_outs": policy_outs,
            "last_obs": last_obs,
            "last_done": last_done,
            "last_policy_out": last_policy_out,
            "num_steps": len(self),
            "num_envs": self.num_envs,
        }
