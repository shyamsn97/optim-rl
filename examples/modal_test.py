from __future__ import annotations
import trainyard
import torch.nn as nn
import torch
import numpy as np
import torch.distributions as td
from typing import Optional
import modal
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
    def rollout(cls, envs, policy, num_steps: int, seed: int, device) -> Rollout:
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
                action, policy_out = policy.act(obs, policy_out, device=device)
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
            # for item in infos:
            #     if "episode" in item.keys():
            #             episodic_return = infos["episode"]["r"]
            env_steps.append(env_step)
        return cls(
            steps=env_steps,
            last_obs=next_obs,
            last_done=next_done,
            num_envs=num_envs,
            episodic_return=episodic_return,
        )


@trainyard.options(env_requirements=["torch"])
@trainyard.car
class PPOCategoricalPolicy(nn.Module):
    def __init__(
        self,
        obs_dims,
        act_dims
    ):
        super().__init__()
        self.obs_dims = obs_dims
        self.act_dims = act_dims

        def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_dims, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_dims, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, self.act_dims), std=0.01),
        )

    def forward(self, obs: torch.Tensor, actions = None):
        logits = self.actor(obs)
        values = self.critic(obs)
        dist = td.Categorical(logits=logits)
        if actions is None:
            actions = dist.sample()
        entropy = dist.entropy()
        log_probs = dist.log_prob(actions)
        return {"actions":actions, "values":values, "log_probs":log_probs, "dist":dist, "logits":logits, "entropy":entropy}

    def act(self, obs: torch.Tensor, prev_output = {}, device = None):
        if device is None:
            device = torch.device("cpu")
        with torch.no_grad():
            out = self.forward(torch.from_numpy(obs).to(device))
            return out["actions"].detach().cpu().numpy(), out

@trainyard.options(env_requirements=["gym", "gym[box2d]"])
@trainyard.car
def make_env(env_name: str, num_envs: int = 4):
    import gym

    def make_env(env_name):
        env =  gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return gym.vector.SyncVectorEnv(
        [lambda: make_env(env_name) for _ in range(num_envs)],
    )

def convert_to_torch(rollout, device):
    import torch

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
        "obs":obs,
        "actions":actions,
        "rewards":rewards,
        "dones":dones,
        "infos":infos,
        "policy_outs":policy_outs,
        "last_obs":last_obs,
        "last_done":last_done
    }

def get_returns_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    last_done: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
):
    import torch

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
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
        return returns, advantages

@trainyard.options(env_requirements=["torch"])
@trainyard.car
class PPOLossFunction:
    def __init__(
        self,
        vf_coef: float = 0.5,
        ent_coef: float = 0.001,
        clip_ratio: float = 0.2,
        clip_vloss: bool = True,
        norm_advantages: bool = True
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
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
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
        return loss, {"pg_loss":pg_loss.item(), "entropy_loss":entropy_loss.item(), "v_loss":v_loss.item()}

@trainyard.options(env_requirements=["torch"])
@trainyard.car
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
        max_grad_norm: float = 0.5
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
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.pi_lr, eps=1e-5)

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

        dones = rollouts["dones"]
        old_values = torch.stack(rollouts["policy_outs"]["values"]).detach().view(num_steps, num_envs)


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
                gae_lambda=self.gae_lambda
            )
            # flatten stuff

        np_returns = np.mean(np.sum(returns.detach().cpu().numpy(), 0))
        rewards = rewards.view(-1)
        dones = dones.view(-1)
        old_values = old_values.view(-1)
        last_value = last_value.view(-1)
        last_done = last_done.view(-1)

        observations = torch.flatten(observations, 0, 1)
        old_log_probs = torch.flatten(old_log_probs, 0,1)
        actions = torch.flatten(actions, 0,1)

        returns = returns.view(batch_size,)
        advantages = advantages.view(batch_size,)

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
                entropy = out["entropy"].view(minibatch_size,)
                values = out["values"].view(minibatch_size,)
                loss, stats = self.loss_fn(
                    log_probs=log_probs,
                    old_log_probs=old_log_probs[mb_inds],
                    values=values,
                    old_values=old_values[mb_inds],
                    returns=returns[mb_inds],
                    advantages=advantages[mb_inds],
                    entropy=entropy
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        return loss.item(), np_returns, stats

@trainyard.options(env_requirements=["tqdm", "torch"])
@trainyard.car
def main(policy, optimizer, envs, seed, num_steps, anneal_lr: bool = True, cuda: bool = False):
    from tqdm import tqdm
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    bar = tqdm(np.arange(20000))
    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    policy = policy.to(device)
    global_steps = 0
    mean_rewards = []
    for iterations in bar:
        if anneal_lr:
            frac = 1.0 - (iterations - 1.0) / 20000
            lrnow = frac * optimizer.pi_lr
            optimizer.optimizer.param_groups[0]["lr"] = lrnow

        with torch.no_grad():
            rollout = Rollout.rollout(envs, policy, num_steps, seed, device=device)
        global_steps += rollout.num_envs*len(rollout)
        loss, rewards, stats = optimizer.step(rollout, device)
        mean_rewards.append(rewards)
        max_rew = np.max(mean_rewards)
        # bar.set_description(f"Loss: {loss}, Sum reward: {np.mean(rewards[-20:])}")
        bar.set_description(f"{global_steps} -- L: {loss} S: {np.mean(mean_rewards[-50:])} M: {max_rew}")

if __name__ == "__main__":
    trainyard.set_local_trainyard_path("~/Code/trainyard/")
    # image = modal.Image.debian_slim().apt_install(["swig"])
    # engine = trainyard.engine(
    #     name="ppo-example", instance_type="A10G", backend="modal", image=image
    # )
    # with engine:
    envs = make_env("LunarLander-v2", 1)
    policy = PPOCategoricalPolicy(8, 4)
    loss_fn = PPOLossFunction(clip_ratio=0.2, ent_coef=0.01, vf_coef=0.5)
    optimizer = PPOOptimizer(policy, loss_fn, pi_lr=2.5e-4, n_updates=4, num_minibatches=4)
    main(policy, optimizer, envs, 0, 128, anneal_lr=True, cuda=False)
