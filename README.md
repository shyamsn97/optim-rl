# optim-rl -- RL algorithms implemented as optimizers

**Note: This library is experimental and currently under development**

`optim-rl` is an RL framework where algorithms are implemented and interacted with like torch optimizers

### Example:

```python
from optimrl.policy import GymPolicy
from optimrl.algorithms.reinforce import ReinforceOptimizer

class CartPolePolicy(GymPolicy):
    def __init__(self):
        super().__init__()
        self.input_dims = 4
        self.output_dims = 2

        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2, bias=False)
        )

    def forward(self, obs: Any) -> Dict[str, Any]:
        logits = self.net(obs)
        dist = td.Categorical(logits=logits)
        action = dist.sample()
        return {"action":action, "dist":dist}

    def act(self, obs: Any):
        out = self.forward(obs)
        return out["action"].item(), out

def reinforce_rollout(
    policy: GymPolicy, env_name: str = None, env=None, env_creation_fn=None, seed = 0
):
    if env_name is None and env is None:
        raise ValueError("env_name or env must be provided!")
    if env is None:
        if env_creation_fn is None:
            env_creation_fn = gym.make
        env = env_creation_fn(env_name)
    env.seed(seed)
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

    return np.array(observations), np.array(actions), np.array(rewards)

policy = CartPolePolicy()
optimizer = ReinforceOptimizer(policy, lr=0.001)
observations, actions, rewards = optimizer.rollout(reinforce_rollout, env_name = "CartPole-v1")

torch_observations = torch.from_numpy(observations).to(policy.device)
torch_actions = torch.from_numpy(actions).float().to(policy.device)
torch_rewards = torch.from_numpy(rewards).float().to(policy.device)

optimizer.zero_grad()
loss = optimizer.loss_fn(torch_observations, torch_actions, torch_rewards)
loss.backward()
torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
optimizer.step()
```

### Install

#### Installing from source
```bash
$ git clone git@github.com:shyamsn97/optim-rl.git
$ cd optim-rl
$ python setup.py install
```
