# Smart Edge-Enabled trAffic Lights (SEAL)
Implementation of Federated Reinforcement Learning for traffic control using SUMO and a real-world testbed.

The goal of this project is to provide a platform for *simple* single- and multi-agent reinforcement learning (RL) and federated reinforcement learning (FedRL) for smart traffic light control. The traffic simulator we use is SUMO.

## Multi-Agent FedRL
We consider a multi-agent approach for FedRL using the OpenAI Gym interface. The gym for this scenario is defined as follows:
```
class MultiAgentEnv(gym.Env):
    def step(self, action_n: List[Any]) -> Tuple:
        obs_n = list()
        reward_n = list()
        done_n = list()
        info_n = {"n": []}
        # ...
        return obs_n, reward_n, done_n, info_n
```
For the multi-agent environment, we will use the `MultiAgentEnv` class provided by the RlLib API.
> https://docs.ray.io/en/master/rllib-env.html#multi-agent-and-hierarchical

Go here for `netgenerate` documentation:
> https://sumo.dlr.de/docs/netgenerate.html