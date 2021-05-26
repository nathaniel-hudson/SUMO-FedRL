import argparse
import fluri.trainer.ray.fed_agent as fedrl
import fluri.trainer.ray.multi_agent as marl
import fluri.trainer.ray.single_agent as sarl
import matplotlib.pyplot as plt
import pandas as pd
import ray
import seaborn as sns

from collections import defaultdict
from datetime import datetime
from fluri.sumo.single_agent_env import SinglePolicySumoEnv
from fluri.sumo.multi_agent_env import MultiPolicySumoEnv
from gym import Space
from os.path import join
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from typing import Dict, Tuple


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--kind", default="marl",
                        choices=["marl", "sarl", "fedrl"], type=str)
    args = parser.parse_args()
    return args


def get_policies() -> Tuple[Space, Space]:
    dummy_env = MultiPolicySumoEnv(config={
        "gui": False,
        "net-file": join("configs", "two_inter", "two_inter.net.xml")
    })
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    return {
        agent_id: (PPOTorchPolicy, obs_space, act_space, {"gamma": 0.95})
        for agent_id in dummy_env._observe()
    }


def load_agent(
    checkpoint: str,
    env_config: Dict=None,
    kind: str="marl"
) -> PPOTrainer:
    env_config = DEFAULT_ENV_CONFIG if env_config is None else env_config

    if kind.lower() == "marl":
        policies = get_policies()
        agent = PPOTrainer(env=MultiPolicySumoEnv, config={
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": lambda agent_id: agent_id,
            },
            "lr": 0.001,
            # "num_gpus": 0,
            "num_workers": 0,  # NOTE: For some reason, this *needs* to be 0.
            "framework": "torch",
            "log_level": "ERROR",
            "env_config": env_config,
        })
        agent.restore(checkpoint)
        return agent

    elif kind.lower() == "sarl":
        policies = get_policies()
        agent = PPOTrainer(env=SinglePolicySumoEnv, config={
            "lr": 0.001,
            # "num_gpus": 0,
            "num_workers": 0,  # NOTE: For some reason, this *needs* to be 0.
            "framework": "torch",
            "log_level": "ERROR",
            "env_config": env_config,
        })
        agent.restore(checkpoint)
        return agent

    elif kind.lower() == "fedrl":
        raise NotImplemented(
            "Support for federated agent is not yet supported.")

    else:
        raise ValueError("Invalid value for parameter 'kind'.")


def marl_step(env, obs, agent) -> None:
    action = {agent_id: agent.compute_action(agent_obs, policy_id=agent_id)
              for agent_id, agent_obs in obs.items()}
    next_obs, reward, done, info = env.step(action)
    done = next(iter(done.values()))
    return (next_obs, reward, done, info)


def rand_step() -> None:
    pass


def sarl_step(env, obs, agent) -> None:
    action = agent.compute_action(obs)
    next_obs, reward, done, info = env.step(action)
    return (next_obs, reward, done, info)


def eval_step() -> None:
    # TODO
    pass


def eval(n_episodes: int=1, kind: str="marl") -> None:

    ray.init()
    if kind == "marl":
        env = MultiPolicySumoEnv({
            "gui": False,
            "net-file": join("configs", "two_inter", "two_inter.net.xml"),
            "rand_routes_on_reset": True,
        })
        agent = load_agent(join("out", "models", "MARL-ray",
                                "checkpoint_100", "checkpoint-100"))

    elif kind == "sarl":
        env = SinglePolicySumoEnv({
            "gui": False,
            "net-file": join("configs", "two_inter", "two_inter.net.xml"),
            "rand_routes_on_reset": True
        })
        agent = load_agent(join("out", "models", "SARL-ray",
                                "checkpoint_100", "checkpoint-100"), kind=kind)

    # STEP 2: Run through the simulation.
    eval_data = defaultdict(list)
    last_reward = defaultdict(float)
    for ep in range(1, n_episodes+1):
        obs = env.reset()
        done, step = False, 0
        while not done:

            if kind == "marl":
                obs, reward, done, _ = marl_step(env, obs, agent)
                for tls_id in reward:
                    tls_reward = reward[tls_id]
                    eval_data["kind"].append(kind)
                    eval_data["tls_id"].append(tls_id)
                    eval_data["episode"].append(ep)
                    eval_data["cum_reward"].append(
                        tls_reward + last_reward[ep-1, tls_id])
                    eval_data["reward"].append(tls_reward)
                    eval_data["step"].append(step)
                    last_reward[ep, tls_id] = eval_data["cum_reward"][-1]

            elif kind == "sarl":
                obs, reward, done, _ = sarl_step(env, obs, agent)
                eval_data["kind"].append(kind)
                eval_data["episode"].append(ep)
                eval_data["cum_reward"].append(reward + last_reward[ep-1])
                eval_data["reward"].append(reward)
                eval_data["step"].append(step)
                last_reward[ep] = eval_data["cum_reward"][-1]

            elif kind == "fedrl":
                raise NotImplemented("Federated agent is not yet supported.")

            step += 1

    env.close()
    ray.shutdown()
    return pd.DataFrame.from_dict(eval_data)


if __name__ == "__main__":
    args = get_args()
    kind = args.kind
    del args
    df = eval(5, kind=kind)
    sns.lineplot(data=df, x="episode", y="cum_reward", hue="tls_id")
    plt.grid(linestyle="--")
    plt.legend(fancybox=False, edgecolor="black")
    plt.show()
