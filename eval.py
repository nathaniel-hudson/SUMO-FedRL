import glob
import os
import pandas as pd
import ray
import torch

from collections import defaultdict
from fluri.trainer.base import BaseTrainer
from fluri.sumo.config import *
from fluri.sumo.abstract_env import AbstractSumoEnv
from fluri.sumo.env import SumoEnv
from fluri.trainer.fed_agent import FedPolicyTrainer
from fluri.trainer.multi_agent import MultiPolicyTrainer
from fluri.trainer.single_agent import SinglePolicyTrainer
from os.path import join
from pandas import DataFrame
from typing import Dict, Tuple
from ray.rllib import agents as raygents, policy
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer

# TODO: For now, this file only supports PPO agents. Fix this later.


def load_config(netfile: str, kind: str, ranked: bool) -> Dict:
    config = {
        "env_config": {
            "gui": False,
            "net-file": netfile,
            "rand_routes_on_reset": True,
            "ranked": ranked,
        },
        "framework": "torch",
        "log_level": "ERROR",
        "lr": 0.001,
        "num_workers": 0,
    }

    dummy_env = SumoEnv(config={
        "gui": False,
        "net-file": netfile,
        "ranked": ranked,
    })
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    if kind == "sarl":
        policies = {
            agent_id: (PPOTorchPolicy, obs_space, act_space, {"gamma": 0.95})
            for agent_id in dummy_env._observe()
        }
    elif (kind == "marl") or (kind == "fedrl"):
        policies = {
            agent_id: (PPOTorchPolicy, obs_space, act_space, {"gamma": 0.95})
            for agent_id in dummy_env._observe()
        }
        config["multiagent"] = {
            "policies": policies,
            "policy_mapping_fn": lambda agent_id: agent_id
        }
    return config


def load_env(netfile: str, kind: str, ranked: bool) -> Tuple[AbstractSumoEnv, Dict]:
    env_class = SumoEnv
    config = load_config(netfile, kind, ranked)
    return env_class, config


def load_agent(
    env: AbstractSumoEnv,
    config: Dict,
    checkpoint: str,
    agent_type: str="ppo"
) -> raygents.Trainer:

    if agent_type == "ppo":
        agent_class = PPOTrainer
    else:
        # TODO: Add the others.
        raise ValueError("Invalid agent type.")

    agent = agent_class(env=env, config=config)
    print(f">>> checkpoint={checkpoint}")
    agent.restore(checkpoint)

    # TODO: Look into how this can be used to reconstruct the policy map used during
    # training.
    policy_map = agent.workers.local_worker().policy_map
    # policy = next(iter(policy_map.values()))
    # policy.export_model("dummy")
    print(f"POLICY_MAP:\n{policy_map}\n")
    exit(0)

    return agent


def eval_step(env, obs, agent) -> Tuple:
    action = {agent_id: agent.compute_action(agent_obs, policy_id=agent_id)
              for agent_id, agent_obs in obs.items()}

    print(f"action:\n{action}")
    exit(0)

    next_obs, reward, done, info = env.step(action)
    done = next(iter(done.values()))
    return (next_obs, reward, done, info)


def eval(
    netfile: str,
    checkpoint: str,
    kind: str,
    ranked: bool,
    n_episodes: int=1,
    agent_type: str="ppo"
) -> DataFrame:
    ray.init()
    env_class, config = load_env(netfile=netfile, kind=kind, ranked=ranked)
    agent = load_agent(env_class, config, checkpoint, agent_type=agent_type)
    eval_data = defaultdict(list)
    last_reward = defaultdict(float)

    temp_delete_later = defaultdict(list)

    env = env_class(config["env_config"])
    for ep in range(1, n_episodes+1):
        obs = env.reset()
        print(f"\n\nobs:\n{obs}\n")
        done, step = False, 0
        last_reward = defaultdict(float)
        while not done:
            obs, reward, done, info = eval_step(env, obs, agent)
            for tls_id, r in reward.items():
                eval_data["netfile"].append(netfile)
                eval_data["ranked"].append(ranked)
                eval_data["agent_type"].append(agent_type)
                eval_data["kind"].append(kind)
                eval_data["tls_id"].append(tls_id)
                eval_data["episode"].append(ep)
                eval_data["cum_reward"].append(r + last_reward[tls_id])
                eval_data["reward"].append(r)
                eval_data["step"].append(step)
                last_reward[tls_id] = eval_data["cum_reward"][-1]
            step += 1

    env.close()
    ray.shutdown()
    return DataFrame.from_dict(eval_data)


def load_last_checkpoint(nettype: str, kind: str, ranked: bool) -> str:
    # TODO: This function is broken and doesn't correctly get the most recent file...
    # We may need ot rethink this setup.

    assert nettype in ["complex_inter", "single_inter", "two_inter"]
    assert kind in ["fedrl", "marl", "sarl"]
    assert isinstance(ranked, bool)

    if kind == "fedrl":
        kind = "FedRL"
    elif kind == "marl":
        kind = "MARL"
    else:
        kind = "SARL"

    ranked = "ranked" if ranked else "unranked"
    dirs = glob.glob(join("out", "models", kind, nettype, f"{ranked}*"))
    model_dir = sorted(dirs, key=lambda x: os.stat(x).st_mtime)[-1]
    # TODO: Reevaluate this approach ^^^
    checkpoint_dirs = glob.glob(join(model_dir, "checkpoint*"))
    checkpoint_dir = sorted(checkpoint_dirs,
                            key=lambda x: int(x.split(os.sep)[-1].split("_")[1]))[-1]  # TODO
    checkpoint_tail = checkpoint_dir.split(os.sep)[-1]
    checkpoint = join(checkpoint_dir, checkpoint_tail.replace("_", "-"))
    return checkpoint


if __name__ == "__main__":
    net_files = [
        join("configs", "complex_inter", "complex_inter.net.xml"),
        join("configs", "single_inter", "single_inter.net.xml"),
        join("configs", "two_inter", "two_inter.net.xml")
    ]

    dataframes = []
    for netfile in net_files:
        for kind in ["fedrl", "marl", "sarl"]:
            for ranked in [True, False]:
                nettype = netfile.split(os.sep)[1]
                checkpoint = load_last_checkpoint(nettype=nettype, kind=kind,
                                                  ranked=ranked)
                
                # ranked_str = "ranked" if ranked else "unranked"
                # checkpoint = join("out", "models", "Simple", f"{ranked_str}-fedrl",
                #                   "checkpoint_100", "checkpoint-100")

                df = eval(netfile, checkpoint, kind, ranked, n_episodes=10)
                dataframes.append(df)
                # print(f">>> TorchScript Model worked! Exiting...")
                print("New eval worked! Exiting.")
                exit(0)

    print(">> EVAL DONE!")
    final_df = pd.concat(dataframes)
    final_df.to_csv("eval_data.csv")
