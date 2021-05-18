import os
import ray

from collections import defaultdict
from os.path import join
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from time import ctime
from fluri.sumo.multi_agent_env import MultiPolicySumoEnv
from fluri.strategies.fedavg import federated_avg
from fluri.sumo.kernel.trafficlights import RANK_DEFAULT
from fluri.trainer.ray.base import BaseTrainer
from fluri.trainer.const import *
from fluri.trainer.util import *

OUT_DIR = "fedrl"

class FedTrainer(BaseTrainer):

    def __init__(self) -> None:
        self

    def on_setup():
        # TODO
        pass


def train(
    n_rounds: int=10, 
    fed_step: int=10, 
    ranked: bool=RANK_DEFAULT,
    model_name: str=None,
    **kwargs    
) -> None:
    """Multi-agent reinforcement learning with Ray's RlLib.

    Args:
        n_rounds (int): Number of training rounds. Defaults to 10.
    """
    dummy_env = MultiPolicySumoEnv(config=get_env_config(ranked=ranked))
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    policies = {
        agent_id: (PPOTorchPolicy, obs_space, act_space, {"gamma": 0.95})
        for agent_id in dummy_env._observe()
    }

    ray.init()
    trainer = PPOTrainer(env=MultiPolicySumoEnv, config={
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": lambda agent_id: agent_id,
        },
        "lr": 0.001,
        "num_gpus": 0,
        "num_workers": 0,  # NOTE: For some reason, this *needs* to be 0.
        "framework": "torch",
        "log_level": "ERROR",
        "env_config": get_env_config(ranked=ranked),
    })
    status = "[Ep. #{}] Mean reward: {:6.2f} -- Mean length: {:4.2f} -- Saved {} ({})."
    
    if model_name is None:
        out_file = join("out", "models", OUT_DIR)
    else:
        out_file = join("out", "models", OUT_DIR, model_name)
    
    training_data = defaultdict(list)
    for r in range(n_rounds):
        # Perform en episode/round of training.
        result = trainer.train()

        # Determine if federated aggregation should be done in this round.
        if fed_step is None:
            time_to_aggregate = False
        elif r != 0 and r % fed_step == 0:
            time_to_aggregate = True
        else:
            time_to_aggregate = False

        # Store the data into the `training_data` dictionary for plotting and such.
        for policy in policies:
            training_data["round"].append(r)
            training_data["trainer"].append("MARL")
            training_data["policy"].append(policy)
            training_data["fed_round"].append(time_to_aggregate)
            for key, value in result.items():
                if isinstance(value, dict):
                    if policy in value:
                        training_data[key].append(value[policy])
                    else:
                        training_data[key].append(value)
                else:
                    training_data[key].append(value)

        # Incorporate federated learning if it is time to aggregate.
        if time_to_aggregate:
            policy_arr = [trainer.get_policy(policy_id)
                          for policy_id in policies]
            new_weights = federated_avg(policy_arr)
            for policy_id in policies:
                trainer.get_policy(policy_id).set_weights(new_weights)

        # Print the status of training.
        trainer.save(out_file)
        print(status.format(
            r+1,
            result["episode_reward_mean"],
            result["episode_len_mean"],
            out_file.split(os.sep)[-1],
            ctime()
        ))

    trainer.save(out_file)
    trainer.stop()
    ray.shutdown()
    trainer.workers.local_worker().env.close()
    return training_data
