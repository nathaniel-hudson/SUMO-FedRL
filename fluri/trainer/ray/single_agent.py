import os
import ray

from collections import defaultdict
from os.path import join
from ray.rllib.agents.ppo import PPOTrainer
from time import ctime
from fluri.sumo.single_agent_env import SinglePolicySumoEnv
from fluri.sumo.kernel.trafficlights import RANK_DEFAULT
from fluri.trainer.ray.base import BaseTrainer
from fluri.trainer.const import *
from fluri.trainer.util import *
from typing import Any, Dict

OUT_DIR = "sarl"

class SinglePolicyTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(
            env=SinglePolicySumoEnv,
            **kwargs
        )

    def init_config(self) -> Dict[str, Any]:
        return {
            "env_config": self.env_config_fn(ranked=self.ranked),
            "framwork": "torch",
            "log_level": self.log_level,
            "lr": self.learning_rate,
            "num_workers": self.num_workers,
        }

    def on_data_recording_step(self) -> None:
        self.training_data["round"].append(self._round)
        self.training_data["trainer"].append("SARL")
        for key, value in self._result.items():
            self.training_data[key].append(value)

def train(
    n_rounds: int=10,
    ranked: bool=RANK_DEFAULT,
    model_name: str=None,
    **kwargs
) -> None:
    """Single-agent reinforcement learning with Ray's RlLib.

    Args:
        n_rounds (int): Number of training rounds. Defaults to 10.
    """
    ray.init()
    trainer = PPOTrainer(env=SinglePolicySumoEnv, config={
        "lr": 0.001,
        "num_workers": 0,  # NOTE: For some reason, this *needs* to be 0.
        "framework": "torch",
        "env_config": get_env_config(ranked=ranked),
        "num_gpus": kwargs.get("num_gpus", 0)
    })
    status = "[Ep. #{}] Mean reward: {:6.2f} -- Mean length: {:4.2f} -- Saved {} ({})."

    if model_name is None:
        out_file = join("out", "models", OUT_DIR)
    else:
        out_file = join("out", "models", OUT_DIR, model_name)

    training_data = defaultdict(list)
    for round in range(n_rounds):
        result = trainer.train()
        trainer.save(out_file)

        # Record training data for plotting and such.
        training_data["round"].append(round)
        training_data["trainer"].append("SARL")
        for key, value in result.items():
            training_data[key].append(value)

        print(status.format(
            round+1,
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
