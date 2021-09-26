'''
RESOURCES:
    + https://docs.ray.io/en/master/_modules/ray/rllib/evaluation/episode.html
    + https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
'''

from collections import defaultdict
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from seal.trainer.communication import *
from typing import Dict


class BaseCommCallback(DefaultCallbacks):

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy], episode: MultiAgentEpisode,
                         env_index: int, **kwargs) -> None:
        self.comm_cost = defaultdict(int)
        episode.user_data["comm_cost"] = defaultdict(int)
        # episode.hist_data["comm_cost"] = defaultdict(list)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs) -> None:
        # self.iteration_reward = episode.total_reward
        for key in self.comm_cost:
            comm_type, policy_id = key
            comm_type = comm_type.replace("_", "-")
            new_key = f"policy_{policy_id}_comm={comm_type}"
            # new_key = FEATURE_DIVIDER.join(sub_key for sub_key in key)
            episode.custom_metrics[new_key] = self.comm_cost[key]
            if new_key not in episode.hist_data:
                episode.hist_data[new_key] = []
            episode.hist_data[new_key].append(self.comm_cost[key])

        # Reset the episode's data.
        episode.user_data["comm_cost"] = defaultdict(int)

    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        result["callback_ok"] = True
        # result["reward_this_iteration"] = self.iteration_reward
