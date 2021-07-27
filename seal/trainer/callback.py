"""
FedRL:
    >> Trainer
        * edge2tls_policy += 1 (each fed round)
        * tls2edge_policy += 1 (each fed round)
    >> Env
        * edge2tls_action += 0
        * edge2tls_rank += 1 (if ranked)
        * tls2edge_obs += 0
        * veh2tls += 1 (per vehicle) [$$$]

SARL:
    >> Trainer
        * edge2tls_policy += 0
        * tls2edge_policy += 0
    >> Env
        * edge2tls_action += 1
        * edge2tls_rank += 1 (if ranked)
        * tls2edge_obs += 1
        * veh2tls += 1 (per vehicle) [$$$]

SARL:
    >> Trainer
        * edge2tls_policy += 0
        * tls2edge_policy += 0
    >> Env
        * edge2tls_action += 0
        * edge2tls_rank += 1 (if ranked)
        * tls2edge_obs += 0
        * veh2tls += 1 (per vehicle) [$$$]

[$$$] -- is the most difficult feature.

RESOURCES:
    + https://docs.ray.io/en/master/_modules/ray/rllib/evaluation/episode.html
    + https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py

comm_cost:
    {"edge2tls_action": {"tls_1": 182, "tls_2": 182,  ...},
     "edge2tls_policy": {"tls_1":   7, "tls_2":   7,  ...}}

"""

import numpy as np

from collections import defaultdict
from collections.abc import Iterable
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from typing import Dict


EDGE2TLS_POLICY = "edge_to_tls_policy_comms"
TLS2EDGE_POLICY = "tls_to_edge_policy_comms"
EDGE2TLS_ACTION = "edge_to_tls_action_comms"
EDGE2TLS_RANK = "edge_to_tls_rank_comms"
TLS2EDGE_OBS = "tls_to_edge_obs_comms"
VEH2TLS_COMM = "veh_to_tls_info_comms"

COMM_TYPES = set([EDGE2TLS_POLICY, TLS2EDGE_POLICY, EDGE2TLS_ACTION, EDGE2TLS_RANK,
                  TLS2EDGE_OBS, VEH2TLS_COMM])


class SinglePolicyCommCallback(DefaultCallbacks):

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy], episode: MultiAgentEpisode,
                         env_index: int, **kwargs) -> None:
        self.comm_cost = defaultdict(int)
        episode.user_data["comm_cost"] = defaultdict(int)
        episode.hist_data["comm_cost"] = defaultdict(int)

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs) -> None:
        # For some reason, the results of this function return a set of tuples of
        # identical keys... Not sure why, but that's why we only consider the 0th
        # elements of tuples.
        agent_ids = set([tuple[0] for tuple in episode.agent_rewards.keys()])
        for idx in agent_ids:
            info_dict = episode.last_info_for(idx)
            self.comm_cost[EDGE2TLS_ACTION, idx] += 1
            self.comm_cost[TLS2EDGE_OBS, idx] += 1
            self.comm_cost[VEH2TLS_COMM, idx] += info_dict["veh2tls_comms"]
            if info_dict["is_ranked"]:
                self.comm_cost[EDGE2TLS_RANK, idx] += 1

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs) -> None:
        for key in self.comm_cost:
            new_key = "___".join(sub_key for sub_key in key)
            episode.custom_metrics[new_key] = self.comm_cost[key]

    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        result["callback_ok"] = True


class MultiPolicyCommCallback(DefaultCallbacks):

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy], episode: MultiAgentEpisode,
                         env_index: int, **kwargs) -> None:
        self.comm_cost = defaultdict(int)
        episode.user_data["comm_cost"] = defaultdict(int)
        episode.hist_data["comm_cost"] = defaultdict(int)

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs) -> None:
        # For some reason, the results of this function return a set of tuples of
        # identical keys... Not sure why, but that's why we only consider the 0th
        # elements of tuples.
        agent_ids = set([tuple[0] for tuple in episode.agent_rewards.keys()])
        for idx in agent_ids:
            info_dict = episode.last_info_for(idx)
            self.comm_cost[EDGE2TLS_ACTION, idx] += 0
            self.comm_cost[TLS2EDGE_OBS, idx] += 0
            self.comm_cost[VEH2TLS_COMM, idx] += info_dict["veh2tls_comms"]
            if info_dict["is_ranked"]:
                self.comm_cost[EDGE2TLS_RANK, idx] += 1

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs) -> None:
        for key in self.comm_cost:
            new_key = "___".join(sub_key for sub_key in key)
            episode.custom_metrics[new_key] = self.comm_cost[key]

    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        result["callback_ok"] = True


## ============================================================================== ##

class FedRLCommCallback(DefaultCallbacks):

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy], episode: MultiAgentEpisode,
                         env_index: int, **kwargs) -> None:
        episode.user_data["comm_cost"] = defaultdict()
        episode.hist_data["comm_cost"] = defaultdict()

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs) -> None:
        agent_ids = set(episode.agent_rewards.keys())

        # ...
        for idx in agent_ids:
            episode.user_data["comm_cost"] = None

    def on_episode_end(self, *, worker: RolloutWorker, samples:
                       SampleBatch, **kwargs) -> None:
        pass

    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        pass

    def on_postprocess_trajectory(
        self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
        agent_id: str, policy_id: str, policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, SampleBatch], **kwargs
    ) -> None:
        pass
