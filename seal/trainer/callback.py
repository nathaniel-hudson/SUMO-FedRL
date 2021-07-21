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
"""

import numpy as np

from collections import defaultdict
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from typing import Dict


class FedRLCommCallback(DefaultCallbacks):

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy], episode: MultiAgentEpisode,
                         env_index: int, **kwargs) -> None:
        episode.user_data["communication_cost"] = defaultdict()
        episode.hist_data["communication_cost"] = defaultdict()

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs) -> None:
        agent_ids = set(episode.agent_rewards.keys())

        ...
        for idx in agent_ids:
            episode.user_data["communication_cost"]

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
        # return super().on_postprocess_trajectory(
        #     worker, episode, agent_id, policy_id,
        #     policies, postprocessed_batch, original_batches, **kwargs
        # )
        pass
