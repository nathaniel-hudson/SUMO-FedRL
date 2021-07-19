import pandas as pd

from collections import defaultdict

from ray.rllib.evaluation.rollout_worker import RolloutWorker


class CommunicationModel:

    __edge_to_tls_action_comms: defaultdict
    __edge_to_tls_rank_comms: defaultdict
    __edge_to_tls_policy_comms: defaultdict
    __tls_to_edge_obs_comms: defaultdict
    __tls_to_edge_policy_comms: defaultdict
    __veh_to_tls_info_comms: defaultdict

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tls_ids = set()
        self.__edge_to_tls_action_comms = defaultdict(int)
        self.__edge_to_tls_rank_comms = defaultdict(int)
        self.__edge_to_tls_policy_comms = defaultdict(int)
        self.__tls_to_edge_obs_comms = defaultdict(int)
        self.__tls_to_edge_policy_comms = defaultdict(int)
        self.__veh_to_tls_info_comms = defaultdict(int)

    def incr_edge_to_tls_action(self, tls_id) -> None:
        self.__edge_to_tls_action_comms[tls_id] += 1
        self.tls_ids.add(tls_id)

    def incr_edge_to_tls_rank(self, tls_id) -> None:
        self.__edge_to_tls_rank_comms[tls_id] += 1
        self.tls_ids.add(tls_id)

    def incr_edge_to_tls_policy(self, tls_id) -> None:
        self.__edge_to_tls_policy_comms[tls_id] += 1
        self.tls_ids.add(tls_id)

    def incr_tls_to_edge_obs(self, tls_id) -> None:
        self.__tls_to_edge_obs_comms[tls_id] += 1
        self.tls_ids.add(tls_id)

    def incr_tls_to_edge_policy(self, tls_id) -> None:
        self.__tls_to_edge_policy_comms[tls_id] += 1
        self.tls_ids.add(tls_id)

    def incr_veh_to_tls_info(self, tls_id) -> None:
        self.__veh_to_tls_info_comms[tls_id] += 1
        self.tls_ids.add(tls_id)

    def to_csv(self, path) -> None:
        aux_data = defaultdict(list)
        for tls in self.tls_ids:
            aux_data["tls_id"].append(tls)
            aux_data["edge_to_tls_action_comms"].append(
                self.__edge_to_tls_action_comms[tls])
            aux_data["edge_to_tls_rank_comms"].append(
                self.__edge_to_tls_rank_comms[tls])
            aux_data["edge_to_tls_policy_comms"].append(
                self.__edge_to_tls_policy_comms[tls])
            aux_data["tls_to_edge_obs_comms"].append(
                self.__tls_to_edge_obs_comms[tls])
            aux_data["tls_to_edge_policy_comms"].append(
                self.__tls_to_edge_policy_comms[tls])
            aux_data["veh_to_tls_info_comms"].append(
                self.__veh_to_tls_info_comms[tls])
        aux_df = pd.DataFrame.from_dict(aux_data)
        aux_df.to_csv(path)


## ================================================================== ##

import numpy as np

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from typing import Dict


class CommunicationCallback(DefaultCallbacks):

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy], episode: MultiAgentEpisode,
                         env_index: int, **kwargs) -> None:
        assert episode.length == 0, \
            "ERROR: `on_episode_start()` callback should be called right after " \
            "env reset!"
        episode.user_data["communication_cost"] = []
        episode.hist_data["communication_cost"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs) -> None:
        assert episode.length > 0, \
            "ERROR: `on_episode_step()` callback should not be called right " \
            "after env reset!"
        # print(f">> episode:\n{episode.last_observation_for()}\n{episode.last_raw_obs_for()}")
        # comm_cost = abs(episode.last_observation_for()[2])
        # raw_comm_cost = abs(episode.last_raw_obs_for()[2])
        # episode.user_data["communication_cost"].append(comm_cost)

        agent_id = next(iter(episode.agent_rewards.keys()))[0]
        info = episode.last_info_for(agent_id)
        episode.user_data["edge_to_tls_action_comm_cost"].append(...)
        episode.user_data["edge_to_tls_policy_comm_cost"].append(...)
        episode.user_data["edge_to_tls_rank_comm_cost"].append(...)
        episode.user_data["tls_to_edge_obs_comm_cost"].append(...)
        episode.user_data["tls_to_edge_policy_comm_cost"].append(...)
        episode.user_data["veh_to_tls_info_comm_cost"].append(...)
        print(f"info[{agent_id}]:\n{info}")
        exit(0)

    ## ---------------------------------------------------------------------------------- ##

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        # Make sure this episode is really done.
        assert episode.batch_builder.policy_collectors["default_policy"].\
            buffers["dones"][-1], \
            "ERROR: `on_episode_end()` should only be called " \
            "after episode is done!"
        comm_cost = np.mean(episode.user_data["communication_cost"])
        print(f"episode {episode.episode_id} (env-idx={env_index}) ended with "
              f"length {episode.length} and comm_cost angles {comm_cost}")
        episode.custom_metrics["communication_cost"] = comm_cost
        episode.hist_data["communication_cost"] = episode.user_data["communication_cost"]

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        print(f"returned sample batch of size {samples.count}")

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        print(f"trainer.train() result: {trainer} -> "
              f"{result['episodes_this_iter']} episodes")
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True
        # TODO: YOU CAN CHANGE RESULTS STUFF HERE!!!

    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
                          result: dict, **kwargs) -> None:
        result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"])
        print("policy.learn_on_batch() result: {} -> sum actions: {}".format(
            policy, result["sum_actions_in_train_batch"]))

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        print("postprocessed {} steps".format(postprocessed_batch.count))
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1
