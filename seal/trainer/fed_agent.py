import numpy as np
import os

from collections import defaultdict
from seal.sumo.env import SumoEnv
from typing import Any, Dict, List, NewType
from seal.trainer.base import BaseTrainer
from seal.trainer.communication.fed_callback import FedRLCommCallback
from seal.trainer.data.parser import DataParser
from seal.trainer.util import *
from seal.trainer.weight_aggr import *
from time import ctime
from typing import Any, Dict, Tuple

Weights = NewType("Weights", Dict[Any, np.array])
Policy = NewType("Policy", Dict[Any, np.array])


MIN_REWARD = -4
DEFAULT_AGGR_FN = "traffic"

WEIGHT_FUNCTIONS = {
        "naive":      naive_weight_function,
        "neg_reward": neg_reward_weight_function, # BAD
        "pos_reward": pos_reward_weight_function, # Experimental
        "traffic":    traffic_weight_function     # Experimental
    }

class FedPolicyTrainer(BaseTrainer):

    def __init__(self, fed_step: int, **kwargs) -> None:
        super().__init__(
            env=SumoEnv,
            sub_dir="FedRL",
            **kwargs
        )
        self.trainer_name = "FedRL"
        self.fed_step = fed_step
        self.idx = self.get_key_count()
        self.incr_key_count()
        self.policy_config = {}
        self.policy_mapping_fn = lambda agent_id: agent_id
        self.communication_callback_cls = FedRLCommCallback
        self.reward_tracker = defaultdict(float)
        self.episode_data = defaultdict(lambda: defaultdict(float))
        self.weight_fn = kwargs.get("weight_fn", DEFAULT_AGGR_FN)
        assert self.weight_fn in WEIGHT_FUNCTIONS

    def __reset_reward_tracker(self) -> None:
        for policy in self.reward_tracker:
            self.reward_tracker[policy] = 0.0

    def on_make_final_policy(self) -> Weights:
        policy_dict = {policy_id: self.ray_trainer.get_policy(policy_id)
                       for policy_id in self.policies
                       if policy_id != GLOBAL_POLICY_VAR}
        return self.fedavg(policy_dict)

    def on_data_recording_step(self) -> None:
        # Determine if aggregation is performed during this training iteration or not.
        aggregate_this_round = self._is_aggregating_step()

        # Record the data for training process evaluation.
        self.training_data["round"].append(self._round)
        self.training_data["trainer"].append("FedRL")
        self.training_data["fed_round"].append(aggregate_this_round)
        self.training_data["ranked"].append(self.ranked)
        self.training_data["weight_aggr_fn"].append(self.weight_fn)
        for key, value in self._result.items():
            self.training_data[key].append(value)

        # Track the reward for this policy during this training step. This is only
        # used for the FedAvg subroutine in the AGGREGATION step.
        parsed_data = DataParser(self._result)
        for policy in self.policies:
            if policy != GLOBAL_POLICY_VAR:
                self.episode_data[policy]["reward"] += parsed_data.policy_reward(policy) 
                self.episode_data[policy]["num_vehicles"] += parsed_data.num_vehicles(policy)
                # 
                self.reward_tracker[policy] += \
                    self._result["policy_reward_mean"].get(policy, MIN_REWARD)

        # Aggregate the weights via the Federated Averaging algorithm.
        if aggregate_this_round:
            policy_dict = {policy_id: self.ray_trainer.get_policy(policy_id)
                           for policy_id in self.policies
                           if policy_id != GLOBAL_POLICY_VAR}
            new_params = self.fedavg(policy_dict)
            for policy_id in self.policies:
                self.ray_trainer.get_policy(policy_id).set_weights(new_params)


    '''
    def on_data_recording_step_v1(self) -> None:
        aggregate_this_round = self._is_aggregating_step()
        parsed_data = DataParser(self._result)
        total_reward = 0
        for policy in self.policies:
            # Record the data for training process evaluation.
            self.training_data["round"].append(self._round)
            self.training_data["trainer"].append("FedRL")
            self.training_data["policy"].append(policy)
            self.training_data["fed_round"].append(aggregate_this_round)
            self.training_data["ranked"].append(self.ranked)
            self.training_data["weight_aggr_fn"].append(self.weight_fn)

            for key, value in self._result.items():
                if isinstance(value, dict):
                    if policy in value:
                        self.training_data[key].append(value[policy])
                    else:
                        self.training_data[key].append(value)
                else:
                    self.training_data[key].append(value)

            # Track the reward for this policy during this training step. This is only
            # used for the FedAvg subroutine in the AGGREGATION step.
            if policy != GLOBAL_POLICY_VAR:
                self.episode_data[policy]["reward"] += parsed_data.policy_reward(policy) 
                self.episode_data[policy]["num_vehicles"] += parsed_data.num_vehicles(policy)
                # 
                self.reward_tracker[policy] += \
                    self._result["policy_reward_mean"].get(policy, MIN_REWARD)

        # Aggregate the weights via the Federated Averaging algorithm.
        if aggregate_this_round:
            policy_dict = {policy_id: self.ray_trainer.get_policy(policy_id)
                           for policy_id in self.policies
                           if policy_id != GLOBAL_POLICY_VAR}
            new_params = self.fedavg(policy_dict)
            for policy_id in self.policies:
                self.ray_trainer.get_policy(policy_id).set_weights(new_params)
    '''


    def on_policy_setup(self) -> Dict[str, Tuple[Any]]:
        dummy_env = self.env(config=self.env_config_fn())
        obs_space = dummy_env.observation_space
        act_space = dummy_env.action_space
        return {
            agent_id: (
                self.policy_type,
                obs_space,
                act_space,
                self.policy_config
            )
            for agent_id in dummy_env._observe()
        }

    def fedavg(
        self, 
        policy_dict: Dict[str, Policy] 
        # weight_fn: str="traffic"
    ) -> Weights:
        # STEP 1: Grab the aggregation function specified at initialization.
        weight_fn = WEIGHT_FUNCTIONS[self.weight_fn]

        # STEP 2: Compute the coefficients for each policy in the system based on reward.
        coeffs = weight_fn(self.episode_data)
        
        # STEP 3: Compute the reward-based averaged policy weights by weight key.
        new_params = {}
        param_keys = next(iter(policy_dict.values())).get_weights().keys()
        for key in param_keys:
            weights = {policy_id: np.array(policy.get_weights()[key])
                       for policy_id, policy in policy_dict.items()}
            new_params[key] = sum(coeffs[policy_id] * weights[policy_id]
                                   for policy_id in policy_dict)

        # STEP 4: Reset the reward trackers for each of the policies.
        self.__reset_reward_tracker()
        return new_params

    # ============================================================ #

    def on_logging_step(self) -> None:
        aggregate_this_round = self._is_aggregating_step()
        status = "{}Ep. #{} | ranked={} | fed_round={} | Mean reward: {:6.2f} | " \
                 "Mean length: {:4.2f} | Saved {} ({})"
        print(status.format(
            "" if self.trainer_name is None else f"[{self.trainer_name}] ",
            self._round+1,
            self.ranked,
            aggregate_this_round,
            self._result["episode_reward_mean"],
            self._result["episode_len_mean"],
            self.model_path.split(os.sep)[-1],
            ctime()
        ))

    def _is_aggregating_step(self) -> bool:
        if self.fed_step is None:
            return True
        elif (self._round + 1) % self.fed_step == 0:
            return True
        else:
            return False