import numpy as np

from seal.sumo.env import SumoEnv
from seal.trainer.base import BaseTrainer
from seal.trainer.communication.multi_callback import MultiPolicyCommCallback
from seal.trainer.data.parser import DataParser
from seal.trainer.util import *
from typing import Any, Dict, Tuple


class MultiPolicyTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(
            env=SumoEnv,
            sub_dir="MARL",
            **kwargs
        )
        self.trainer_name = "MARL"
        self.idx = self.get_key_count()
        self.incr_key_count()
        self.policy_config = {}
        self.policy_mapping_fn = lambda agent_id: agent_id
        self.communication_callback_cls = MultiPolicyCommCallback

    def on_make_final_policy(self) -> Weights:
        """This function takes the policy weights for each of the traffic light policies
           and merges them using a naive gradient averaging approach (i.e., policy weights
           are summed up and then divided by the number of policies in the system).

        Returns:
            Weights: The naively averaged policy weights.
        """
        policies = [self.ray_trainer.get_policy(policy_id)
                    for policy_id in self.policies if policy_id != GLOBAL_POLICY_VAR]
        weights = np.array([policy.get_weights() for policy in policies])
        policy_keys = policies[0].get_weights().keys()
        new_weights = {}
        for key in policy_keys:
            weights = np.array([policy.get_weights()[key]
                                for policy in policies])
            new_weights[key] = sum(1/len(policies) * weights[k]
                                   for k in range(len(policies)))
        return new_weights

    def on_data_recording_step(self) -> None:
        parsed_data = DataParser(self._result)
        total_reward = 0
        for policy in self.policies:
            self.training_data["round"].append(self._round)
            self.training_data["trainer"].append("MARL")
            self.training_data["policy"].append(policy)
            self.training_data["fed_round"].append(False)
            self.training_data["ranked"].append(self.ranked)

            # NOTE: Consider removing this later.
            # if policy != GLOBAL_POLICY_VAR:
            #     total_reward += parsed_data.policy_reward(policy)
            #     self.training_data["policy_reward"].append(parsed_data.policy_reward(policy))
            #     self.training_data["num_vehicles"].append(parsed_data.num_vehicles(policy))
            # else:
            #     self.training_data["policy_reward"].append("N/A")
            #     self.training_data["num_vehicles"].append("N/A")

            for key, value in self._result.items():
                if isinstance(value, dict):
                    if policy in value:
                        self.training_data[key].append(value[policy])
                    else:
                        self.training_data[key].append(value)
                else:
                    self.training_data[key].append(value)

    def on_policy_setup(self) -> Dict[str, Tuple[Any]]:
        dummy_env = self.env(config=self.env_config_fn())
        obs_space = dummy_env.observation_space
        act_space = dummy_env.action_space
        return {
            agent_id: (self.policy_type,
                       obs_space,
                       act_space,
                       self.policy_config)
            for agent_id in dummy_env._observe()
        }
