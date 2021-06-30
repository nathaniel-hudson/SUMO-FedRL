import numpy as np

from fluri.sumo.env import SumoEnv
from fluri.trainer.base import BaseTrainer
from fluri.trainer.util import *
from typing import Any, Dict, Tuple


# TODO: Add communication cost trade-off code.
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

    def on_make_final_policy(self) -> Weights:
        policies = [policy for key, policy in self.policies.items() 
                    if key != GLOBAL_POLICY_VAR]
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
        for policy in self.policies:
            self.training_data["round"].append(self._round)
            self.training_data["trainer"].append("MARL")
            self.training_data["policy"].append(policy)
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
