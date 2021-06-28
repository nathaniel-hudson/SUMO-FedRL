import numpy as np

from fluri.sumo.multi_agent_env import MultiPolicySumoEnv
from typing import Any, Dict, List, NewType
from fluri.trainer.base import BaseTrainer
from fluri.trainer.util import *
from typing import Any, Dict, Tuple

Weights = NewType("Weights", Dict[Any, np.array])
Policy = NewType("Policy", Dict[Any, np.array])


# TODO: Add communication cost trade-off code.
class FedPolicyTrainer(BaseTrainer):

    def __init__(self, fed_step: int, **kwargs) -> None:
        super().__init__(
            env=MultiPolicySumoEnv,
            sub_dir="FedRL",
            multi_policy_flag=True,
            **kwargs
        )
        self.trainer_name = "FedRL"
        self.fed_step = fed_step
        self.idx = self.get_key_count()
        self.incr_key_count()
        self.policy_config = {}

    def init_config(self) -> Dict[str, Any]:
        return {
            "env_config": self.env_config_fn(),
            "framework": "torch",
            "log_level": self.log_level,
            "lr": self.learning_rate,
            "multiagent": {
                "policies": self.policies,
                "policy_mapping_fn": lambda agent_id: agent_id
            },
            "num_gpus": self.num_gpus,
            "num_workers": self.num_workers,
        }

    def on_data_recording_step(self) -> None:
        if self.fed_step is None:
            aggregate_this_round = False
        elif self._round != 0 and self._round % self.fed_step == 0:
            aggregate_this_round = True
        else:
            aggregate_this_round = False

        for policy in self.policies:
            self.training_data["round"].append(self._round)
            self.training_data["trainer"].append("FedRL")
            self.training_data["policy"].append(policy)
            self.training_data["fed_round"].append(aggregate_this_round)
            for key, value in self._result.items():
                if isinstance(value, dict):
                    if policy in value:
                        self.training_data[key].append(value[policy])
                    else:
                        self.training_data[key].append(value)
                else:
                    self.training_data[key].append(value)

        # Aggregate the weights via the Federated Averaging algorithm.
        if aggregate_this_round:
            policy_arr = [self.ray_trainer.get_policy(policy_id)
                          for policy_id in self.policies]
            new_weights = FedPolicyTrainer.fedavg(policy_arr)
            for policy_id in self.policies:
                self.ray_trainer.get_policy(policy_id).set_weights(new_weights)

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

    @classmethod
    def fedavg(cls, policies: List[Policy], C: float=1.0) -> Weights:
        weights = np.array([policy.get_weights() for policy in policies])
        policy_keys = policies[0].get_weights().keys()
        new_weights = {}
        for key in policy_keys:
            weights = np.array([policy.get_weights()[key]
                                for policy in policies])
            new_weights[key] = sum(1/len(policies) * weights[k]
                                   for k in range(len(policies)))
        return new_weights
