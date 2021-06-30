import numpy as np

from fluri.sumo.env import SumoEnv
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

    def on_make_final_policy(self) -> Weights:
        policy_arr = [self.ray_trainer.get_policy(policy_id)
                      for policy_id in self.policies if policy_id != GLOBAL_POLICY_VAR]
        return FedPolicyTrainer.fedavg(policy_arr)

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
        ## TODO: Adjust this so that it's not longer the basic, worthless, naive avg.
        weights = np.array([policy.get_weights() for policy in policies])
        policy_keys = policies[0].get_weights().keys()
        new_weights = {}
        for key in policy_keys:
            weights = np.array([policy.get_weights()[key]
                                for policy in policies])
            new_weights[key] = sum(1/len(policies) * weights[k]
                                   for k in range(len(policies)))
        return new_weights
