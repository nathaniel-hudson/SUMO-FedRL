from seal.sumo.env import SumoEnv
from seal.trainer.base import BaseTrainer
from seal.trainer.communication.single_callback import SinglePolicyCommCallback
from seal.trainer.data.parser import DataParser
from seal.trainer.util import *
from typing import Any, Dict, Tuple


class SinglePolicyTrainer(BaseTrainer):

    POLICY_KEY: str = "sarl_policy"

    def __init__(self, **kwargs):
        name = "SARL"
        super().__init__(
            env=SumoEnv,
            sub_dir=name,
            **kwargs
        )
        self.trainer_name = name
        self.idx = self.get_key_count()
        self.incr_key_count()
        self.policy_config = {}
        self.policy_mapping_fn = lambda _: SinglePolicyTrainer.POLICY_KEY
        self.communication_callback_cls = SinglePolicyCommCallback

    def on_make_final_policy(self) -> Weights:
        policy = self.ray_trainer.get_policy(SinglePolicyTrainer.POLICY_KEY)
        return policy.get_weights()

    def on_data_recording_step(self) -> None:
        parsed_data = DataParser(self._result)
        total_reward = 0
        # print(f"\n\n\nSinglePolicyTrainer.on_data_recording_step():\n\tself.policies={self.policies}\n")
        # exit(0)
        for policy in self.policies:
            self.training_data["round"].append(self._round)
            self.training_data["trainer"].append("SARL")
            self.training_data["policy"].append(policy)
            
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
            SinglePolicyTrainer.POLICY_KEY: (
                self.policy_type,
                obs_space,
                act_space,
                self.policy_config
            )
        }
