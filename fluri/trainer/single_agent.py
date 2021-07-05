from fluri.sumo.env import SumoEnv
from fluri.trainer.base import BaseTrainer
from fluri.trainer.util import *
from typing import Any, Dict, Tuple


# TODO: Add communication cost trade-off code.
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

    def on_make_final_policy(self) -> Weights:
        policy = self.ray_trainer.get_policy(SinglePolicyTrainer.POLICY_KEY)
        return policy.get_weights()

    def on_data_recording_step(self) -> None:
        for policy in self.policies:
            self.training_data["round"].append(self._round)
            self.training_data["trainer"].append("SARL")
            for key, value in self._result.items():
                if isinstance(value, dict):
                    if policy in value:
                        self.training_data[key].append(value[policy])
                    else:
                        self.training_data[key].append(value)
                else:
                    self.training_data[key].append(value)
        # for key, value in self._result.items():
        #     self.training_data[key].append(value)

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
