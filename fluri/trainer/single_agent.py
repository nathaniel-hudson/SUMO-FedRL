from fluri.sumo.multi_agent_env import MultiPolicySumoEnv
from fluri.sumo.single_agent_env import SinglePolicySumoEnv
from fluri.trainer.base import BaseTrainer
from fluri.trainer.util import *
from typing import Any, Dict, Tuple


# TODO: Add communication cost trade-off code.
class SinglePolicyTrainer(BaseTrainer):

    POLICY_KEY: str = "sarl_policy"

    def __init__(self, **kwargs):
        name = "SARL"
        super().__init__(
            env=MultiPolicySumoEnv,
            sub_dir=name,
            **kwargs
        )
        self.trainer_name = name
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
                "policy_mapping_fn": lambda _: SinglePolicyTrainer.POLICY_KEY
            },
            "num_gpus": self.num_gpus,
            "num_workers": self.num_workers,
        }

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
