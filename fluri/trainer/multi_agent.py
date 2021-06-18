from fluri.sumo.multi_agent_env import MultiPolicySumoEnv
from fluri.trainer.base import BaseTrainer
from fluri.trainer.util import *
from typing import Any, Dict


class MultiPolicyTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(
            env=MultiPolicySumoEnv,
            sub_dir="MARL",
            multi_policy_flag=True,
            **kwargs
        )
        self.trainer_name = "MARL"
        self.idx = self.get_key_count()
        self.incr_key_count()
        self.multi_agent_policy_config = {}

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
