from fluri.sumo.multi_agent_env import MultiPolicySumoEnv
from fluri.sumo.single_agent_env import SinglePolicySumoEnv
from fluri.trainer.base import BaseTrainer
from fluri.trainer.util import *
from typing import Any, Dict


class SinglePolicyTrainer(BaseTrainer):
    # TODO: Add communication cost trade-off code.
    def __init__(self, **kwargs):
        name = "SARL"
        super().__init__(
            env=MultiPolicySumoEnv,
            sub_dir=name,
            **kwargs
        )
        # super().__init__(
        #     env=SinglePolicySumoEnv,
        #     sub_dir=name,
        #     **kwargs
        # )
        self.trainer_name = name
        self.idx = self.get_key_count()
        self.incr_key_count()

    def init_config(self) -> Dict[str, Any]:
        return {
            "env_config": self.env_config_fn(),
            "framework": "torch",
            "log_level": self.log_level,
            "lr": self.learning_rate,
            "num_gpus": self.num_gpus,
            "num_workers": self.num_workers,
        }

    def on_data_recording_step(self) -> None:
        self.training_data["round"].append(self._round)
        self.training_data["trainer"].append("SARL")
        for key, value in self._result.items():
            self.training_data[key].append(value)
