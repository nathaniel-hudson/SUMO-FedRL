import ray

from abc import ABC, abstractmethod
from pandas import DataFrame
from ray.rllib.agents import (a3c, dqn, ppo)


class BaseTrainer(ABC):

    def __init__(
        self,
        policy: str="ppo",
        gamma: float=0.95,
    ) -> None:
        assert 0 <= gamma <= 1
        self.env = None
        self.gamma = gamma

        if policy == "a3c":
            self.trainer_type = a3c.A2CTrainer
            self.policy_type = a3c.a3c_torch_policy
        elif policy == "dqn":
            self.trainer_type = dqn.DQNTrainer
            self.policy_type = dqn.DQNTorchPolicy
        elif policy == "ppo":
            self.trainer_type = ppo.PPOTrainer
            self.policy_type = ppo.PPOTorchPolicy
        else:
            raise NotImplemented(f"Do not support policies for `{policy}`.")

    def train(self, num_rounds: int, **kwargs) -> None:
        self.on_multi_policy_setup()
        ray.init()
        self.on_setup()
        for r in range(num_rounds):
            result = self.ray_trainer.train()
            self.on_data_recording_step(result)
            self.on_logging_step(result)
        self.on_tear_down()

    def on_multi_policy_setup(self) -> None:
        dummy_env = self.environment(config=config_factory(config_args))
        obs_space = dummy_env.observation_space
        act_space = dummy_env.action_space
        policies = {
            agent_id: (self.policy, obs_space, act_space, {"gamma": self.gamma})
            for agent_id in dummy_env._observe()
        }

    @abstractmethod
    def on_setup(self) -> None:
        pass

    @abstractmethod
    def on_logging_step(self) -> None:
        print(self.status.format(
            ...
        ))

    @abstractmethod
    def on_data_recording_step(self) -> None:
        pass

    def on_tear_down(self) -> DataFrame:
        self.ray_trainer.save(...)
        self.ray_trainer.stop()
        ray.shutdown()
        self.ray_trainer.local_worker.env.close()
        return DataFrame.from_dict(self.data)
