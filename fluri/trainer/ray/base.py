from abc import ABC, abstractmethod

class BaseTrainer(ABC):

    def __init__(self):
        pass

    
    def train(self, num_rounds: int, **kwargs) -> None:
        self.on_multi_policy_setup()
        self.on_setup()
        for r in range(num_rounds):
            self.on_training_step()
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
    def on_training_step(self) -> None:
        pass

    @abstractmethod
    def on_tear_down(self) -> None:
        pass