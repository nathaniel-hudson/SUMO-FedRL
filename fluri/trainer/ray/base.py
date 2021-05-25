import fluri.trainer.ray.defaults as defaults
import os
import ray

from abc import ABC, abstractmethod
from collections import defaultdict
from pandas import DataFrame
from ray.rllib.agents import (a3c, dqn, ppo)
from time import ctime
from typing import Any, Dict, List

from fluri.trainer.ray.counter import Counter
from fluri.trainer.ray.defaults import *
from fluri.sumo.sumo_env import SumoEnv


class BaseTrainer(ABC):

    counter: Counter
    idx: int
    num_gpus: int
    env: SumoEnv
    learning_rate: float
    log_level: str
    gamma: float
    multi_policy_flag: bool
    num_gpus: int
    num_workers: int
    policy: str
    policy_type: Any  # TODO: Change this.
    trainer_type: Any  # TODO: Change this.

    def __init__(
        self,
        checkpoint_freq: int=1,
        env: SumoEnv=None,
        gamma: float=0.95,
        learning_rate: float=0.001,
        log_level: str="ERROR",
        model_name: str=None,
        multi_policy_flag: bool=False,
        num_gpus: int=0,
        num_workers: int=0,
        root_dir: List[str]=["out"],
        sub_dir: str=None,
        policy: str="ppo",
        **kwargs
    ) -> None:
        assert 0 <= gamma <= 1
        self.checkpoint_freq = checkpoint_freq
        self.counter = Counter()
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.log_level = log_level
        self.model_name = model_name
        self.multi_policy_flag = multi_policy_flag
        self.num_gpus = num_gpus
        self.num_workers = num_workers

        self.out_data_dir = os.path.join(*(root_dir + ["data"]))
        self.out_model_dir = os.path.join(*(root_dir + ["models"]))
        if sub_dir is not None:
            self.out_data_dir = os.path.join(self.out_data_dir, sub_dir)
            self.out_model_dir = os.path.join(self.out_model_dir, sub_dir)
        
        self.gui = kwargs.get("gui", defaults.GUI)
        self.net_file = kwargs.get("net_file", defaults.NET_FILE)
        self.ranked = kwargs.get("ranked", defaults.RANKED)
        self.rand_routes_on_reset = kwargs.get("rand_routes_on_reset",
                                                defaults.RAND_ROUTES_ON_RESET)

        self.net_dir = self.net_file.split(os.sep)[-1].split(".")[0]
        self.out_data_dir = os.path.join(self.out_data_dir, self.net_dir)
        self.out_model_dir = os.path.join(self.out_model_dir, self.net_dir)

        if not os.path.isdir(self.out_data_dir):
            os.makedirs(os.path.join(self.out_data_dir))
        if not os.path.isdir(self.out_model_dir):
            os.makedirs(os.path.join(self.out_model_dir))

        if policy == "a3c":
            self.trainer_type = a3c.A3CTrainer
            self.policy_type = a3c.a3c_torch_policy
        elif policy == "dqn":
            self.trainer_type = dqn.DQNTrainer
            self.policy_type = dqn.DQNTorchPolicy
        elif policy == "ppo":
            self.trainer_type = ppo.PPOTrainer
            self.policy_type = ppo.PPOTorchPolicy
        else:
            raise NotImplemented(f"Do not support policies for `{policy}`.")
        
        self.trainer_name = None
        self.idx = None
        self.multi_agent_policy_config = None


    def train(self, num_rounds: int, save_on_end: bool=True, **kwargs) -> DataFrame:
        if self.multi_policy_flag:
            self.on_multi_policy_setup()
        self.on_setup()
        for r in range(num_rounds):
            self._round = r
            self._result = self.ray_trainer.train()
            self.on_data_recording_step()
            self.on_logging_step()
            if r % self.checkpoint_freq == 0:
                self.ray_trainer.save(self.model_path)
        dataframe = self.on_tear_down()
        if save_on_end:
            path = os.path.join(self.out_data_dir, f"{self.get_filename()}.csv")
            dataframe.to_csv(path)
        return dataframe


    def on_multi_policy_setup(self) -> None:
        dummy_env = self.env(config=self.env_config_fn())
        obs_space = dummy_env.observation_space
        act_space = dummy_env.action_space
        self.policies = {
            agent_id: (self.policy_type, obs_space, act_space, 
                       self.multi_agent_policy_config)
            for agent_id in dummy_env._observe()
        }


    def on_setup(self) -> None:
        ray.init()
        self.ray_trainer = self.trainer_type(env=self.env, config=self.init_config())
        self.model_path = os.path.join(self.out_model_dir, self.get_filename())
        self.training_data = defaultdict(list)


    def on_tear_down(self) -> DataFrame:
        self.ray_trainer.save(self.model_path)
        self.ray_trainer.stop()
        ray.shutdown()
        self.ray_trainer.workers.local_worker().env.close()
        return DataFrame.from_dict(self.training_data)


    def on_logging_step(self) -> None:
        status = "[Ep. #{}] Mean reward: {:6.2f} -- Mean length: {:4.2f} -- Saved {} ({})."
        print(status.format(
            self._round+1,
            self._result["episode_reward_mean"],
            self._result["episode_len_mean"],
            self.model_path.split(os.sep)[-1],
            ctime()
        ))

    def get_key(self) -> str:
        if self.trainer_name is None:
            raise ValueError("`trainer_name` cannot be None.")
        ranked = "ranked" if self.ranked else "unranked"
        key = f"{self.trainer_name}_{self.net_dir}_{ranked}"
        return key

    def get_key_count(self) -> int:
        return self.counter.get(self.get_key())

    def incr_key_count(self) -> None:
        self.counter.increment(self.get_key())

    def get_filename(self) -> str:
        if self.trainer_name is None:
            raise ValueError("`trainer_name` cannot be None.")
        ranked = "ranked" if self.ranked else "unranked"
        return f"{ranked}_{self.idx}"

    def env_config_fn(self) -> Dict:
        return {
            "gui": self.gui,
            "net-file": self.net_file,
            "rand_routes_on_reset": self.rand_routes_on_reset,
            "ranked": self.ranked,
        }

    @abstractmethod
    def init_config(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def on_data_recording_step(self) -> None:
        pass