import json
import os
import ray

from abc import ABC, abstractmethod
from collections import defaultdict
from pandas import DataFrame
from ray.rllib.agents import (a3c, dqn, ppo)
from time import ctime
from typing import Any, Callable, Dict, List

from fluri.sumo.sumo_env import SumoEnv


class BaseTrainer(ABC):

    env_config_fn: Callable
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
        env: SumoEnv=None,
        gamma: float=0.95,
        learning_rate: float=0.001,
        log_level: str="ERROR",
        model_name: str=None,
        multi_policy_flag: bool=False,
        num_gpus: int=0,
        num_workers: int=0,
        out_dir: List[str]=["out"],
        policy: str="ppo",
    ) -> None:
        assert 0 <= gamma <= 1
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.log_level = log_level
        self.model_name = model_name
        self.multi_policy_flag = multi_policy_flag
        self.num_gpus = num_gpus
        self.num_workers = num_workers
        self.out_data_dir = os.path.join(out_dir + ["data"])
        self.out_model_dir = os.path.join(out_dir + ["models"])

        if not os.path.isdir(self.out_data_dir):
            os.makedirs(os.path.join(self.out_data_dir))
            self.init_dir_counter_file(self.out_data_dir)
        if not os.path.isdir(self.out_model_dir):
            os.makedirs(os.path.join(self.out_model_dir))
            self.init_dir_counter_file(self.out_model_dir)

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

    def train(self, num_rounds: int, **kwargs) -> DataFrame:
        if self.multi_policy_flag:
            self.on_multi_policy_setup()
        self.on_setup()
        for r in range(num_rounds):
            self._round = r
            self._result = self.ray_trainer.train()
            self.on_data_recording_step()
            self.on_logging_step()
            if r % self.checkpoint_freq == 0:
                self.ray_trainer.save(out_file) # TODO
        dataframe = self.on_tear_down()
        return dataframe

    def on_multi_policy_setup(self, config_args) -> None:
        dummy_env = self.env(config=self.env_config_fn(config_args))
        obs_space = dummy_env.observation_space
        act_space = dummy_env.action_space
        self.policies = {
            agent_id: (self.policy_type, obs_space, act_space)
            for agent_id in dummy_env._observe()
        }

    def on_setup(self) -> None:
        ray.init()
        self.ray_trainer = self.trainer_type(env=self.env, config=self.init_config())
        if self.model_name is None:
            out_file = os.path.join("out", "models", OUT_DIR)
        else:
            out_file = os.path.join("out", "models", OUT_DIR, self.model_name)
        self.training_data = defaultdict(list)


    def on_tear_down(self) -> DataFrame:
        self.ray_trainer.save(...) # TODO
        self.ray_trainer.stop()
        ray.shutdown()
        self.ray_trainer.local_worker.env.close()
        return DataFrame.from_dict(self.data)


    def on_logging_step(self) -> None:
        status = "[Ep. #{}] Mean reward: {:6.2f} -- Mean length: {:4.2f} -- Saved {} ({})."
        print(status.format(
            self._round+1,
            self._result["episode_reward_mean"],
            self._result["episode_len_mean"],
            self._out_file.split(os.sep)[-1],
            ctime()
        ))

    @staticmethod
    def init_dir_counter_file(self, _dir: str):
        counter_json = {
            "fedrl": {"ranked": 0, "unranked": 0},
            "marl":  {"ranked": 0, "unranked": 0},
            "sarl":  {"ranked": 0, "unranked": 0},
        }
        path = os.path.join(_dir, "counter.json")
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(counter_json, f)

    def get_count(self, _dir: str, kind: str, ranked: bool) -> int:
        path = os.path.join(_dir, "counter.json")
        with open(path, "r") as f:
            data_json = json.load(f)

        ranked = "ranked" if ranked else "unranked"
        idx = data_json[kind][ranked]
        data_json[kind][ranked] += 1
        with open(path, "w") as f:
            json.dump(data_json, f)

        return idx


    @abstractmethod
    def init_config(self) -> Dict[str, Any]:
        pass


    @abstractmethod
    def on_data_recording_step(self) -> None:
        pass