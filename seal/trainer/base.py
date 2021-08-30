import seal.trainer.defaults as defaults
import os
import pickle
import ray

from abc import ABC, abstractmethod
from collections import defaultdict
from pandas import DataFrame
from ray.rllib.agents import (a3c, dqn, ppo)
from ray.rllib.agents.callbacks import DefaultCallbacks
from time import ctime
from typing import Any, Callable, Dict, List, Tuple

from seal.trainer.counter import Counter
from seal.trainer.defaults import *
from seal.trainer.util import *
from seal.sumo.abstract_env import AbstractSumoEnv

RAY_TRAINER_SEED = 54321


class BaseTrainer(ABC):

    communication_callback_cls: DefaultCallbacks
    counter: Counter
    idx: int
    num_gpus: int
    env: AbstractSumoEnv
    learning_rate: float
    log_level: str
    gamma: float
    num_gpus: int
    num_workers: int
    out_checkpoint_dir: str
    out_data_dir: str
    out_weights_dir: str
    policy: str
    policy_mapping_fn: Callable
    policy_type: Any  # TODO: Change this.
    trainer_type: Any  # TODO: Change this.

    def __init__(
        self,
        checkpoint_freq: int=25,
        env: AbstractSumoEnv=None,
        gamma: float=0.95,
        learning_rate: float=0.001,
        log_level: str="ERROR",
        model_name: str=None,
        num_gpus: int=0,
        num_workers: int=0,
        root_dir: List[str]=["out"],
        sub_dir: str=None,
        policy: str="ppo",
        **kwargs
    ) -> None:
        assert 0 <= gamma <= 1
        self.communication_callback_cls = None
        self.checkpoint_freq = checkpoint_freq
        self.counter = Counter()
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.log_level = log_level
        self.model_name = model_name
        self.num_gpus = num_gpus
        self.num_workers = num_workers

        self.out_checkpoint_dir = os.path.join(*(root_dir + ["checkpoints"]))
        self.out_data_dir = os.path.join(*(root_dir + ["data"]))
        self.out_weights_dir = os.path.join(*(root_dir + ["weights"]))
        if sub_dir is not None:
            self.out_checkpoint_dir = os.path.join(
                self.out_checkpoint_dir, sub_dir)
            self.out_data_dir = os.path.join(self.out_data_dir, sub_dir)
            self.out_weights_dir = os.path.join(self.out_weights_dir, sub_dir)

        self.gui = kwargs.get("gui", defaults.GUI)
        self.net_file = kwargs.get("net_file", defaults.NET_FILE)
        self.ranked = kwargs.get("ranked", defaults.RANKED)
        self.rand_routes_on_reset = kwargs.get("rand_routes_on_reset",
                                               defaults.RAND_ROUTES_ON_RESET)
        self.rand_routes_config = kwargs.get("rand_routes_config",
                                             defaults.RAND_ROUTES_CONFIG)

        self.net_dir = self.net_file.split(os.sep)[-1].split(".")[0]
        self.out_checkpoint_dir = os.path.join(
            self.out_checkpoint_dir, self.net_dir)
        self.out_data_dir = os.path.join(self.out_data_dir, self.net_dir)
        self.out_weights_dir = os.path.join(self.out_weights_dir, self.net_dir)

        if not os.path.isdir(self.out_checkpoint_dir):
            os.makedirs(os.path.join(self.out_checkpoint_dir))
        if not os.path.isdir(self.out_data_dir):
            os.makedirs(os.path.join(self.out_data_dir))
        if not os.path.isdir(self.out_weights_dir):
            os.makedirs(os.path.join(self.out_weights_dir))

        self.policy = policy
        self.__load_policy_type()

        self.trainer_name = None
        self.idx = None
        self.policy_config = None
        self.policy_mapping_fn = None

    # ------------------------------------------------------------------------------ #

    def load(self, checkpoint: str) -> None:
        if type(self) is BaseTrainer:
            raise NotImplementedError("Cannot load policy using abstract `BaseTrainer` "
                                      "class.")
        self.on_setup()
        self.ray_trainer.restore(checkpoint)

    # ------------------------------------------------------------------------------ #

    def train(self, num_rounds: int, save_on_end: bool=True, **kwargs) -> DataFrame:
        if kwargs.get("checkpoint", None) is not None:
            self.load(kwargs["checkpoint"])
        else:
            self.policies = self.on_policy_setup()
            if GLOBAL_POLICY_VAR in self.policies:
                raise ValueError(f"Sub-classes of `BaseTrainer` cannot have "
                                 f"policies with key '{GLOBAL_POLICY_VAR}'.")
            else:
                temp = next(iter(self.policies.values()))
                self.policies[GLOBAL_POLICY_VAR] = temp
            self.on_setup()
        for r in range(num_rounds):
            self._round = r
            self._result = self.ray_trainer.train()
            self.on_data_recording_step()
            self.on_logging_step()
            # TODO: Seeding step using `r` for random route generation.
            if r % self.checkpoint_freq == 0:
                self.ray_trainer.save(self.model_path)
        # Get the global test policy weights and then save them to a PICKLE file object.
        # This will then be used to reload the test policy's weights for evaluation
        # in both the synthetic simulations and real-world implementation.
        weights = self.on_make_final_policy()
        ranked_str = "ranked" if self.ranked else "unranked"
        with open(os.path.join(self.out_weights_dir, f"{ranked_str}.pkl"), "wb") as f:
            pickle.dump(weights, f)
        self.ray_trainer.get_policy(GLOBAL_POLICY_VAR).set_weights(weights)
        # Get the data from the training process and output it for visualization to see
        # how training performed over time.
        dataframe = self.on_tear_down()
        if save_on_end:
            path = os.path.join(self.out_data_dir, f"{self.get_filename()}.csv")
            dataframe.to_csv(path)
        return dataframe

    # ------------------------------------------------------------------------------ #

    def __load_policy_type(self) -> None:
        if self.policy == "a3c":
            self.trainer_type = a3c.A3CTrainer
            self.policy_type = a3c.a3c_torch_policy
        elif self.policy == "dqn":
            self.trainer_type = dqn.DQNTrainer
            self.policy_type = dqn.DQNTorchPolicy
        elif self.policy == "ppo":
            self.trainer_type = ppo.PPOTrainer
            self.policy_type = ppo.PPOTorchPolicy
        else:
            raise NotImplemented(f"Do not support policies for `{policy}`.")

    # ------------------------------------------------------------------------------ #

    def init_config(self) -> Dict[str, Any]:
        return {
            "env_config": self.env_config_fn(),
            "framework": "torch",
            "log_level": self.log_level,
            "lr": self.learning_rate,
            "multiagent": {
                "policies": self.policies,
                "policy_mapping_fn": self.policy_mapping_fn
            },
            "num_gpus": self.num_gpus,
            "num_workers": self.num_workers,
            "seed": RAY_TRAINER_SEED,
            "callbacks": self.communication_callback_cls,
        }

    def env_config_fn(self) -> Dict[str, Any]:
        return {
            "gui": self.gui,
            "net-file": self.net_file,
            "rand_routes_on_reset": self.rand_routes_on_reset,
            "ranked": self.ranked,
            "use_dynamic_seed": True,
            "rand_route_args": {
                "seed": 0,
                "vehicles_per_lane_per_hour": 90
            }
        }

    # ------------------------------------------------------------------------------ #

    def on_setup(self) -> None:
        ray.init()
        self.ray_trainer = self.trainer_type(env=self.env,
                                             config=self.init_config())
        self.model_path = os.path.join(
            self.out_checkpoint_dir, self.get_filename())
        self.training_data = defaultdict(list)

    def on_tear_down(self) -> DataFrame:
        self.ray_trainer.save(self.model_path)
        self.ray_trainer.stop()
        ray.shutdown()
        self.ray_trainer.workers.local_worker().env.close()
        return DataFrame.from_dict(self.training_data)

    def on_logging_step(self) -> None:
        status = "[Ep. #{}] Mean reward: {:6.2f}, Mean length: {:4.2f}, Saved {} ({})."
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
        return f"{ranked}"
        # return f"{ranked}_{self.idx}"

    def get_weights_filename(self) -> str:
        ranked = "ranked" if self.ranked else "unranked"
        return f"{ranked}"

    def set_rand_route_seed(self, seed) -> None:
        self.env

    # ------------------------------------------------------------------------------ #

    @abstractmethod
    def on_make_final_policy() -> Weights:
        """This function is to be used for defining the weights used for the final policy
           to be used during evaluation. Each Trainer sub-class will come up with their
           own way for doing this procedure. For instance, simply grabbing one of the
           trained policies at random and returning its weights is sufficient (though
           likely not a desirable approach). The returned weights will then be used to
           in the GLOBAL policy that evaluation will be used.

        Raises:
            NotImplementedError: Cannot be called for the abstract BaseTrainer class.

        Returns:
            Weights: The model weights to be used in the GLOBAL model for evaluation.
        """
        raise NotImplementedError("Must implement abstract function "
                                  "`on_make_final_policy`.")

    @abstractmethod
    def on_data_recording_step(self) -> None:
        raise NotImplementedError("Must implement abstract function "
                                  "`on_data_recording_step`.")

    @abstractmethod
    def on_policy_setup(self) -> Dict[str, Tuple[Any]]:
        raise NotImplementedError("Must implement abstract function "
                                  "`on_policy_setup`.")
