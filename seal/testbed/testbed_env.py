import gym
import numpy as np

from ray.rllib.env import MultiAgentEnv
from gym.spaces import *
from typing import List, Tuple


class TestbedEnv(MultiAgentEnv):
    """
    This is a possible option for constructing the environment to work
    with the testbed. Though, it may be less straightforward and introduce
    more room for error. But, we have this skeleton setup for implementation
    if we choose to go this way.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.ranked = config.get("ranked", DEFUALT_RANKED)

        # Check if the user provided a route-file to be used for simulations and if the
        # user wants random routes to be generated for EACH trial (rand_routes_on_reset).
        # If user's want random routes generated (i.e., "route-files" is None in provided
        # config), then a private flag, `__first_rand_routes_flag`, is set to True. This
        # forces the `reset()` function to generate at least ONE random route file before
        # being updated to False. This will never change back to True (hence the privacy).

        self.rand_routes_on_reset = False
        self.__first_rand_routes_flag = False

        self.kernel = TestbedKernel(self.config)
        self.action_timer = ActionTimer(len(self.kernel.tls_hub))
        self.reset()
    def step(self, action: List[int]) -> Tuple:
        pass

    def _do_action(self, action: List[int]) -> List[int]:
        pass

    def _get_reward(self, obs: np.ndarray) -> float:
        pass

    def _observe(self) -> np.array:
        pass

    @property
    def action_space(self) -> spaces.Space:
        return spaces.Dict({})

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Dict({})