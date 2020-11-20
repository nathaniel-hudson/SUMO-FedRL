import gym
import numpy as np
import traci

from gym import spaces
from typing import Tuple

from .sumo_env import SumoEnv
from .kernel.kernel import SumoKernel

class MultiSumoEnv(SumoEnv):

    def __init__(self, sim: SumoKernel, world_dim: Tuple[int, int] = None):
        super().__init__(sim, world_dim)

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        n_traffic_lights = 10
        return spaces.MultiDiscrete([5
                                     for i in range(n_traffic_lights)])

    @property
    def observation_Space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=10,
            shape=self.get_obs_dims(),
            dtype=np.int8
        )

    def reset(self):
        obs_n = dict()
        reward_n = dict()
        done_n = dict()
        info_n = {"n": []}
        # ...
        return obs_n, reward_n, done_n, info_n

    def step(self, actions: dict):
        # Perform given actions for each agent and then take ONE simulation step in SUMO.
        for agent in self.agent:
            self._do_action(agent, actions[agent])
        self.sim.step()
        self.__update_world()

    def _do_action(self, agent_id, action):
        pass

    def _get_world(self, agent_id) -> np.ndarray:
        pass

    def _get_reward(self, agent_id) -> float:
        pass

    def __update_world(self) -> None:
        """To (efficiently) get an accurate view of each agents' observation space, this 
           function simply updates the cached view of the entire world's state. This is
           then used to grab the sub-matrices of the world to represent each agents'
           view or observation subspace.
        """   
        sim_h, sim_w = self.get_sim_dims()
        obs_h, obs_w = self.get_obs_dims()
        h_scalar = obs_h / sim_h
        w_scalar = obs_w / sim_w
        world = np.zeros(shape=(obs_h, obs_w), dtype=np.int32)

        veh_ids = list(traci.vehicle.getIDList())
        for veh_id in veh_ids:
            # Get the (scaled-down) x/y coordinates for the observation world.
            x, y = traci.vehicle.getPosition(veh_id)
            x, y = int(x * w_scalar), int(y * h_scalar)

            # Add a normalized weight to the respective coordinate in the world. For it to
            # be normalized, we need to change `dtype` to a float-based value.
            world[y, x] += 1

        self.world = world
