import gym
import numpy as np
import traci

from gym.spaces import Box
from sumo_sim import SUMOSimulation

from typing import Any, List, Tuple

"""
TODO:
+ We need to dynamically collect the names of the traffic light programs for a given
  SUMO config to support the action performance.
+ Streamline the entire SUMO/TraCI interface with `SUMOGym`.
"""

class SUMOGym(gym.Env):
    """Custom Gym environment designed for simple RL experiments using SUMO/TraCI."""
    metadata = {"render.modes": ["sumo", "sumo-gui"]}
    n_change_steps = 5
    MAIN_TASK = 0
    TRANS_TASK = 1
    GRID_KEY = "grid"
    TLS_KEY = "traffic_lights"
    
    def __init__(self, sim_config: List[str]):
        self.name = "SUMO-v1"
        self.sim_config = sim_config
        self.trafficlight_ids = None
        self.trafficlight_radius: int = 0
        self.n_trafficlights = 1
        self.n_programs = 4
        self.curr_light = np.zeros(shape=(self.n_trafficlights,))
        self.mask = np.zeros(shape=(self.n_trafficlights))
        self.bounding_box = ((0.0, 0.0), (1020.0, 1020.0))


    def __do_action(self, action) -> None:
        """
        PSEUDOCODE
            (1)
            (2)
        """
        trafficlight_that_can_change = (self.mask == 0)
        for tlsID in range(len(trafficlight_that_can_change)):
            if trafficlight_that_can_change[tlsID]:
                program_id = action[tlsID] // 2
                traci.trafficlight.setProgram(str(tlsID), str(program_id))
                self.curr_light[tlsID] = (program_id, self.MAIN_TASK)


    def __get_observation(self) -> np.ndarray:
        """TODO"""
        (x_min, y_min), (x_max, y_max) = self.bounding_box
        width = int(x_max - x_min)
        height = int(y_max - y_min)
        obs = {
            self.GRID_DICT_KEY: np.zeros(shape=(width, height), dtype=np.int8),
            self.TLS_DICT_KEY:  np.zeros(shape=(2*len(self.curr_light)), dtype=np.int8)
        }

        for veh_id in list(traci.vehicle.getIDList()):
            x, y = traci.vehicle.getPosition(veh_id)
            obs[self.GRID_DICT_KEY][int(x), int(y)] = 1

        for tls_id in range(len(self.curr_light)):
            prog_id, task_id = self.curr_light[tls_id]
            obs[self.TLS_KEY][tls_id] = 2 * prog_id + task_id

        return obs


    def __get_reward(self) -> float:
        """TODO"""
        return -1.0


    def __get_done(self) -> bool:
        """TODO"""
        return traci.simulation.getMinExpectedNumber() > 0


    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        """TODO"""
        self.__do_action(action)
        traci.simulationStep()

        observation = self.__get_observation()
        reward = self.__get_reward()
        done = self.__get_done()
        info = {}

        return observation, reward, done, info


    def reset(self):
        """TODO"""
        try:
            traci.close()
        except KeyError:
            # This basically means no SUMO experiment connected to via TraCI has started.
            # But, we do this as a precaution.
            pass
        traci.start(self.sim_config)


    @property
    def action_space(self):
        """TODO"""
        return gym.spaces.Box(low=0,
                              high=self.n_programs,
                              shape=(self.n_trafficlights,),
                              dtype=np.int32)


    @property
    def observation_space(self):
        """TODO"""
        (x_min, y_min), (x_max, y_max) = ((0.0, 0.0), (1020.0, 1020.0))
        width = int(x_max - x_min)
        height = int(y_max - y_min)

        veh_shape = (width, height)
        light_shape = (self.n_trafficlights,)

        grid_space = Box(
            low=0, 
            high=1, 
            shape=veh_shape, 
            dtype=np.int8
        )
        tls_space = Box(
            low=0, 
            high=2 * self.n_programs - 1, 
            shape=light_shape, 
            dtype=np.int8
        )

        return gym.spaces.Dict({self.GRID_KEY: grid_space, self.TLS_KEY: tls_space})


if __name__ == "__main__":
    from collections import defaultdict
    from configs.example.runner import generate_routefile
    from os.path import join

    env = SUMOGym(None)
    action = env.action_space.sample()
    obs = env.observation_space.sample()

    print(f"Action -> {action}\nObs -> {obs}")
    print(obs["traffic_lights"])
    # env.reset()

    # obs, reward, done, info = env.step(action)
    # first, generate the route file for this simulation
    generate_routefile()
    sumo_binary = "sumo"

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    path = join("flux", "sumo", "configs", "example")
    config = [sumo_binary, "-c", join(path, "traffic.sumocfg"),
                           "--tripinfo-output", join(path, "tripinfo.xml")]

    env = SUMOGym(config)
    done = False
    step = 0

    """Execute the TraCI training loop."""
    env.reset()
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        print(f"Observation {step}:\n{obs}")
        step += 1

    traci.close()