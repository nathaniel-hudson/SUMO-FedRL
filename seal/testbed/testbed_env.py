import gym
import numpy as np
import rospy
from gym import spaces
from typing import *
from ray.rllib.env import MultiAgentEnv
from seal.sumo.config import *
from seal.testbed.kernel.kernel import TestbedKernel
from seal.testbed.timer import ActionTimer
from seal.testbed.rosmanager import ROSManager


class TestbedEnv(MultiAgentEnv):
    """
    This is a possible option for constructing the environment to work
    with the testbed. Though, it may be less straightforward and introduce
    more room for error. But, we have this skeleton setup for implementation
    if we choose to go this way.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.path = config["net-file"]
        self.ranked = config["ranked"]
        self.rand_routes_on_reset = False
        self.__first_rand_routes_flag = False

        rospy.init_node("n", anonymous = True)
        self.pynode = ROSManager()
        lights = self.pynode.get_network(config["net-file"])	#got a list of trafffic lights (IDs)
        self.pynode.init_publishers(lights)					#initialized publishers for each light_ids
        rospy.loginfo("agent says Hi! from pynode of TestbedEnv")

        self.kernel = TestbedKernel(self.config, self.ranked, lights, self.pynode)    #TODO
        self.action_timer = ActionTimer(len(lights))   #TODO
        self.reset()

    def reset(self) -> Any:
        """Start the simulation and get details surrounding the world.

        Returns
        -------
        Any
            The observation of the state space upon resetting the simulation/environment.
        """
        if self.rand_routes_on_reset or self.__first_rand_routes_flag:
            self.rand_routes()
            self.__first_rand_routes_flag = False
        self.kernel.start()
        self.action_timer.restart()
        return self._observe()

    def rand_routes(self) -> None:
        net_name = self.config["net-file"]
        rand_args = self.config.get("rand_route_args", dict())
        # NOTE: Simplifies process, so leave this for now.
        rand_args["n_routefiles"] = 1
        generate_random_routes(net_name=net_name, path=self.path, **rand_args)
        # TODO: Implement dynamic seed.

    def close(self) -> None:
        self.kernel.close()

    @property
    def multi_action_space(self) -> spaces.Space:
        return spaces.Dict({
            idx: self.kernel.tls_hub[idx].action_space
            for _, idx in self.kernel.tls_hub.index2id.items()
        })

    @property
    def action_space(self) -> spaces.Space:
        """This is the action space defined for a *single* traffic light. It is
           defined this way to support RlLib more easily.

        Returns:
            Space: Action space for a single traffic light.
        """
        first = self.kernel.tls_hub.ids[0]
        return self.kernel.tls_hub[first].action_space

    @property
    def observation_space(self) -> spaces.Space:
        """This is the observation space defined for a *single* traffic light. It is
           defined this way to support RlLib more easily.

        Returns:
            Space: Observation space for a single traffic light.
        """
        first = self.kernel.tls_hub.ids[0]
        return self.kernel.tls_hub[first].observation_space

    def action_spaces(self, tls_id) -> spaces.Space:
        return self.kernel.tls_hub[tls_id].action_space

    def observation_spaces(self, tls_id) -> spaces.Space:
        return self.kernel.tls_hub[tls_id].observation_space

    def step(self, action_dict: Dict[Any, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        taken_action = self._do_action(action_dict)
        self.kernel.step()

        obs = self._observe()
        reward = {
            tls.id: self._get_reward(obs[tls.id])
            for tls in self.kernel.tls_hub
        }
        done = {"__all__": self.kernel.done()}
        info = {"taken_action": taken_action,
                "total_reward": sum(reward.values())}
        info = {}

        return obs, reward, done, info

    def _do_action(self, actions: Dict[Any, int]) -> List[int]:
        """Perform the provided action for each trafficlight.

        Args:
            actions (Dict[Any, int]): The action that each trafficlight will take

        Returns:
            Dict[Any, int]: Returns the action taken -- influenced by which moves are
                legal or not.
        """
        taken_action = actions.copy()
        for tls in self.kernel.tls_hub:
            if actions[tls.id] == 1 and self.action_timer.can_change(int(tls.id)):
                tls.next_phase()
                self.action_timer.restart(int(tls.id))
            else:
                self.action_timer.decr(int(tls.id))
                taken_action[tls.id] = 0
        return taken_action

    def _get_reward(self, obs: np.ndarray) -> float:
        """Negative reward function based on the number of halting vehicles, waiting time,
           and travel time.

        Parameters
        ----------
        obs : np.ndarray
            Numpy array (containing float64 values) representing the observation.

        Returns
        -------
        float
            The reward for this step
        """
        return -(obs[LANE_OCCUPANCY] + obs[HALTED_LANE_OCCUPANCY])

    # def _observe(self) -> Dict[Any, np.ndarray]:
    #     """Get the observations across all the trafficlights, indexed by trafficlight id.

    #     Returns
    #     -------
    #     Dict[Any, np.ndarray]
    #         Observations from each trafficlight.
    #     """
    #     obs = {tls.id: tls.get_observation() for tls in self.kernel.tls_hub}
    #     if self.ranked:
    #         # print('ranked')
    #         self._get_ranks(obs)
    #     return obs

    def _observe(self) -> Dict[Any, np.ndarray]:
        """Get the observations across all the trafficlights, indexed by trafficlight id.

        Returns
        -------
        Dict[Any, np.ndarray]
            Observations from each trafficlight.
        """
        obs = {tls.id: tls.get_observation() for tls in self.kernel.tls_hub}
        if self.ranked:
            self._get_ranks(obs, halted=False)
            self._get_ranks(obs, halted=True)
        # Clean the observation of NaN and (+/-) Inf values.
        for tls in obs:
            for i in range(len(obs[tls])):
                feature = obs[tls][i]
                if feature == np.nan or feature == float('-inf'):
                    obs[tls][i] = 0.0
                elif feature == float('inf'):
                    obs[tls][i] = 1.0
        return obs


    # def _get_ranks(self, obs: Dict) -> None:    #TODO
    #     """Appends global and local ranks to the observations in an inplace fashion.

    #     Args:
    #         obs (Dict): Observation provided by a trafficlight.
    #     """
    #     # print(obs[0])
    #     pairs = [(tls_id, tls_state[LANE_OCCUPANCY])
    #              for tls_id, tls_state in obs.items()]
    #     pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    #     graph = self.kernel.tls_hub.tls_graph  # Adjacency list representation.

    #     # Calculate the GLOBAL ranks for each tls in the road network.
    #     for global_rank, (tls_id, _) in enumerate(pairs):
    #         try:
    #             obs[str(tls_id)][GLOBAL_RANK] = 1 - (global_rank / (len(graph)-1))
    #         except ZeroDivisionError:
    #             obs[str(tls_id)][GLOBAL_RANK] = 1

    #     # Calculate LOCAL ranks based on global ranks from above.
    #     for tls_id in graph:
    #         local_rank = 0
    #         for neighbor in graph[tls_id]:
    #             # print(type(tls_id), type(neighbor))
    #             if obs[str(tls_id)][GLOBAL_RANK] > obs[str(neighbor)][GLOBAL_RANK]:
    #                 local_rank += 1
    #         try:
    #             # We do *not* subtract the denominator by 1 (as we do with global
    #             # rank) because `len(graph[tls_id])` does not include `tls_id` as a
    #             # node in the sub-network when it should be included. This means that
    #             # +1 node cancels out the -1 node.
    #             print(graph)
    #             obs[str(tls_id)][LOCAL_RANK] = 1 - \
    #                 (local_rank / len(graph[tls_id]))
    #         except ZeroDivisionError:
    #             obs[str(tls_id)][LOCAL_RANK] = 1

    def _get_ranks(self, obs: Dict, halted: bool=False) -> None:
        """Appends global and local ranks to the observations in an inplace fashion.

        Args:
            obs (Dict): Observation provided by a trafficlight.
        """
        if halted:
            print("obs: ", obs)
            pairs = [(tls, tls_state[HALTED_LANE_OCCUPANCY])
                     for tls, tls_state in obs.items()]
            local_index = LOCAL_HALT_RANK
            global_index = GLOBAL_HALT_RANK
        else:
            print("obs: ", obs)
            pairs = [(tls, tls_state[LANE_OCCUPANCY])
                     for tls, tls_state in obs.items()]
            local_index = LOCAL_RANK
            global_index = GLOBAL_RANK
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        graph = self.kernel.tls_hub.tls_graph  # Adjacency list representation.

        # Calculate the GLOBAL ranks for each tls in the road network.
        for global_rank, (tls, _) in enumerate(pairs):
            try:
                obs[tls][global_index] = 1 - (global_rank / (len(graph)-1))
            except ZeroDivisionError:
                obs[tls][global_index] = 1

        # Calculate LOCAL ranks based on global ranks from above.
        for tls in graph:
            local_rank = 0
            for neighbor in graph[tls]:
                if obs[str(tls)][global_index] > obs[str(neighbor)][global_index]:
                    local_rank += 1
            try:
                # We do *not* subtract the denominator by 1 (as we do with global
                # rank) because `len(graph[tls])` does not include `tls` as a
                # node in the sub-network when it should be included. This means that
                # +1 node cancels out the -1 node.
                obs[str(tls)][local_index] = 1 - (local_rank/len(graph[tls]))
            except ZeroDivisionError:
                obs[str(tls)][local_index] = 1



        # Check if the user provided a route-file to be used for simulations and if the
        # user wants random routes to be generated for EACH trial (rand_routes_on_reset).
        # If user's want random routes generated (i.e., "route-files" is None in provided
        # config), then a private flag, `__first_rand_routes_flag`, is set to True. This
        # forces the `reset()` function to generate at least ONE random route file before
        # being updated to False. This will never change back to True (hence the privacy).

        # if self.config.get("route-files", None) is None:
        #     self.config["route-files"] = os.path.join(
        #         self.path, "traffic.rou.xml")
        #     self.rand_routes_on_reset = self.config.get(
        #         "rand_routes_on_reset", True)
        #     self.__first_rand_routes_flag = True
        # else:
        #     self.rand_routes_on_reset = False