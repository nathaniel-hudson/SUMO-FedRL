import random

from . import sumo_util as utils
from .const import *
from .sumo_sim import SumoSim

from typing import Dict

class TrafficLights:

    def __init__(self, sim: SumoSim):
        self.sim = sim
        self.ids = self.sim.get_traffic_light_ids()
        self.states = self.sim.get_all_possible_tls_states()
        self.network = utils.make_tls_state_network(self.states)
        self.num = len(self.ids)
        self.curr_states = self.random_states()

        # TODO: Currenlty not being considered.
        self.radii = None

    def random_states(self) -> Dict[str, str]:
        """Initialize a random state for all the traffic lights in the network.

        Returns
        -------
        Dict[str, str]
            Random state.
        """
        return {
            tls_id: random.choice([self.network.nodes[u]["state"] 
                                   for u in self.network.neighbors(tls_id)])
            for tls_id in self.ids
        }

    def update_curr_states(self) -> None:
        """Set the current state by interfacing with SUMO directly using SumoSim.
        """
        self.curr_states = self.sim.get_all_curr_tls_states()