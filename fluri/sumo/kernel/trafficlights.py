import networkx as nx
import numpy as np
import random
import traci
import warnings
import xml.etree.ElementTree as ET

from ..utils import helper as utils
from ..utils.core import get_node_id
from ..const import *

from typing import Any, Dict, List, Set, Tuple, Union

SORT_DEFAULT = True
NEXT_STATES = {
    "G": set(["G", "g", "y"]),
    "g": set(["G", "g", "y"]),
    "y": set(["y", "r"]),
    "r": set(["G", "g", "r"])
}

def get_tls_position(
    tls_id: str, 
    road_netfile: str
) -> Tuple[str, str]:
    """This function reads the provided *.net.xml file to find the (x,y) positions of
       each traffic light (junction) in the respective road network. By default, this
       function returns a dictionary (indexed by trafficlight ID) with positions for
       every traffic light. However, the optional `tls_id` argument allows users to
       specify a single traffic light. However, the output remains constant for
       consistency (i.e., a dictionary with one item pair).

    Parameters
    ----------
    tls_id : str, optional
        The id of the traffic light the user wishes to specify, by default None

    Returns
    -------
    Tuple[str, str]
        The X/Y coordinate of the traffic light of interest in the given road network.
    """
    tree = ET.parse(road_netfile)        
    trafficlights = tree.findall("junction[@type='traffic_light']")
    x, y = None, None
    for tls in trafficlights:
        if tls.attrib["id"] == tls_id:
            x, y = tls.attrib["x"], tls.attrib["y"]
    
    if (x == None) or (y == None):
        warnings.warn("The X/Y coordinates for the position are both `None`. This "
                     f"suggests the given traffic light ID, `{tls_id}`, is invalid.")
    return (x, y)


def possible_tls_states(
    tls_id: str, 
    road_netfile: str, 
    sort_states: bool=SORT_DEFAULT,
    force_all_red: bool=False
) -> List[str]:
    """Get the possible traffic light states for the specific traffic light via the
        given `tls_id`.

    Parameters
    ----------
    tls_id : str
        The traffic light id.
    sort_states : bool, optional
        Sorts the possible states if True, by default SORT_DEFAULT

    Returns
    -------
    List[str]
        A list of all the possible states the given traffic light can take.
    """
    with open(road_netfile, "r") as f:
        tree = ET.parse(f)
        logic = tree.find(f"tlLogic[@id='{tls_id}']")
        states = [phase.attrib["state"] for phase in logic]
        
        # If specified (via `force_all_red`), there needs to be an "all reds" state where 
        # every light is red. This is an absorbing state and is not guaranteed to be in a 
        # traffic light logic. So, this bit of code ensures it is included as a possible 
        # state.
        if force_all_red:
            all_reds = len(states[0]) * "r"
            if all_reds not in states:
                states.append(all_reds)

        return states if (sort_states == False) else sorted(states)


class TrafficLight:
    """This class represents an indidivual `trafficlight` (tls) in SUMO. The purpose is to
       simplify the necessary code for the needs of the RL environments in FLURI.
    """
    def __init__(
        self, 
        tls_id: int, 
        road_netfile: str,
        sort_states: bool=SORT_DEFAULT,
        index: int=None,
        force_all_red: bool=False
    ):
        # The `index` data member is for the consistently simple indexing for actions
        # that are represented via lists. This is important for the `stable-baselines`
        # implementation that does not support Dict spaces.
        self.index: int = index 
        self.id: int = tls_id
        self.program: List[str] = possible_tls_states(
            self.id, 
            road_netfile, 
            sort_states, 
            force_all_red=force_all_red
        )
        self.num_phases: int = len(self.program)
        self.state: str = random.randrange(self.num_phases)
        self.phase: int = self.program[self.phase]

    def curr_state(self) -> None:
        """Update the current state by interacting with `traci`."""
        try:
            self.phase = traci.trafficlight.getRedYellowGreenState(self.id)
            self.state = self.program.index(self.phase)
        except traci.exceptions.FatalTraCIError:
            pass

    def next_phase(self) -> None:
        next_state = (self.state+1) % self.num_phases
        next_phase = self.program[next_state]
        try:
            traci.trafficlight.setRedYellowGreenState(self.phase)
            self.state = next_state
            self.phase = next_phase
        except traci.exceptions.FatalTraCIError:
            pass

    def valid_next_state(self, next_state: str) -> bool:
        """Determines if `next_state` is valid given the current state.

        Parameters
        ----------
        next_state : str
            The proposed next state provided by the user.

        Returns
        -------
        bool
            Returns True if `next_state` is a valid transition, False otherwise.
        """
        curr_node = get_node_id(self.id, self.state)
        next_node = get_node_id(self.id, next_state)
        is_valid = next_node in self.action_transition_net.neighbors(curr_node)
        return is_valid

    def get_observation(self) -> np.ndarray:
        """Returns an observation of the traffic light, with each of the features of 
           interest.

        Returns
        -------
        np.ndarray
            Array containing the values of each of the features.
        """
        n_vehicles, avg_speed, n_occupancy, wait_time, travel_time, n_halt = range(6)


class TrafficLightHub:
    """
    TODO: Fill in.
    """

    def __init__(
        self, 
        road_netfile: str, 
        sort_states: bool=SORT_DEFAULT, 
        obs_radius: int=None
    ):
        self.road_netfile = road_netfile
        self.ids = sorted([tls_id for tls_id in self.get_traffic_light_ids()])
        self.hub = {
            tls_id: TrafficLight(tls_id, self.road_netfile, sort_states, index=i)
            for i, tls_id in enumerate(self.ids)
        }

        # TODO: Currently not being considered.
        self.radii = None

    def get_traffic_light_ids(self) -> List[str]:
        """Get a list of all the traffic light IDs in the *.net.xml file.

        Returns
        -------
        List[str]
            A list of all the traffic light IDs.
        """
        with open(self.road_netfile, "r") as f:
            tree = ET.parse(f)
            junctions = tree.findall("junction")
            trafficlights = []
            for j in junctions:
                if j.attrib["type"] == "traffic_light":
                    trafficlights.append(j.attrib["id"])
            return trafficlights

    def update_current_states(self) -> None:
        """Update the current states by interfacing with SUMO directly using SumoKernel.
        """
        for tls in self.hub.values():
            tls.update_current_state()

    def __iter__(self):
        return self.hub.values().__iter__()

    def __getitem__(self, tls_id: str) -> TrafficLight:
        return self.hub[tls_id]

    def __len__(self) -> int:
        return len(self.ids)

