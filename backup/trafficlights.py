import networkx as nx
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

def is_cycle(graph: nx.DiGraph) -> bool:
    """Returns True if the provided graph is a cycle graph, False otherwise. A cycle 
       graph is a directed graph where each node has one edge to another node and 
       the topology resembles a circle.

    Parameters
    ----------
    graph : nx.DiGraph
        A NetworkX directed graph.

    Returns
    -------
    bool
        True if a cycle graph, False otherwise.
    """
    if not nx.is_directed(graph):
        return False
    
    if graph.number_of_edges() != graph.number_of_nodes():
        return False
    
    node, traversed_nodes, counter = set(), graph.number_of_nodes(), list(graph.nodes)[0]
    while counter:
        # Get the current node's neighbors and check if it's valid for a cycle graph.
        neighbors = list(graph.neighbors(node))
        if len(neighbors) != 1:
            return False
        
        # Go to the next node.
        traversed_nodes.add(node)
        node = neighbors[0]
        counter -= 1
        
    return len(traversed_nodes) == graph.number_of_nodes()

def in_order(state: str, next_state: str) -> bool:
    """Based on the definition of the `NEXT_STATES` constant, this function returns True
       if and only if the provided `next_state` is a valid successor to `state`, False
       otherwise. For example, `in_order("GGrr", "yyrr") -> True` and, on the contrary,
       `in_order("GGrr", "rrGG") -> False`.

    Parameters
    ----------
    state : str
        The current state (e.g., "GGrr").
    next_state : str
        The next state (e.g., "yyrr").

    Returns
    -------
    bool
        Returns True if `next_state` is a valid transition, False otherwise.
    """
    if state == next_state:
        return True
    
    if len(state) != len(next_state):
        return False
        
    for i in range(len(state)):
        if next_state[i] not in NEXT_STATES[state[i]]:
            return False
    return True


def make_tls_state_network(tls_id: str, possible_states: List[str]) -> nx.DiGraph:
    """Given a dict of the form `{'0': ['yyryyr', 'GGrGGr', 'rrGrrG', 'rryrry']}` where
       the key is the traffic light ID and the value is a list of possible states for that
       traffic light, this function returns a *directed ego network* that represents the
       possible action transitions available to a traffic light given its current action.
       
       The ego node in this network is the traffic light ID itself, so simplify the
       indexing for logic outside of this function. Since each traffic light gets its own
       subgraph, the ego node is a more succinct and logically straight-forward way to
       identify each traffic light's action subgraph withougt additional expensive 
       iteration. 

    Parameters
    ----------
    possible_states : Dict[str, List[str]]
        A dictionary where keys are the traffic light ids and the values are lists of all
        possible states the respective traffic light can take.

    Returns
    -------
    nx.DiGraph
        An ego network that corresponds with possible actions for each traffic light.
    """
    g = nx.DiGraph()
    # g.add_node(tls_id)
    for state in possible_states:
        node_id = get_node_id(tls_id, state)
        g.add_node(node_id, tls_id=tls_id, state=state)
        g.add_edge(tls_id, node_id)

    for state in possible_states:
        for next_state in possible_states:
            if in_order(state, next_state):
                u = get_node_id(tls_id, state)
                v = get_node_id(tls_id, next_state)
                g.add_edge(u, v)

    return g


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
    """
    TODO: Fill in.
    """

    def __init__(
        self, 
        tls_id: int, 
        road_netfile: str,
        sort_states: bool=SORT_DEFAULT,
        index: int=None,
        force_all_red: bool=True
    ):
        # The `index` data member is for the consistently simple indexing for actions
        # that are represented via lists. This is important for the `stable-baselines`
        # implementation that does not support Dict spaces.
        self.index = index 
        self.id = tls_id
        self.position = get_tls_position(self.id, road_netfile)
        self.possible_states = possible_tls_states(self.id, 
                                                   road_netfile, 
                                                   sort_states, 
                                                   force_all_red=force_all_red)
        self.action_transition_net = make_tls_state_network(self.id, self.possible_states)
        self.transition_net_is_cycle_graph = is_cycle(self.action_transition_net)
        self.state = random.choice([
            self.action_transition_net.nodes[u]["state"] 
            for u in self.action_transition_net.neighbors(self.id)
        ])

    def update_current_state(self) -> None:
        """Update the current state by interacting with `traci`."""
        try:
            self.state = traci.trafficlight.getRedYellowGreenState(self.id)
        except traci.exceptions.FatalTraCIError:
            pass

    def get_position(self) -> Tuple[int, int]:
        return self.position

    def get_state(self, index: int) -> str:
        return self.possible_states[index]

    def get_next_state(self, return_index: bool=True) -> Union[int, str]:
        assert self.transition_net_is_cycle_graph, "Action graph must be a cycle graph."
        
        curr_node = get_node_id(self.id, self.state)
        next_node = self.action_transition_net.neighbors(curr_node)[0]
        return next_node

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

