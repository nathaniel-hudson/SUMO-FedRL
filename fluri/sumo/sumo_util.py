import networkx as nx

from typing import Any, Dict, List, Set

NEXT_STATES = {
    "G": set(["G", "g", "y"]),
    "g": set(["G", "g", "y"]),
    "y": set(["y", "r"]),
    "r": set(["G", "g", "r"])
}

def in_order(state: str, next_state: str) -> bool:
    """Based on the definition of the `NEXT_STATES` constant, this function returns True
       if and only if the provided `next_state` is a valid successor to `state`, False
       otherwise. 
       
       For example, `in_order("GGrr", "yyrr") -> True` and, on the contrary,
       `in_order("GGrr", "rrGG") -> False`.
    """
    if state == next_state:
        return True
    
    if len(state) != len(next_state):
        return False
        
    for i in range(len(state)):
        if next_state[i] not in NEXT_STATES[state[i]]:
            return False
    return True

def get_possible_next_states(state) -> Set[str]:
    states = set()


def get_node_id(pre: Any, suf:str) -> str:
    return f"{pre}:{suf}"

def make_tls_state_network(possible_states: Dict[str, List[str]]):
    """Given a dict of the form `{'0': ['yyryyr', 'GGrGGr', 'rrGrrG', 'rryrry']}` where
       the key is the traffic light ID and the value is a list of possible states for that
       traffic light, this function returns a *directed ego network* that represents the
       possible action transitions available to a traffic light given its current action.
       
       The ego node in this network is the traffic light ID itself, so simplify the
       indexing for logic outside of this function. Since each traffic light gets its own
       subgraph, the ego node is a more succinct and logically straight-forward way to
       identify each traffic light's action subgraph withougt additional expensive 
       iteration. 
    """
    g = nx.DiGraph()

    for tls_id in possible_states:
        g.add_node(tls_id)
        for state in possible_states[tls_id]:
            node_id = get_node_id(tls_id, state)
            g.add_node(node_id, tls_id=tls_id, state=state)
            g.add_edge(tls_id, node_id)

        for state in possible_states[tls_id]:
            for next_state in possible_states[tls_id]:
                if in_order(state, next_state):
                    u = get_node_id(tls_id, state)
                    v = get_node_id(tls_id, next_state)
                    g.add_edge(u, v)

    return g
