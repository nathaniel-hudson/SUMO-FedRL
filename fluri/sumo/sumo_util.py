import networkx as nx

from typing import Set

NEXT_STATES = {
    "G": set(["G", "g", "y"]),
    "g": set(["G", "g", "y"]),
    "y": set(["y", "r"]),
    "r": set(["G", "g", "r"])
}

def in_order(state: str, next_state: str) -> bool:
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


def make_tls_state_network(possible_states):
    """{'0': {'yyryyr', 'GGrGGr', 'rrGrrG', 'rryrry'}}"""
    g = nx.DiGraph()
    get_node_id = lambda pre, suf: f"{pre}:{suf}"

    for tls_id in possible_states:
        for state in possible_states[tls_id]:
            u = get_node_id(tls_id, state)
            g.add_node(u, tls_id=tls_id, state=state)

        for state in possible_states[tls_id]:
            for next_state in possible_states[tls_id]:
                if in_order(state, next_state):
                    u = get_node_id(tls_id, state)
                    v = get_node_id(tls_id, next_state)
                    g.add_edge(u, v)

    return g






# Python 3 program to print all  
# possible strings of length k 
      
# The method that prints all  
# possible strings of length k. 
# It is mainly a wrapper over  
# recursive function printAllKLengthRec() 
def printAllKLength(set, k): 
  
    n = len(set)  
    printAllKLengthRec(set, "", n, k) 
  
# The main recursive method 
# to print all possible  
# strings of length k 
def printAllKLengthRec(set, prefix, n, k): 
      
    # Base case: k is 0, 
    # print prefix 
    if (k == 0) : 
        print(prefix) 
        return
  
    # One by one add all characters  
    # from set and recursively  
    # call for k equals to k-1 
    for i in range(n): 
  
        # Next character of input added 
        newPrefix = prefix + set[i] 
          
        # k is decreased, because  
        # we have added a new character 
        printAllKLengthRec(set, newPrefix, n, k - 1) 