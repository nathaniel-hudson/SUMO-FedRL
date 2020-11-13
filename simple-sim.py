
import matplotlib.pyplot as plt
import networkx as nx
import traci

from fluri.sumo.sumo_sim import SumoSim
from fluri.sumo.utils.helper import make_tls_state_network
from networkx.drawing.nx_agraph import to_agraph 
from os.path import join

def main() -> None:

    path = join("configs", "example")
    sim = SumoSim(config={
        "gui": False,
        "net-file": join(path, "traffic.net.xml"),
        "route-files": join(path, "traffic.rou.xml"),
        "additional-files": join(path, "traffic.det.xml"),
        "tripinfo-output": join(path, "tripinfo.xml")
    })
    
    print(sim.get_command_args())

    sim.start()

    while not sim.done():
        sim.step()

    # Print some basic information related to the simulation configuration.
    print(sim.get_tls_position())
    # print(f"trafficlights -> {sim.get_traffic_light_ids()}")
    # print(f"boundaryBox() -> {sim.get_bounding_box()}")
    # print(f"getRedYellowGreenState() -> {traci.trafficlight.getRedYellowGreenState('0')}")
    # print(f"controlled lanes -> {traci.trafficlight.getControlledLanes('0')}")
    # print(f"controlled links -> {traci.trafficlight.getControlledLinks('0')}")
    # print(f"current state -> {sim.get_all_curr_tls_states()}")
    # print(f"get_possible_tls_states('0') -> {sim.get_possible_tls_states('0')}")
    # possible_states = sim.get_all_possible_tls_states()
    # print(f"get_all_possible_tls_states() -> {sim.get_all_possible_tls_states()}")
    sim.close()

    # Make the action graph for the single traffic light and draw it.
    # g = make_tls_state_network(possible_states)    
    # A = to_agraph(g)
    # A.layout("dot")
    # A.draw("action-graph.png")

if __name__ == "__main__":
    main()