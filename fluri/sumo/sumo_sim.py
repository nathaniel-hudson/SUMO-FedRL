import traci
import warnings
import xml.etree.ElementTree as ET

from typing import Any, Dict, List, Set

class SUMOSim():

    def __init__(self, config: Dict[str, Any]=None):
        """Initialize a wrapper for a SUMO simulation by passing a `config` dict object
           that stores the command-line arguments needed for a SUMO simulation.

        Parameters
        ----------
        config : Dict[str, Any], optional
            The command-line arguments required for running a SUMO simulation, by default 
            None.
        """
        self.config = {
            "gui": config.get("gui", False),
            "configuration-file": config.get("configuration-file", None),
            "net-file": config.get("net-file", None),
            "route-files": config.get("route-files", None),
            "additional-files": config.get("additional-files", None),
            "tripinfo-output": config.get("tripinfo-output", None),
        }

    def get_command_args(self) -> List[str]:
        """TODO"""
        program_cmd = "sumo-gui" if self.config["gui"] == True else "sumo"
        command_args = [program_cmd]
        for cmd, args in self.config.items():
            if cmd == "gui" or args == None:
                continue
            if not isinstance(args, list):
                args = [args]

            command_args.append(f"--{cmd}")
            command_args.append(",".join(arg for arg in args))
            # for arg in args:
            #     command_args.append(arg)
        return command_args

    def get_traffic_lights(self) -> List[str]:
        """TODO"""
        return list(traci.trafficlight.getIDList())

    def get_traffic_programs(
        self, traffic_programs: str=None, key: str="programID"
    ) -> List[str]:
        """Reads the XML file for the traffic programs and returns the IDs of traffic 
           programs.
        """
        tree = ET.parse(traffic_programs)
        return [tlLogic.attrib[key] for tlLogic in tree.findall("tlLogic")]

    def generate_routes(self):
        """TODO"""
        pass

    def get_curr_tls_state(self, tls_id: str) -> List[str]:
        return traci.trafficlight.getRedYellowGreenState(tls_id)

    def get_possible_tls_states(self, tls_id: str, sort_states: bool=True) -> str:
        """Get the possible traffic light states for the specific traffic light via the
           given `tls_id`.
        """
        states = []
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
        for phase in logic.phases:
            states.append(phase.state)
        
        if sort_states:
            return sorted(states)
        else:
            return states

    def get_all_curr_tls_states(self) -> Dict[str, str]:
        curr_states = {}
        for tls_id in self.get_traffic_lights():
            curr_states[tls_id] = self.get_curr_tls_state(tls_id)
        return curr_states

    def get_all_possible_tls_states(self, sort_states: bool=True) -> Dict[str, List[str]]:
        """Get the possible states for all of the available traffic lights in the provided
           road network file (*.net.xml). The returned data is stored in a dict object
           where the key is the traffic light ID (tls_id) and the values are the possible
           states.
        """
        all_states = {}
        for tls_id in self.get_traffic_lights():
            all_states[tls_id] = self.get_possible_tls_states(tls_id)
        return all_states


    def is_loaded(self) -> bool:
        """Returns a boolean based on whether or not a connection has been loaded (True),
           or not (False).
        """
        try:
            traci.getConnection("")
            return True
        except:
            return False

    def close(self) -> None:
        """TODO"""
        if self.is_loaded():
            traci.close()

    def done(self) -> bool:
        """TODO"""
        return not traci.simulation.getMinExpectedNumber() > 0

    def start(self) -> None:
        """TODO"""
        if not self.is_loaded():
            traci.start(self.get_command_args())
        else:
            warnings.warn("Simulation already started.")

    def step(self) -> None:
        """TODO"""
        traci.simulationStep()



if __name__ == "__main__":
    from os.path import join

    # sim = SUMOSim()
    # filename = join("configs", "example", "tls_program.add.xml")
    # logics = sim.get_traffic_programs(filename)
    # print(logics)

    path = join("configs", "example")
    sim = SUMOSim(config={
        "gui": False,
        "net-file": join(path, "traffic.net.xml"),
        "route-files": join(path, "traffic.rou.xml"),
        "additional-files": join(path, "traffic.det.xml"),
        "tripinfo-output": join(path, "tripinfo.xml")
    })
    
    print(sim.get_command_args())
    # exit(0)

    sim.start()
    sim.start()
    if sim.is_loaded():
        print(sim.get_traffic_lights())
    else:
        print("> Simulation is not started!")

    while not sim.done():
        sim.step()

    ## TODO: We need to figure out how to generate ALL possible states for a given traffic
    ##       light. This is going to be difficult. Umm, I'm not sure if the possible 
    ##       states are exoplicitly restricted in the .net.xml file or not.
    print(f"getRedYellowGreenState() -> {traci.trafficlight.getRedYellowGreenState('0')}")
    print(f"controlled lanes -> {traci.trafficlight.getControlledLanes('0')}")
    print(f"controlled links -> {traci.trafficlight.getControlledLinks('0')}")
    print(f"current state -> {sim.get_all_curr_tls_states()}")
    possible_states = sim.get_all_possible_tls_states()
    print(f"all possible states -> {possible_states}") 
    sim.close()

    import matplotlib.pyplot as plt
    import networkx as nx
    
    from sumo_util import make_tls_state_network
    from networkx.drawing.nx_agraph import to_agraph 
    g = make_tls_state_network(possible_states)
    A = to_agraph(g)
    A.layout("dot")
    A.draw("delete.png")

    nx.draw_networkx(g)
    plt.show()
    print(g.edges())