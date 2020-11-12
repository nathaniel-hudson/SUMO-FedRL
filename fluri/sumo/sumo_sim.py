import time
import traci
import warnings
import xml.etree.ElementTree as ET

from typing import Any, Dict, List, Set, Tuple

SORT_DEFAULT = True

class SumoSim():

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
        """This generates a list of strings that are used by the TraCI API to start a
           SUMO simulation given the provided parameters that are stored in the `config`
           dict object.
        """
        program_cmd = "sumo-gui" if self.config["gui"] == True else "sumo"
        command_args = [program_cmd]
        
        for cmd, args in self.config.items():
            if cmd == "gui" or args == None:
                continue
            if not isinstance(args, list):
                args = [args]

            command_args.append(f"--{cmd}")
            command_args.append(",".join(arg for arg in args))

        return command_args


    def get_bounding_box(self) -> Tuple[float]:
        with open(self.config["net-file"], "r") as net_file:
            # Load the provided XML file for the road network and get the location tag.
            # There should only be one location tag (hence find the first one).
            tree = ET.parse(net_file)
            location = tree.find("location")

            # Get the string attribute for the boundary box and then convert it into a 
            # Tuple of floats.
            boundary_box = location.attrib["convBoundary"] 
            return tuple([float(value) for value in boundary_box.split(",")])


    def get_traffic_light_ids(self) -> List[str]:
        """Get a list of all the traffic light IDs in the *.net.xml file."""
        with open(self.config["net-file"], "r") as net_file:
            tree = ET.parse(net_file)
            junctions = tree.findall("junction")
            trafficlights = []
            for j in junctions:
                if j.attrib["type"] == "traffic_light":
                    trafficlights.append(j.attrib["id"])
            return trafficlights


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


    def curr_tls_state(self, tls_id: str) -> List[str]:
        return traci.trafficlight.getRedYellowGreenState(tls_id)


    def get_tls_position(self, tls_id: str=None) -> Dict[str, Tuple[str, str]]:
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
        Dict[str, Tuple[str, str]]
            A dictionary where each item is (traffic light ID, [x,y] position).
        """
        tree = ET.parse(self.config["net-file"])        
        trafficlights = tree.findall("junction[@type='traffic_light']")
        positions = {
            tls.attrib["id"]: (tls.attrib["x"], tls.attrib["y"])
            for tls in trafficlights
            if (tls_id is None) or (tls.attrib["id"] == tls_id)
        }
        return positions


    def get_all_curr_tls_states(self) -> Dict[str, str]:
        """Get a dictionary containing the current states (e.g., "GGrr") of all the 
           traffic lights in the network.

        Returns
        -------
        Dict[str, str]
            Dictionary containing traffic light states.
        """
        curr_states = {}
        for tls_id in self.get_traffic_light_ids():
            curr_states[tls_id] = self.curr_tls_state(tls_id)
        return curr_states


    def get_possible_tls_states(
        self, tls_id: str, sort_states: bool=SORT_DEFAULT
    ) -> List[str]:
        """Get the possible traffic light states for the specific traffic light via the
           given `tls_id`.
        """
        tree = ET.parse(self.config["net-file"])
        logic = tree.find(f"tlLogic[@id='{tls_id}']")
        states = [phase.attrib["state"] for phase in logic]
        
        all_reds = len(states[0]) * "r"
        if all_reds not in states:
            states.append(all_reds)

        return states if (sort_states == False) else sorted(states)


    def get_all_possible_tls_states(
        self, sort_states: bool=SORT_DEFAULT
    ) -> Dict[str, List[str]]:
        """Get the possible states for all of the available traffic lights in the provided
           road network  file  (*.net.xml). The returned data  is stored in a dict  object
           where the key is the traffic light ID  (tls_id) and the values are the possible
           states.
        """
        tree = ET.parse(self.config["net-file"])
        states = {}
        for logic in tree.findall("tlLogic"):
            idx = logic.attrib["id"]
            states[idx] = self.get_possible_tls_states(idx, sort_states=sort_states)

            # for phase in logic:
            #     states[idx].append(phase.attrib["state"])
            
            # if sort_states:
            #     states[idx] = sorted(states[idx])

        return states


    def is_loaded(self) -> bool:
        """Returns a boolean based on whether a connection has been loaded (True), or not 
        (False).
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
        """Starts or resets the simulation based on whether or not it has been started
           or not."""
        if self.is_loaded():
            traci.load(self.get_command_args()[1:])
        else:
            traci.start(self.get_command_args())


    def step(self) -> None:
        """TODO"""
        traci.simulationStep()