import numpy as np
import time
import traci
import warnings
import xml.etree.ElementTree as ET

from typing import Any, Dict, List, Tuple, Union

from seal.sumo.kernel.trafficlight.hub import TrafficLightHub

SORT_DEFAULT = False
VERBOSE_DEFAULT = 0


class SumoKernel():

    def __init__(self, config: Dict[str, Any]=None, ranked: bool=True):
        """Initialize a wrapper for a SUMO simulation by passing a `config` dict object
           that stores the command-line arguments needed for a SUMO simulation.

        Args:
            config (Dict[str, Any], optional): The command-line arguments required for 
                running a SUMO simulation (default None).
            ranked (bool, optional): Designates whether the SumoKernel is going to use
                ranks in its state space (default True).
        """
        self.config = {
            "gui": config.get("gui", False),
            "configuration-file": config.get("configuration-file", None),
            "net-file": config.get("net-file", None),
            "route-files": config.get("route-files", None),
            "additional-files": config.get("additional-files", None),
            "tripinfo-output": config.get("tripinfo-output", None),
        }
        self.tls_hub = TrafficLightHub(
            self.config["net-file"], 
            ranked=config.get("ranked", ranked)
        )
        # TODO: Create a function that "validates" a config so that it has everything 
        #       necessary for a SUMO simulation.


    def get_command_args(
        self,
        verbose=VERBOSE_DEFAULT,
        no_step_log: bool=True
    ) -> List[str]:
        """This generates a list of strings that are used by the TraCI API to start a
           SUMO simulation given the provided parameters that are stored in the `config`
           dict object.

        Args:
            verbose (int, optional): If the passed int value is not 0, then warnings 
                on SUMO's end will be displayed; otherwise they will be hidden 
                (default 0).
        
        Returns:
            List[str]: The command line argument list of commands to be used for starting
                SUMO.
        """
        program_cmd = "sumo-gui" if self.config["gui"] == True else "sumo"
        command_args = [program_cmd]
        if verbose == 0:
            command_args.extend(["--no-warnings", "true"])

        if no_step_log:
            command_args.extend([f"--no-step-log"])

        for cmd, args in self.config.items():
            if cmd == "gui" or args == None:
                continue
            if not isinstance(args, list):
                args = [args]
            command_args.append(f"--{cmd}")
            command_args.append(",".join(arg for arg in args))

        return command_args


    def update(self, ignore_tls: bool=False) -> None:
        """Updates the trafficlight hub objects. For the time being, this is
           NOT being used.

        Args:
            ignore_tls (bool, optional): Whether to update the trafficlights (True) or 
            not (False). Defaults to False.
        """
        if not ignore_tls:
            self.tls_hub.update()


    def is_loaded(self) -> bool:
        """Checks whether a simulation is loaded or not.

        Returns:
            bool: Returns True if a connection is loaded, False otherwise.
        """
        try:
            traci.getConnection("")
            return True
        except:
            return False


    def close(self) -> None:
        """Closes the SUMO simulation through TraCI if one is up and running."""
        if self.is_loaded():
            traci.close()


    def done(self) -> bool:
        """Returns whether or not the simulation handled by this Kernel instance is
           finished or not. This is decided if there are still some number of expected
           vehicles that have yet to complete their routes.

        Returns:
            bool: Returns True if the simulation is done, False otherwise.
        """
        return not traci.simulation.getMinExpectedNumber() > 0


    def start(self) -> None:
        """Starts or resets the simulation based on whether or not it has been started
           or not.
        """
        if self.is_loaded():
            traci.load(self.get_command_args()[1:])
        else:
            traci.start(self.get_command_args())


    def step(self) -> None:
        """Iterates the simulation to the next simulation step."""
        traci.simulationStep()


    def get_road_capacity(self) -> float:
        was_loaded = self.is_loaded()
        original_gui = self.config.get("gui", False)
        original_route_file = self.config.get("route-files", None)
        self.config["gui"] = False
        del self.config["route-files"]
        if not was_loaded:
            self.start()
        # Loop through all Lane objects and sum up their length. The conditional is
        # included to ignore INTERNAL Lanes which handle overlapping lanes (usually
        # in intersections). Since they're overlapping, they introduce redundant lengths.
        # For clarity, SUMO generates internal lanes and denotes them with ':'.
        lane_capacity = sum(traci.lane.getLength(idx) 
                            for idx in traci.lane.getIDList()
                            if not idx.startswith(":"))
        if not was_loaded:
            self.close()
        self.config["gui"] = original_gui
        self.config["route-files"] = original_route_file
        return lane_capacity


    def get_num_of_vehicles(self) -> int:
        return len(traci.vehicle.getIDList())