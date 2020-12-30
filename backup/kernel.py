import numpy as np
import time
import traci
import warnings
import xml.etree.ElementTree as ET

from typing import Any, Dict, List, Tuple, Union

from .trafficlights import TrafficLight, TrafficLightHub
from .world import World

from ..utils.random_routes import generate_random_routes

SORT_DEFAULT = True
VERBOSE_DEFAULT = 0

class SumoKernel():

    def __init__(self, config: Dict[str, Any]=None, scale_factor: float=1.0):
        """Initialize a wrapper for a SUMO simulation by passing a `config` dict object
           that stores the command-line arguments needed for a SUMO simulation.
        Parameters
        ----------
        config : Dict[str, Any], optional
            The command-line arguments required for running a SUMO simulation, by default 
            None.
        """

        '''
        ITEMS THAT WE NEED:
            + Road network file (*.net.xml).
            + Whether we use GUI or not.
            + Route file (if needed).
            + Additional files (likely not really needed).
        '''

        assert 0.0 < scale_factor and scale_factor <= 1.0

        # TODO: Create a function that "validates" a config so that 
        #       it has everything necessary for a SUMO simulation.
        self.config = {
            "gui": config.get("gui", False),
            "configuration-file": config.get("configuration-file", None),
            "net-file": config.get("net-file", None),
            "route-files": config.get("route-files", None),
            "additional-files": config.get("additional-files", None),
            "tripinfo-output": config.get("tripinfo-output", None),
        }
        self.world = World(self.config["net-file"], scale_factor)
        self.tls_hub = TrafficLightHub(self.config["net-file"]) # TODO: Slow as hell.


    def get_command_args(self, verbose=VERBOSE_DEFAULT) -> List[str]:
        """This generates a list of strings that are used by the TraCI API to start a
           SUMO simulation given the provided parameters that are stored in the `config`
           dict object.
           Parameters
           ----------
           verbose : int, optional
               If the passed int value is not 0, then warnings on SUMO's end will be 
               displayed; otherwise they will be hidden (default 0).
        """
        program_cmd = "sumo-gui" if self.config["gui"] == True else "sumo"
        command_args = [program_cmd]
        if verbose == 0:
            command_args.extend(["--no-warnings", "true"])
        
        for cmd, args in self.config.items():
            if cmd == "gui" or args == None:
                continue
            if not isinstance(args, list):
                args = [args]

            command_args.append(f"--{cmd}")
            command_args.append(",".join(arg for arg in args))

        return command_args


    def generate_routes(self):
        """TODO"""
        pass


    def get_tls_observations(
        self, 
        obs_width: float=0.2,
        return_dict: bool=True,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        assert 0 < obs_width and obs_width <= 1

        observations = {} if return_dict else []
        width = int(self.world.get_dimensions()[0] * obs_width)
        for tls in self.tls_hub:
            (x, y) = self.world.convert_coords(tls.get_position())
            top_left = (x - width, y - width)
            bottom_right = (x + width, y + width)
            
            obs = self.world.observe(top_left, bottom_right)
            if return_dict:
                observations[tls.id] = obs
            else:
                observations.append(obs)

        if return_dict:
            return observations
        else:
            return np.array(observations)


    def update(self) -> None:
        self.world.update()
        self.tls_hub.update()


    # <|==============================================================================|> #
    # <|==============================================================================|> #
    # <|==============================================================================|> #


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