import numpy as np
import time
import traci
import warnings
import xml.etree.ElementTree as ET

from typing import Any, Dict, List, Tuple, Union

from seal.sumo.kernel.trafficlight.hub import TrafficLightHub

SORT_DEFAULT = True
VERBOSE_DEFAULT = 0


class SumoKernel():

    def __init__(self, config: Dict[str, Any]=None, ranked: bool=True):
        """Initialize a wrapper for a SUMO simulation by passing a `config` dict object
           that stores the command-line arguments needed for a SUMO simulation.

        Parameters
        ----------
        config : Dict[str, Any], optional
            The command-line arguments required for running a SUMO simulation, by default
            None.
        """

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
        self.tls_hub = TrafficLightHub(
            self.config["net-file"], 
            ranked=config.get("ranked", True)
        )


    def get_command_args(
        self,
        verbose=VERBOSE_DEFAULT,
        no_step_log: bool=True
    ) -> List[str]:
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

        Parameters
        ----------
        ignore_tls : bool, optional
            Whether to update the trafficlights (True) or not (False), by default False.
        """
        if not ignore_tls:
            self.tls_hub.update()


    # <|==============================================================================|> #
    # <|==============================================================================|> #
    # <|==============================================================================|> #


    def is_loaded(self) -> bool:
        """Checks whether a simulation is loaded or not.

        Returns
        -------
        bool
            Returns True if a connection is loaded, False otherwise.
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

        Returns
        -------
        bool
            Returns True if the simulation is done, False otherwise.
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
