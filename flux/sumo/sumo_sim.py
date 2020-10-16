import traci
import xml.etree.ElementTree as ET

from typing import List

class SUMOSimulation():

    def __init__(self):
        self.network = None
        self.routes = None
        self.trip = None
        self.detector = None
        self.config = None
        self.traffic_lights = None

    def get_traffic_lights(self):
        self.traffic_light_ids = list(traci.trafficlight.getIDList())

    def get_traffic_programs(
        self, traffic_programs: str=None, key: str="programID"
    ) -> List[str]:
        """Reads the XML file for the traffic programs and returns the IDs of traffic 
           programs.
        """
        tree = ET.parse(traffic_programs)
        return [tlLogic.attrib[key] for tlLogic in tree.findall("tlLogic")]

    def generate_routes(self):
        pass

    def start(self, config=None):
        traci.start(config)

    def step(self):
        traci.simulationStep()

    def close(self):
        traci.close()


if __name__ == "__main__":
    from os.path import join

    sim = SUMOSimulation()
    filename = join("configs", "example", "tls_program.add.xml")
    logics = sim.get_traffic_programs(filename)
    print(logics)