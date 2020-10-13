import traci

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