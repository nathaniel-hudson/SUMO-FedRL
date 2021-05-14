# data = defaultdict(list)

# def add_record(action, step, policy, reward) -> None: 
#     for i, a_i in enumerate(action):
#         data["tls"].append(i)
#         data["action"].append(a_i)
#         data["step"].append(step)
#         data["policy"].append(policy)
#         data["reward"].append(reward)

def get_lane_features(lane) -> Tuple[int, float, float, float, float, int]:
    # NOTE: It is interesting/important to remember/note that all of the returned values
    #       are singular integers or floats
    num_vehs, avg_speed, num_occupacy, wait_time, travel_time, num_halt = \
        traci.lane.getLastStepVehicleNumber(lane), \
        traci.lane.getLastStepMeanSpeed(lane), \
        traci.lane.getLastStepOccupancy(lane), \
        traci.lane.getWaitingTime(lane), \
        traci.lane.getTraveltime(lane), \
        traci.lane.getLastStepHaltingNumber(lane)
    return num_vehs, avg_speed, num_occupacy, wait_time, travel_time, num_halt

def store_lane_features(storage, step):
    for tls_id in traci.trafficlight.getIDList():
        for lane in traci.trafficlight.getControlledLanes(tls_id):
            num_vehs, avg_speed, num_occupacy, wait_time, travel_time, num_halt = \
                get_lane_features(lane)
            
            storage["val"].append(num_vehs);     storage["kind"].append("num_vehs")
            storage["val"].append(avg_speed);    storage["kind"].append("avg_speed")
            storage["val"].append(num_occupacy); storage["kind"].append("num_occupancy")
            storage["val"].append(wait_time);    storage["kind"].append("wait_time")
            storage["val"].append(travel_time);  storage["kind"].append("travel_time")
            storage["val"].append(num_halt);     storage["kind"].append("num_halt")

            for _ in range(6):
                storage["step"].append(step)
                storage["tls"].append(tls_id)
                storage["lane"].append(lane)





# data = defaultdict(list)

def add_record(action, step, policy, reward) -> None: 
    for i, a_i in enumerate(action):
        data["tls"].append(i)
        data["action"].append(a_i)
        data["step"].append(step)
        data["policy"].append(policy)
        data["reward"].append(reward)

def get_lane_features(lane) -> Tuple[int, float, float, float, float, int]:
    # NOTE: It is interesting/important to remember/note that all of the returned values
    #       are singular integers or floats
    num_vehs, avg_speed, num_occupacy, wait_time, travel_time, num_halt = \
        traci.lane.getLastStepVehicleNumber(lane), \
        traci.lane.getLastStepMeanSpeed(lane), \
        traci.lane.getLastStepOccupancy(lane), \
        traci.lane.getWaitingTime(lane), \
        traci.lane.getTraveltime(lane), \
        traci.lane.getLastStepHaltingNumber(lane)
    return num_vehs, avg_speed, num_occupacy, wait_time, travel_time, num_halt

def store_lane_features(storage, step):
    for tls_id in traci.trafficlight.getIDList():
        for lane in traci.trafficlight.getControlledLanes(tls_id):
            num_vehs, avg_speed, num_occupacy, wait_time, travel_time, num_halt = \
                get_lane_features(lane)
            
            storage["val"].append(num_vehs);     storage["kind"].append("num_vehs")
            storage["val"].append(avg_speed);    storage["kind"].append("avg_speed")
            storage["val"].append(num_occupacy); storage["kind"].append("num_occupancy")
            storage["val"].append(wait_time);    storage["kind"].append("wait_time")
            storage["val"].append(travel_time);  storage["kind"].append("travel_time")
            storage["val"].append(num_halt);     storage["kind"].append("num_halt")

            for _ in range(6):
                storage["step"].append(step)
                storage["tls"].append(tls_id)
                storage["lane"].append(lane)