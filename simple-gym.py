import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import traci

from collections import defaultdict
from fluri.sumo.single_agent_env import SingleSumoEnv
from fluri.sumo.kernel.kernel import SumoKernel
from fluri.sumo.utils.random_routes import generate_random_routes
from os.path import join
from typing import Tuple

sns.set_style("ticks")
data = defaultdict(list)

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

if __name__ == "__main__":
    n_episodes = 1
    env = SingleSumoEnv(config={
        "gui": False,
        "net-file": join("configs", "two_inter", "two_inter.net.xml"),
        "rand_route_args": {
            "n_vehicles": (100, 500),
            "end_time": 300
        }
    })

    tmp_data = defaultdict(list)
    tls, link, lane = "light1", 1, "e7_0" ## NOTE: Temporary.
    for ep in range(n_episodes):
        env.reset()
        print(f"\n\nphase:\n{env.kernel.tls_hub['light1'].possible_states}\n\n")
        tls_ids = env.kernel.tls_hub.get_traffic_light_ids()
        done, step, total_reward = False, 0, 0
        while not done:
            store_lane_features(tmp_data, step)
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            add_record(info["taken_action"], step, "Random", total_reward)
            step += 1

    # print(f"controlled lanes: {traci.trafficlight.getControlledLanes('light1')}")
    # print(f"controlled links: {traci.trafficlight.getControlledLinks('light1')}")
    env.close()

    # Do a simple lineplot of the actions taken over time.
    data = pd.DataFrame.from_dict(data)
    sns.lineplot(x="step", y="action", hue="policy", style="tls", data=data)
    plt.show()

    tmp_data = pd.DataFrame.from_dict(tmp_data)
    print(tmp_data.head(), len(tmp_data))
    sns.relplot(
        x="step", 
        y="val", 
        col="kind", 
        col_wrap=3,
        kind="line",
        hue="tls", 
        facet_kws={"sharex": False, "sharey": False},
        data=tmp_data
    )
    plt.show()
    