import numpy as np
import random
from gym import spaces
from scipy import stats
from typing import List
from seal.sumo.config import *
from seal.sumo.kernel.const import *
from seal.sumo.kernel.trafficlight.space import trafficlight_space

class TrafficLight():
    def __init__(self, tls_id, config, ranked, pynode):
        self.index = tls_id
        self.id = tls_id
        self.config = config
        self.program = self.get_program()
        self.num_phases = len(self.program)
        self.state = random.randrange(self.num_phases)
        self.phase = self.program[self.state]
        self.ranked = ranked
        self.rosnode = pynode

    def get_program(self):
        program_order = ["GGrr", "yyrr", "rrGG", "rryy"]
        return program_order

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=1, shape=(1,), dtype=int)

    @property
    def observation_space(self) -> spaces.Tuple:
        return trafficlight_space(self.ranked)

    def update(self):
        """Update the current state by interacting with `traci`."""
        # node = self.config["ros-node"]
        node = self.rosnode
        self.state = node.get_status(self.index)
        self.phase = self.program(self.state)

    def next_phase(self):
        # node = self.config["ros-node"]
        node = self.rosnode
        next_state = (self.state+1) % self.num_phases
        next_phase = self.program[next_state]
        self.state = next_state
        self.phase = next_phase
        node.set_status(self.id, self.state, self.phase)


    def get_observation(self):

        # Initialize the observation list (obs).
        n_features = N_RANKED_FEATURES if self.ranked else N_UNRANKED_FEATURES
        obs = [0 for _ in range(n_features)]


        #hard-coded intersection to lane served. intersection ID to lane ID matching.
        lights_dict = {}
        lights_dict['0'] = [0,6,5,3]
        lights_dict['1'] = [1,2,4,7]
        # print("current phase: ", self.phase)

        #get data from the subscriber
        node = self.rosnode
        traffic_data = node.get_observations()#self.index
        traffic_info = []
        if traffic_data:
            for each_bot in traffic_data:
                if each_bot.lane in lights_dict[self.index]:
                    print(each_bot.lane, "appended\n")
                    traffic_info.append([each_bot.x, each_bot.y, each_bot.v])
            # traffic_data = 0.8
        else:
            traffic_data = 0.0
        
        # Extract the lane-specific features.

        max_lane_speeds = vehicle_speeds = 0
        total_lane_length = vehicle_lengths = halted_vehicle_lengths = 0

        for l in lights_dict[self.index]:
            total_lane_length += DEFAULT_LANE_LENGTH
            max_speed = DEFAULT_MAX_SPEED

            for v in  traffic_info:
                speed = v[2]
                vehicle_speeds += speed
                max_lane_speeds += max_speed
                vehicle_lengths += DEFAULT_VEH_LENGTH

                if speed < TEST_HALTING_SPEED:
                    halted_vehicle_lengths += DEFAULT_VEH_LENGTH

        obs[LANE_OCCUPANCY] = vehicle_lengths / total_lane_length
        obs[HALTED_LANE_OCCUPANCY] = halted_vehicle_lengths / total_lane_length
        try:
            # We need to "clip" average speed because drivers sometimes exceed the
            # speed limit.
            obs[SPEED_RATIO] = min(1.0, vehicle_speeds / max_lane_speeds)
        except ZeroDivisionError:
            # This happens when 0 vehicles are on lanes controlled by the traffic light.
            obs[SPEED_RATIO] = 0.0

        # Extract descriptive statistics features for the current traffic light state.
        curr_tls_state = self.phase
        for light in curr_tls_state:
            if light == STATE_r_STR:
                obs[PHASE_STATE_r] += 1/len(curr_tls_state)
            elif light == STATE_y_STR:
                obs[PHASE_STATE_y] += 1/len(curr_tls_state)
            elif light == STATE_g_STR:
                obs[PHASE_STATE_g] += 1/len(curr_tls_state)
            elif light == STATE_G_STR:
                obs[PHASE_STATE_G] += 1/len(curr_tls_state)
            # elif light == STATE_s_STR:
            #     obs[PHASE_STATE_s] += 1/len(curr_tls_state)
            elif light == STATE_u_STR:
                obs[PHASE_STATE_u] += 1/len(curr_tls_state)
            elif light == STATE_o_STR:
                obs[PHASE_STATE_o] += 1/len(curr_tls_state)
            elif light == STATE_O_STR:
                obs[PHASE_STATE_O] += 1/len(curr_tls_state)
        return np.array(obs)




    # def __get_lane_occupancy(self) -> float:
    #     pass

    # def __get_halted_lane_occupancy(self) -> float:
    #     pass

    # def __get_speed_ratio(self) -> float:
    #     pass

    # def __get_phase_mode(self) -> float:
    #     pass

    # def __get_phase_std(self) -> float:
    #     pass