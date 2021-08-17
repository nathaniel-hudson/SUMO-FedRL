from numpy import float32

MIN_DELAY = 4
MAX_DELAY = 120
DEFUALT_RANKED = True

# This is is the observed vehicle length that is true for all vehicle types
# since we are not varying that. This will be used to provide a sense of
# how many vehicles are needed to accrue congestion in a road network based
# on its lane capacity.
VEHICLE_LENGTH = 5.0 

# N_RANKED_FEATURES = 7
# N_UNRANKED_FEATURES = N_RANKED_FEATURES - 2
# LANE_OCCUPANCY, HALTED_LANE_OCCUPANCY, SPEED_RATIO, PHASE_STATE_MODE, PHASE_STATE_STD, \
#     LOCAL_RANK, GLOBAL_RANK = range(N_RANKED_FEATURES)

N_RANKED_FEATURES = 14
N_UNRANKED_FEATURES = N_RANKED_FEATURES - 4

LANE_OCCUPANCY, HALTED_LANE_OCCUPANCY, SPEED_RATIO, \
PHASE_STATE_r, PHASE_STATE_y, PHASE_STATE_g, PHASE_STATE_G, PHASE_STATE_u, \
PHASE_STATE_o, PHASE_STATE_O, \
LOCAL_RANK, GLOBAL_RANK, LOCAL_HALT_RANK, GLOBAL_HALT_RANK \
    = range(N_RANKED_FEATURES)

HALTING_SPEED = 0.1

SPACE_DTYPE = float32

## ................................................... ##
##            Traffic light state constants            ##
## ................................................... ##
SORT_DEFAULT = True
RANK_DEFAULT = False
NEXT_STATES = {  # NOTE: Do we want to consider additional state transitions (see below)?
    "G": set(["G", "g", "y"]),
    "g": set(["G", "g", "y"]),
    "y": set(["y", "r"]),
    "r": set(["G", "g", "r"])
}

'''
TRAFFIC LIGHT STATE CONSTANTS.
More information can be found here:
https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html
'''
MAX_MPH_SPEED = 70
NUM_TLS_STATES = 8

STATE_r, STATE_y, STATE_g, STATE_G, STATE_s, STATE_u, STATE_o, STATE_O = \
    range(NUM_TLS_STATES)
STATES = [STATE_r, STATE_y, STATE_g, STATE_G,
          STATE_s, STATE_u, STATE_o, STATE_O]

STATE_r_STR, STATE_y_STR, STATE_g_STR, STATE_G_STR, STATE_s_STR, \
    STATE_u_STR, STATE_o_STR, STATE_O_STR = iter("rygGsuoO")
STATE_STRS = [STATE_r_STR, STATE_y_STR, STATE_g_STR, STATE_G_STR, STATE_s_STR,
              STATE_u_STR, STATE_o_STR, STATE_O_STR]

STATE_INT_TO_STR = {int_id: str_id
                    for (int_id, str_id) in zip(STATES, STATE_STRS)}
STATE_STR_TO_INT = {str_id: int_id
                    for (int_id, str_id) in zip(STATES, STATE_STRS)}
