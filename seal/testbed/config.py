from numpy import float32

MIN_DELAY = 4
MAX_DELAY = 120
DEFUALT_RANKED = True

N_RANKED_FEATURES = 14
N_UNRANKED_FEATURES = N_RANKED_FEATURES - 4
LANE_OCCUPANCY, HALTED_LANE_OCCUPANCY, SPEED_RATIO, \
    PHASE_STATE_r, PHASE_STATE_y, PHASE_STATE_g, PHASE_STATE_G, \
    PHASE_STATE_u, PHASE_STATE_o, PHASE_STATE_O, \
    LOCAL_RANK, GLOBAL_RANK, LOCAL_HALT_RANK, GLOBAL_HALT_RANK \
    = range(N_RANKED_FEATURES)

#added for testbed
DEFAULT_LANE_LENGTH = 5
DEFAULT_VEH_LENGTH = 2
DEFAULT_MAX_SPEED = 0.3
TEST_HALTING_SPEED = 0.05

HALTING_SPEED = 0.05

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

