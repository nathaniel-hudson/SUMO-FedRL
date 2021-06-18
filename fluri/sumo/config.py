from numpy import float32

MIN_DELAY = 4
DEFUALT_RANKED = True
N_RANKED_FEATURES = 7
N_UNRANKED_FEATURES = N_RANKED_FEATURES - 2

CONGESTION, HALT_CONGESTION, AVG_SPEED, CURR_STATE_MODE, CURR_STATE_STD, \
    LOCAL_RANK, GLOBAL_RANK = range(N_RANKED_FEATURES)

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
