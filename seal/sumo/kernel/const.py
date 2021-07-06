SORT_DEFAULT = True
RANK_DEFAULT = False
NEXT_STATES = {
    "G": set(["G", "g", "y"]),
    "g": set(["G", "g", "y"]),
    "y": set(["y", "r"]),
    "r": set(["G", "g", "r"])
}

## Traffic light state constants.
NUM_TLS_STATES = 8
STATE_r, STATE_y, STATE_g, STATE_G, STATE_s, STATE_u, STATE_o, STATE_O = \
    range(NUM_TLS_STATES)