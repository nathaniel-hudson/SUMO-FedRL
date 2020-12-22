MIN_DELAY = 2 # 5

## Indices for the SUMO features for the observation spaces.
N_FEATURES = 7
NUM_VEHICLES, AVG_SPEED, NUM_OCCUPANCY, WAIT_TIME, TRAVEL_TIME, NUM_HALT, CURR_STATE = \
    range(N_FEATURES)