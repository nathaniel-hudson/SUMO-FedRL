import numpy as np

from gym import spaces
from typing import Any, List


def TrafficLightSpace(lane_ids: List[Any], ranked: bool=False) -> spaces.Space:
    congestion_space = None
    num_halt_space = None
    avg_speed_space = None
    curr_state_space = None

    def get_lane_space():
        space_list = [
            spaces.Box(...),  # Congestion.
            spaces.Box(...),  # Number of halted vehicles.
            spaces.Box(...),  # Average speed per lane.
            spaces.Box(...),  # Current traffic light phase.
        ]
        if ranked:
            space_list.extend([
                spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64),
                spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64)
            ])
        return tuple(space_list)

    spaces.Dict({
        lane_id: spaces.Tuple(get_lane_space())
        for lane_id in lane_ids
    })
