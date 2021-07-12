from os.path import join
from typing import Any, Dict

GUI: bool = False
NET_FILE: str = join("configs", "two_inter", "two_inter.net.xml")
RAND_ROUTES_ON_RESET: bool = True
RAND_ROUTES_CONFIG: Dict[str, Any] = {
    "n_vehicles": (1000, 10000),
    "generator": "uniform",
    "end_time": (None, None),
    "seed": 1234
}
RANKED: bool = True