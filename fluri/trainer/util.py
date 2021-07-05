from numpy import array
from os.path import join
from typing import Any, Dict, NewType

GLOBAL_POLICY_VAR = "TEST-EVAL_POLICY"

DEFAULT_GUI = False
DEFAULT_NET_FILE = join("configs", "two_inter", "two_inter.net.xml")
DEFAULT_RAND_ROUTES_ON_RESET = True
DEFAULT_RANKED = True

Weights = NewType("Weights", Dict[Any, array])
Policy = NewType("Policy", Dict[Any, array])

def get_env_config(**kwargs):
    config = {
        "gui": kwargs.get("gui", DEFAULT_GUI),
        "net-file": kwargs.get("net-file", DEFAULT_NET_FILE),
        "rand_routes_on_reset": kwargs.get("rand_routes_on_reset", DEFAULT_RAND_ROUTES_ON_RESET),
        "ranked": kwargs.get("ranked", DEFAULT_RANKED),
    }
    return config

def eval_policy_mapping_fn(key) -> str:
    return GLOBAL_POLICY_VAR