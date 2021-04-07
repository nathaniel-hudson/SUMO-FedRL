import numpy as np

from typing import Any, Dict, List, NewType

Weights = NewType("Weights", Dict[Any, np.array])
Policy = NewType("Policy", Dict[Any, np.array])

def federated_avg(policies: List[Policy], C: float=1.0) -> Weights:
    weights = np.array([policy.get_weights() for policy in policies])
    policy_keys = policies[0].get_weights().keys()
    # coeffs = np.array()
    # coeffs = coeffs / sum(coeffs)
    new_weights = {}
    for key in policy_keys:
        weights = np.array([policy.get_weights()[key] for policy in policies])
        new_weights[key] = sum(1/len(policies) * weights[k] 
                               for k in range(len(policies)))

    return new_weights

