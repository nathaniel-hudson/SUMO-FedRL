from typing import Dict

'''
episode_data = {
    'policy1': {'reward': ..., 'num_vehicles': ...},
    'policy2': {'reward': ..., 'num_vehicles': ...},
    ...
}
'''

def naive_weight_function(episode_data: Dict) -> Dict[str, float]:
    coeffs = {
        policy: 1 / len(episode_data)
        for policy in episode_data
    }
    return coeffs


def neg_reward_weight_function(episode_data: Dict) -> Dict[str, float]:
    total_reward = abs(sum(policy_data["reward"] 
                           for policy_data in episode_data.values()))
    unnormalized_coeffs = {
        policy: total_reward / (policy_data["reward"] - 1)
        for (policy, policy_data) in episode_data.items()
    }
    try:
        coeffs = {
            policy: unnormalized_coeffs[policy] / sum(unnormalized_coeffs.values())
            for policy in episode_data
        }
    except ZeroDivisionError:
        coeffs = naive_weight_function(episode_data)
    return coeffs


def pos_reward_weight_function(episode_data: Dict) -> Dict[str, float]:
    total_reward = sum(policy_data["reward"] 
                       for policy_data in episode_data.values())
    try:
        coeffs = {
            policy: policy_data["reward"] / total_reward
            for (policy, policy_data) in episode_data.items()
        }
    except ZeroDivisionError:
        coeffs = naive_weight_function(episode_data)
    return coeffs


def traffic_weight_function(episode_data: Dict) -> Dict[str, float]:
    total_vehicles = sum(policy_data["num_vehicles"] 
                         for policy_data in episode_data.values())
    try:
        coeffs = {
            policy: policy_data["num_vehicles"] / total_vehicles
            for (policy, policy_data) in episode_data.items()
        }
    except ZeroDivisionError:
        coeffs = naive_weight_function(episode_data)
    return coeffs