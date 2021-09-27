

from seal.trainer.communication import COMM_TYPES, VEH2TLS_COMM


class DataParser:

    def __init__(self, result_data) -> None:
        self.result_data = result_data

    def episode_reward(self, iteration: int=-1) -> float:
        reward = self.result_data["hist_stats"]["episode_reward"][iteration]
        return reward

    def policy_reward(self, policy_id: str, iteration: int=-1) -> float:
        """Gets the reward that the passed in policy received during training iteration
           (`iteration`). By default, the returned policy reward will be the most recent.

        Args:
            policy_id (str): ID of the policy.
            iteration (int, optional): Iteration index during the episode. Defaults to -1.

        Returns:
            float: Policy reward during the specified iteration of the results.
        """
        policy_key = f"policy_{policy_id}_reward"
        # print(f"\n\n\nresults.keys():\n{self.result_data.keys()}\n\n\n")

        print(self.result_data["hist_stats"])

        reward = self.result_data["hist_stats"][policy_key][iteration]
        return reward

    def episode_comm_cost(self, comm_type: str=None, iteration: int=-1) -> int:
        """Gets the total communication cost for either a specified communication type
           (`comm_type`) or, by default, across all communication types (i.e., 
           `comm_type=None`). This is done for the specified `iteration`, which by 
           default is to consider the most recent comm_cost.

        Args:
            comm_type (str, optional): Communication type to return. Defaults to None.
            iteration (int, optional): Which iteration to consider. Defaults to -1.

        Raises:
            ValueError: This might occur for some reason unknown at this point.

        Returns:
            int: Communication costs based on specified arguments.
        """
        if comm_type is None:
            return sum(self.episode_comm_cost(c, iteration) 
                       for c in COMM_TYPES)
        else:
            assert comm_type in COMM_TYPES
            query = f"comm={comm_type}"
            for key in self.result_data["hist_stats"]:
                if query in key:
                    return self.result_data["hist_stats"][key][iteration]
            raise ValueError("Somehow, an error occurred when trying to get "
                             "communication cost.")

    def policy_comm_cost(self, policy_id: str, comm_type: str, iteration: int=-1) -> int:
        policy_key = f"policy_{policy_id}_comm={comm_type}"
        comm_cost = self.result_data["hist_stats"][policy_key][iteration]
        return comm_cost

    def num_vehicles(self, policy_id: str, iteration=-1) -> int:
        
        print(self.result_data["hist_stats"])

        policy_key = f"policy_{policy_id}_comm={VEH2TLS_COMM}"
        n_vehicles = self.result_data["hist_stats"][policy_key][iteration]
        return n_vehicles

    ## -------------------------------------------------------------------------------- ##

    @property
    def episode_reward_max(self) -> float:
        return self.result_data.episode_reward_max