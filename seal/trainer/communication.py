import pandas as pd

from collections import defaultdict


class CommunicationModel:

    __edge_to_tls_action_comms: defaultdict
    __edge_to_tls_rank_comms: defaultdict
    __edge_to_tls_policy_comms: defaultdict
    __tls_to_edge_obs_comms: defaultdict
    __tls_to_edge_policy_comms: defaultdict
    __veh_to_tls_info_comms: defaultdict

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tls_ids = set()
        self.__edge_to_tls_action_comms = defaultdict(int)
        self.__edge_to_tls_rank_comms = defaultdict(int)
        self.__edge_to_tls_policy_comms = defaultdict(int)
        self.__tls_to_edge_obs_comms = defaultdict(int)
        self.__tls_to_edge_policy_comms = defaultdict(int)
        self.__veh_to_tls_info_comms = defaultdict(int)

    def incr_edge_to_tls_action(self, tls_id) -> None:
        self.__edge_to_tls_action_comms[tls_id] += 1
        self.tls_ids.add(tls_id)

    def incr_edge_to_tls_rank(self, tls_id) -> None:
        self.__edge_to_tls_rank_comms[tls_id] += 1
        self.tls_ids.add(tls_id)

    def incr_edge_to_tls_policy(self, tls_id) -> None:
        self.__edge_to_tls_policy_comms[tls_id] += 1
        self.tls_ids.add(tls_id)

    def incr_tls_to_edge_obs(self, tls_id) -> None:
        self.__tls_to_edge_obs_comms[tls_id] += 1
        self.tls_ids.add(tls_id)

    def incr_tls_to_edge_policy(self, tls_id) -> None:
        self.__tls_to_edge_policy_comms[tls_id] += 1
        self.tls_ids.add(tls_id)

    def incr_veh_to_tls_info(self, tls_id) -> None:
        self.__veh_to_tls_info_comms[tls_id] += 1
        self.tls_ids.add(tls_id)

    def to_csv(self, path) -> None:
        aux_data = defaultdict(list)
        for tls in self.tls_ids:
            aux_data["tls_id"].append(tls)
            aux_data["edge_to_tls_action_comms"].append(
                self.__edge_to_tls_action_comms[tls])
            aux_data["edge_to_tls_rank_comms"].append(
                self.__edge_to_tls_rank_comms[tls])
            aux_data["edge_to_tls_policy_comms"].append(
                self.__edge_to_tls_policy_comms[tls])
            aux_data["tls_to_edge_obs_comms"].append(
                self.__tls_to_edge_obs_comms[tls])
            aux_data["tls_to_edge_policy_comms"].append(
                self.__tls_to_edge_policy_comms[tls])
            aux_data["veh_to_tls_info_comms"].append(
                self.__veh_to_tls_info_comms[tls])
        aux_df = pd.DataFrame.from_dict(aux_data)
        aux_df.to_csv(path)
