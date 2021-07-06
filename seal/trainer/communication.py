from collections import defaultdict

class CommunicationModel:

    def __init__(self):
        self.edge_to_tls_decision_comms = defaultdict(int)
        self.edge_to_tls_rank_comms = defaultdict(int)
        self.tls_to_edge_observation_comms = defaultdict(int)
        self.tls_to_edge_policy_comms = defaultdict(int)