# These are module-wide constants that define the communication types. These strings
# will be used as keys in various data and processes during training.
EDGE2TLS_POLICY = "edge-to-tls-policy-comms"
TLS2EDGE_POLICY = "tls-to-edge-policy-comms"
EDGE2TLS_ACTION = "edge-to-tls-action-comms"
EDGE2TLS_RANK = "edge-to-tls-rank-comms"
TLS2EDGE_OBS = "tls-to-edge-obs-comms"
VEH2TLS_COMM = "veh-to-tls-info-comms"
COMM_TYPES = set([EDGE2TLS_POLICY, TLS2EDGE_POLICY, EDGE2TLS_ACTION, EDGE2TLS_RANK,
                  TLS2EDGE_OBS, VEH2TLS_COMM])