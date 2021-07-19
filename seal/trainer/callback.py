"""
FedRL:
    >> Trainer
        * edge2tls_policy += 1 (each fed round)
        * tls2edge_policy += 1 (each fed round)
    >> Env
        * edge2tls_action += 0
        * edge2tls_rank += 1 (if ranked)
        * tls2edge_obs += 0
        * veh2tls += 1 (per vehicle) [$$$]

SARL:
    >> Trainer
        * edge2tls_policy += 0
        * tls2edge_policy += 0
    >> Env
        * edge2tls_action += 1
        * edge2tls_rank += 1 (if ranked)
        * tls2edge_obs += 1
        * veh2tls += 1 (per vehicle) [$$$]

SARL:
    >> Trainer
        * edge2tls_policy += 0
        * tls2edge_policy += 0
    >> Env
        * edge2tls_action += 0
        * edge2tls_rank += 1 (if ranked)
        * tls2edge_obs += 0
        * veh2tls += 1 (per vehicle) [$$$]

[$$$] -- is the most difficult feature.

RESOURCES:
    + https://docs.ray.io/en/master/_modules/ray/rllib/evaluation/episode.html
    + https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
"""
