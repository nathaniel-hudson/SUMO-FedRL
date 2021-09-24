from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from seal.trainer.communication import *
from seal.trainer.communication.base_callback import BaseCommCallback

class MultiPolicyCommCallback(BaseCommCallback):
    '''
    TRAINER:
        * edge2tls_policy += 0
        * tls2edge_policy += 0
    ENVIRONMENT:
        * edge2tls_action += 0
        * edge2tls_rank   += 1 (if ranked)
        * tls2edge_obs    += 0
        * veh2tls         += 1 (per vehicle)
    '''
    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs) -> None:
        # For some reason, the results of this function return a set of tuples of
        # identical keys... Not sure why, but that's why we only consider the 0th
        # elements of tuples.
        agent_ids = set([tuple[0] for tuple in episode.agent_rewards.keys()])
        for idx in agent_ids:
            info_dict = episode.last_info_for(idx)
            self.comm_cost[EDGE2TLS_POLICY, idx] += 0
            self.comm_cost[TLS2EDGE_POLICY, idx] += 0
            self.comm_cost[EDGE2TLS_ACTION, idx] += 0
            self.comm_cost[EDGE2TLS_RANK, idx] += int(info_dict["is_ranked"])
            self.comm_cost[TLS2EDGE_OBS, idx] += 0
            self.comm_cost[VEH2TLS_COMM, idx] += info_dict["veh2tls_comms"]