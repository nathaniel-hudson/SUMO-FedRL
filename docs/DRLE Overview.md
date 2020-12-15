# DRLE: Decentralized Reinforcement Learning at the Edge for Traffic Light Control
> Authors: *Pegnyuan Zhou, Xianfu Chen, Zhi Liu, Tristan Braud, Pan Hui, Jussi Kangasharju*

In this recent work, the authors approach the problem of traffic light control while considering (decentralized) reinforcement learning being performed at the edge. This is a *very* relevant work for our research. This document aims to succinctly describe some of the assumptions incorporated in this work to establish what sets our work apart.

### State Space, Action Space, and Rewards
These are crucial components of any reinforcement learning algorithm. However, that said, it does appear that there representations for these components are a bit simple and homogeneous in nature. The work is still robust and novel, but there is plenty of room for expansion.

> **State Space:** They consider $s_k$ as the state at the $k^{th}$ cycle, including the number of halting vehicles $H^{k} = \{H_m^k | m=1, \cdots, M\}$, the average speed-lag of the vehicles $\Delta V^k=\{\Delta V_m^k | m=1, \cdots, M\}$, and the traffic light states $\theta^k=\{\theta_c^k|c=1, \cdots, C\}$, where $M$ is the total number of the lanes in the area.

A limitation of the considered state space (that is not obvious from this brief quote) is that the authors consider the routes vehicles take to be completely horizontal/verticle (i.e., no left/right turns). We can incorporate various vehicle-wise metrics to feature into our state space in a similar fashion: https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html.

> **Action Space:** They consider $a^k$ as the action operated by agent $k$ upon observing state $s^k$. More specifically, $a^k=\{a_c^k|c=1, \cdots, C\}$, where $a_c^k\in\{0,1\}$ represents the decision of switching the traffic light.

A limitation of their action space is that it does not allow alternate decisions being made --- only changes to a predefined sequence of actions.

> **Reward Function:** Authors consider the reward function $R^k$ defined as the additive inverse of the average speed-lag and number of halting vehicles, namely, $R^k = \sum_{m=1}^{M} (-w_1 \cdot H_m^k - w_2 \cdot \Delta V_m^k)$. 

