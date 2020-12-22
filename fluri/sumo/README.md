# SUMO-Gym Design Description

## Action Space.
In this work, we consider a `Box` action space of dimensionality `(n_trafficlights,)`,
with a high value of the number of actions and a low of $0$. To put it simply, an action 
$a^t$, at time-step $t$, in the action space is a vector wherein each element $a_i^t$ is
the action (or traffic light phase) an agent will take in the $t^{th}$ time-step.

There is a need to discuss the concerns of the actions for traffic lights in SUMO. First, 
traffic light changes are sequentially related to one another. For instance, a traffic 
light *cannot* transition from **green** to **red** without first becoming **yellow**. 
Additionally, this intermittent transition must provide adequate time before
transitioning to **red**. As such, we have to address the action space for this
environment with care. 

As such, we consider the following for each possible available action:
* First, an action is composed of two central sub-task: a ***main*** task and a 
  ***transition*** task. An action's main task is (as its name suggests) the bulk of the action. The transition task is the portion wherein the activated lights that need to transition (predominantly `Green` to `red`) do so.
* Second, the main task takes place for however long the agent decides and the agent is
  free to choose to change the action to another action if desired; the transition task
  takes place for a *fixed* time period so as to provide adequate time for cars to slow
  down — during which, the agent ***cannot*** change the action.

In order to implement this framework, we consider the following. Given an action 
$a^t=[0, 3, 1, 2, \cdots, 0]$, we also consider a mask vector at time-step $t$, 
$m^t=[-3, -2, 0, 0, \cdots, -2]$. A value $m_i^t \in m^t$ represents the number of needed
steps in the simulation in order for the agent to change the action (i.e., a negative
value means the traffic light is in its transition task).

## Miscellaneous Notes.
* Interestingly, `num_workers` in the Trainer config for an `RlLib` algorithm must be `num_workers=0`, otherwise an issue is caused by `OptParser` from the route generation script.