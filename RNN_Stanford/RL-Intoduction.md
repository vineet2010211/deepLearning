## Deep Learning: Reinforcement Learning Introduction

1. A typical rl problems usually consists of three characterstics: 
      * Some sort of closed loop system
      * Not having direct instructions as to what actions to take 
      * Consequences of actions, including reward signals, play out over extended period of time.
      
2. Exploration vs Exploitation Dellima.

  The agent has to exploit what it already knows in order to obtain reward, but it has to also explore in order to make better action selection in the future. Learning by doing rather than notes/wrote learning.


#### Elements of Reinforcement Learning 

* __Agent__
* __Environment__ 
* __A Policy__
     * *A policy defines the learning agent’s way of behaving at a given time. Roughly speaking, a policy is a mapping from perceived states of the environment to actions to be taken when in those states. It corresponds to what in psychology would be called a set of stimulus–response rules or associations (provided that stimuli include those that can come from within the animal). In some cases the policy may be a simple function or lookup table, whereas in others it may involve extensive computation such as a search process. The policy is the core of a reinforcement learning agent in the sense that it alone is sufficient to determine behavior. In general, policies may be stochastic.*
* __A reward function__ 
     * Short term desirability of moving from a particular to state to the next.
* __A value function__
     * Long term desirability of taking certain actions in the environment. Value and reward function may contradict at certain time step in the process.
* [optional] __A model of the environment__
