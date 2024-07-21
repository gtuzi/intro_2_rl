[Sutton & Barto RL Book]: http://incompleteideas.net/book/RLbook2020.pdf
[Dyna Maze Game]: https://github.com/konantian/Dyna-Maze-Game/tree/master

# Planning Methods
A unified view of reinforcement learning methods that require a model of the environment.
A _model_ of the environment we mean anything that an agent can use to predict how the environment will respond to its actions.
_Planning_ refers to any computational process that takes a model as input and produces or improves a policy for interacting with the modeled environment.
State-space _planning_ is the search through the state space for an optimal policy or an optimal path to a goal.

### State-space _planning_
* All state-space planning methods involve computing value functions as a key intermediate step toward improving the policy 
* They compute value functions by updates or backup operations applied to simulated experience.

## Table of Contents
- [Introduction](#introduction)
- [Implemented Algorithms](#implemented-algorithms)
- [Execution](#execution)
- [Environments](#environments)

## Introduction
This section contains implemented methods from Chapter 8, in [Sutton & Barto RL Book].
These are methods in which the agent uses _state-space planning_ to improve their policies.
The various state-space planning methods vary in the kinds of updates they do.

## Implemented Algorithms
- [x] Tabular Dyna-Q/Dyna-Q+ (Section 8.2): `agents/TabularDynaQAgent`

### Dyna Agents
Conceptually, planning, acting, model-learning, and direct RL occur 
simultaneously and in parallel in Dyna agents.

### Dyna-Q

<img src="images/dyna_agent.png" alt="Grid" width="350"/>

_Online planning_ agent which uses a _sample_ model (a sample model produces a possible transition). 
Under the assumption of a deterministic environment, this model _learns_ 
(stores) the next-state & reward for each state-action pair.
During the _planning_ phase, the agent simulates the experience from the model.
The simulation consists in the random sampling of the already visited (state, actions)
and generating the (next-state, reward) simulated experiences, for a number 
of planning steps. The agent uses these simulated experiences to improve its policy, using these. 
The Direct-RL path of Dyna-Q is Q-Learning (SarsaMax) is used to perform the policy improvement.

##### Dyna-Q+
When the environment is (slightly?) stochastic, our model will most likely be wrong.
In order to promote updating of the model, Dyna-Q+ uses a model-exploration heuristic, 
where during planning update, the simulated  reward is increased by visitation
staleness of a state by a _bonus factor_ of `𝜅*sqrt(𝜏)` where `𝜏` is the 
time since last visit. Note that this bonus  factor will affect the 
state-action values (Q(s, a)) - which is how this modification
in planning, promotes adaptation to a changing environment.

To improve model exploration, the algorithm allows for actions that have 
not been taken before for an already visited state. Refer to section 8.3 
(page 168) footnote in [Sutton & Barto RL Book].


## Execution
Run code in `main.py`. Each algorithm has its own `experiments` task.

### Running DynaMaze
When running Dyna-Q/Q+, the DynaMaze environment can be configured. 
A grid window pops up with the starting (S) and goal (G) state.
The user can configure the blocks by clicking/unclicking on the grid. 
Press "Spacebar" to confirm obstacle configuration. 
A `maze_config.pkl` is generated saving the environment config.
When running the script again, the grid loads from this config file. 
The user can modify the grid, or simply confirm the loaded environment by pressing the spacebar.

As the simulation runs, state (grid cell) values _V(s) = E<sub>a</sub>[Q(s, a)]_ are displayed as green colors.

<img src="images/grid.png" alt="Grid" width="350"/>

## Environments
- Extension of [Dyna Maze Game], which implements the environment in _Example 8.1_: Dyna Maze in the book.
