[Sutton & Barto RL Book]: http://incompleteideas.net/book/RLbook2020.pdf
[Dyna Maze Game]: https://github.com/konantian/Dyna-Maze-Game/tree/master


# Planning Methods
A unified view of reinforcement learning methods that require a model of the environment

## Table of Contents
- [Introduction](#introduction)
- [Implemented Algorithms](#implemented-algorithms)
- [Execution](#execution)
- [Environments](#environments)

## Introduction
This section contains methods from Chapter 8, in [Sutton & Barto RL Book].

## Implemented Algorithms
- [x] Tabular Dyna-Q (Section 8.2): `agents/TabularDynaQAgent`

## Execution
Run code in `main.py`. Each algorithm has its own `experiments` task.

### Running DynaMaze
When running Dyna-Q, the DynaMaze environment can be configured. 
A grid window pops up with the starting (S) and goal (G) state.
The user can configure the blocks by clicking/unclicking on the grid. 
Press "Spacebar" to confirm obstacle configuration. 
A `maze_config.pkl` is generated saving the environment config.
When running the script again, the grid loads from this config file. 
The user can modify the grid, or simply confirm the loaded environment by pressing the spacebar.

As the simulation runs, state (grid cell) values _V(s) = E<sub>a</sub>[Q(s, a)]_ are displayed as green colors.

![grid](images/grid.png)


## Environments
- Extension of [Dyna Maze Game], which implements the environment in _Example 8.1_: Dyna Maze in the book.
