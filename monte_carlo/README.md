[Sutton & Barto RL Book]: http://incompleteideas.net/book/RLbook2020.pdf

# Monte Carlo Methods
Monte Carlo methods require only experience—sample sequences of states, actions, and rewards from actual or simulated interaction with an environment. Here we learn value functions from _sample_ returns. The estimates for each state are independent. The estimate for one state does not build upon the estimate of any other state, as is the case in DP. In other words, Monte Carlo methods do not bootstrap.


## Table of Contents
- [Introduction](#introduction)
- [Implemented Algorithms](#implemented-algorithms)
- [Execution](#execution)
- [Environments](#environments)


## Introduction
This section contains methods from Chapter 5, in [Sutton & Barto RL Book].

## Implemented Algorithms
- [x] On-policy first-visit MC control (5.4): `agents/MCOnPolicyFirstVisitGLIE`
- [x] Off-Policy MC control for estimating optimal π (5.7): `agents/MCOffPolicy`

## Execution
Run code in `main.py`. Each algorithm has its own `experiments` task.

## Environments
- [Gymnasium] - `FrozenLake-v1`