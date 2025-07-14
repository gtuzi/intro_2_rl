[Sutton & Barto RL Book]: http://incompleteideas.net/book/RLbook2020.pdf


# On-policy Control with Approximation

## Table of Contents
- [Theoretical Background](#theoretical-background)
- [Implemented Algorithms](#Implemented-Algorithms)
- [Semi-Gradient Sarsa Algorithms](#semi-gradient-sarsa-algorithms)
- [Semi-Gradient Sarsa Algorithm Experiments](#semi-gradient-sarsa-algorithm-experiments)
- [n-Step Semi-Gradient Sarsa Algorithms](#n-step-semi-gradient-sarsa-algorithms)
- [n-Step Semi-Gradient Sarsa Experiments](#n-step-semi-gradient-sarsa-experiments)
- [Continuing Task Semi-Gradient Control: 1-step](#continuing-task-semi-gradient-control-1-step)
- [Continuing Task Semi-Gradient Experiments: 1-step](#continuing-task-semi-gradient-experiments-1-step)
- [Continuing Task Semi-Gradient Control: n-step](#continuing-task-semi-gradient-control-n-step)
- [Continuing Task Semi-Gradient Experiments: n-step](#continuing-task-semi-gradient-control-experiments-n-step)
- [Environment](#environment)


## Theoretical Background
See expanded discussion of the algorithms and concepts covered here: [summary.ipynb](summary.ipynb).

Topics covered in the notebook:
* __Linear Methods__
* __Prediction Objective__
  * Stochastic Gradient
  * Stochastic Semi-Gradient
* __Episodic Control__
  * One-Step Semi-Gradient
  * $n$-Step Semi-Gradient
* __Continuous Control: Average Reward Setting__
  * Differential Semi-Gradient
  * One-Step Differential Semi-Gradient
  * $n$-Step Differential Semi-Gradient


## Implemented Algorithms
- [x]  Semi-gradient Sarsa (Section: 10.1): `agents.py/SemiGradientSarsa`
- [x]  Semi-gradient Expected Sarsa (extension of Sarsa): `agents.py/SemiGradientExpectedSarsa`
- [x]  Semi-gradient SarsaMax (QLearning) (extension of Sarsa): `agents.py/SemiGradientQLearning`
- [x]  n-Step Semi-gradient Sarsa (Section: 10.2): `agents.py/nStepSemiGradientSarsa`
- [x]  n-Step Semi-gradient Expected Sarsa (extension of Sarsa): `agents.py/nStepSemiGradientExpectedSarsa`
- [x]  n-Step Semi-gradient SarsaMax / QLearning (extension of Sarsa): `agents.py/nStepSemiGradientQLearning`
- [x]  Differential Semi-Gradient Sarsa (Section: 10.3): `agents.py/DifferentialSemiGradientSarsa`
- [x]  Differential Semi-Gradient QLearning (Section: 10.3): `agents.py/DifferentialSemiGradientQLearning`
- [x]  Differential Semi-Gradient Expected Sarsa (extension of Sarsa): `agents.py/DifferentialSemiGradientExpectedSarsa`
- [x]  DifferentialSemiGradient_nStepSarsa (Section: 10.5): `agents.py/DifferentialSemiGradient_nStepSarsa`
- [x]  DifferentialSemiGradient_nStepExpectedSarsa (extension of Sarsa): `agents.py/DifferentialSemiGradient_nStepExpectedSarsa`
- [x]  DifferentialSemiGradient_nStepQLearning (extension of Sarsa): `agents.py/DifferentialSemiGradient_nStepQLearning`

## Semi-Gradient Sarsa Algorithms
For the one-step Sarsa, the target value $U_t = R_{t+1} + \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w})$. 
The one-step algorithm listed in the book is the following:
<img src="images/Semi_Gradient_Sarsa.png" alt="Grid" width="800"/>

Following the formulations for the tabular case (ref Chapter 6), 
the one-step Sarsa target value $U_t$ can be easily extended to Expected
Sarsa and Q-Learning. The rest of the boxed algorithm remains the same.

### Expected Sarsa 
$U_t = R_{t+1} + \gamma\hat{v}(S_{t+1}, \mathbf{w}) = R_{t+1} + \gamma{\sum_{a}{\pi(a|S_{t+1})\hat{q}(S_{t+1},a, \mathbf{w})}}$

### Q-Learning
$U_t = R_{t+1} + \gamma\max_a{\hat{q}(S_{t+1}, a, \mathbf{w})}$


### Semi-Gradient Sarsa Algorithm Experiments

###### Simulation Parameters
The following simulation parameters are used:
* 5 seeds
* Search parameter: $\varepsilon$ = `[0.01, 0.05, 0.1, 0.3]`. 
  * Results for one $\varepsilon$ shown in table. Others in `/results`
* Num episodes = $200$
* Evaluation frequency = $5$
* MaxSteps = $999$

###### Learning Parameters
* learning rate $\alpha$ was decayed from `start` to `end`
  * `start` = $\frac{1}{2NumTilings} = \frac{1}{16}$
  * `end` = $\frac{1}{10NumTilings} = \frac{1}{80}$

  
| Algorithm      | Parameters           | Train                                                                                                      | Evaluation                                                                                                | 
|----------------|----------------------|------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Sarsa          | $\varepsilon$ = 0.01 | <img src="images/results/BaseReward_SemiGradientSarsa_Train_eps_0.01.png" alt="Grid" width="400"/>         | <img src="images/results/BaseReward_SemiGradientSarsa_Eval_eps_0.01.png" alt="Grid" width="400"/>         |
| Expected-Sarsa | $\varepsilon$ = 0.01 | <img src="images/results/BaseReward_SemiGradientExpectedSarsa_Train_eps_0.01.png" alt="Grid" width="400"/> | <img src="images/results/BaseReward_SemiGradientExpectedSarsa_Eval_eps_0.01.png" alt="Grid" width="400"/> |
| Q-Learning     | $\varepsilon$ = 0.01 | <img src="images/results/BaseReward_SemiGradientQLearning_Train_eps_0.01.png" alt="Grid" width="400"/>     | <img src="images/results/BaseReward_SemiGradientQLearning_Eval_eps_0.01.png" alt="Grid" width="400"/>     |


## $n$-Step Semi-Gradient Sarsa Algorithms
An n-step version of episodic semi-gradient Sarsa by using an n-step return as the update target in the semi-gradient Sarsa update equation.
The n-step return immediately generalizes from its tabular form to a function approximation form.

$G_{t:t+n} \overset{\cdot}{=} R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n \hat{q}(S_{t+n}, A_{t+n}, \mathbf{w}_{t+n-1}), \ t + n < T$

The integrated algorithm is shown below:

<img src="images/nStep_Semi_Gradient_Sarsa.png" alt="Grid" width="800"/>

Likewise, following the diagrams in Figure 7.3, we can extend the n-Step 
tabular methods for computing $G_{t:t+n}$ as follows

### n-Step Expected Sarsa
$G_{t:t+n} \overset{\cdot}{=} R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n{\sum_{a}{\pi(a|S_{t+n})\hat{q}(S_{t+n}, a, \mathbf{w}_{t+n-1})}}, \ t + n < T$

### n-Step Sarsa Max/QLearning
$G_{t:t+n} \overset{\cdot}{=} R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n{\max_{a}\hat{q}(S_{t+n}, a, \mathbf{w}_{t+n-1})}, \ t + n < T$

## $n$-Step Semi-Gradient Sarsa Experiments

###### Simulation Parameters
The following simulation parameters are used:
* 5 seeds
* Num episodes = $200$
* Evaluation frequency = $5$
* MaxSteps = $999$

###### Learning Parameters
* learning rate $\alpha$ was decayed from `start` to `end`
  * `start` = $\frac{1}{2NumTilings} = \frac{1}{16}$
  * `end` = $\frac{1}{10NumTilings} = \frac{1}{80}$


#### Comparing different Agents (Algorithms)
  
| Algorithm      | Parameters                    | Train                                                                                                           | Evaluation                                                                                                     | 
|----------------|-------------------------------|-----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| Sarsa          | $n = 4$, $\varepsilon = 0.01$ | <img src="images/results/BaseReward_4StepSemiGradientSarsa_Train_eps_0.01.png" alt="Grid" width="400"/>         | <img src="images/results/BaseReward_4StepSemiGradientSarsa_Eval_eps_0.01.png" alt="Grid" width="400"/>         |
| Expected-Sarsa | $n = 4$, $\varepsilon = 0.01$ | <img src="images/results/BaseReward_4StepSemiGradientExpectedSarsa_Train_eps_0.01.png" alt="Grid" width="400"/> | <img src="images/results/BaseReward_4StepSemiGradientExpectedSarsa_Eval_eps_0.01.png" alt="Grid" width="400"/> |
| Q-Learning     | $n = 4$, $\varepsilon = 0.01$ | <img src="images/results/BaseReward_4StepSemiGradientQLearning_Train_eps_0.01.png" alt="Grid" width="400"/>     | <img src="images/results/BaseReward_4StepSemiGradientQLearning_Eval_eps_0.01.png" alt="Grid" width="400"/>     |


#### Comparing different n-steps

* $\varepsilon = 0.01$

| Algorithm      | $n = 2$                                                                                                        | $n = 4$                                                                                                        | $n = 6$                                                                                                        | $n = 8$                                                                                                        |
|----------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| Sarsa          | <img src="images/results/BaseReward_2StepSemiGradientSarsa_Eval_eps_0.01.png" alt="Grid" width="400"/>         | <img src="images/results/BaseReward_4StepSemiGradientSarsa_Eval_eps_0.01.png" alt="Grid" width="400"/>         | <img src="images/results/BaseReward_6StepSemiGradientSarsa_Eval_eps_0.01.png" alt="Grid" width="400"/>         | <img src="images/results/BaseReward_8StepSemiGradientSarsa_Eval_eps_0.01.png" alt="Grid" width="400"/>         |
| Expected-Sarsa | <img src="images/results/BaseReward_2StepSemiGradientExpectedSarsa_Eval_eps_0.01.png" alt="Grid" width="400"/> | <img src="images/results/BaseReward_4StepSemiGradientExpectedSarsa_Eval_eps_0.01.png" alt="Grid" width="400"/> | <img src="images/results/BaseReward_6StepSemiGradientExpectedSarsa_Eval_eps_0.01.png" alt="Grid" width="400"/> | <img src="images/results/BaseReward_8StepSemiGradientExpectedSarsa_Eval_eps_0.01.png" alt="Grid" width="400"/> |
| Q-Learning     | <img src="images/results/BaseReward_2StepSemiGradientQLearning_Eval_eps_0.01.png" alt="Grid" width="400"/>     | <img src="images/results/BaseReward_4StepSemiGradientQLearning_Eval_eps_0.01.png" alt="Grid" width="400"/>     | <img src="images/results/BaseReward_6StepSemiGradientQLearning_Eval_eps_0.01.png" alt="Grid" width="400"/>     | <img src="images/results/BaseReward_8StepSemiGradientQLearning_Eval_eps_0.01.png" alt="Grid" width="400"/>     |

Results for other parameters and reward shaping functions are located under `images/results` folder.


## Continuing Task Semi-Gradient Control: 1-step
#### One-Step Differential Semi-Gradient Sarsa
The differential semi-gradient Sarsa algorithm (for estimating q) is shown below:

<img src="images/DifferentialSemiGradientSarsa.png" alt="Grid" width="800"/>

## Continuing Task Semi-Gradient Experiments: 1-step
The same MountainCar environment is used for these experiments, where the 
environment is set to be continuous (40k steps). Since the environment 
"out of the box" simply resets the vehicle to the starting position 
when it reaches the flag, and continues to generate "-1" reward, the environment
was modified to award a reward of "100" upon reaching the flag. This was done 
to show the effect of the __average reward__ setting on the agent's learning,
since a continuing reward of "-1" has the same average reward (of -1).

* Parameters
  * $40k$ steps to approximate continuing task 
  * $\varepsilon$ linearly decayed over $\frac{40k}{3}$ steps
  

| Algorithm      | Parameters  | Results                                                                                                         | 
|----------------|-------------|-----------------------------------------------------------------------------------------------------------------|
| Sarsa          | $eps = 0.3$ | <img src="images/results/BaseReward_DifferentialSemiGradientSarsa_eps_0.3.png" alt="Grid" width="400"/>         |
| Expected Sarsa | $eps = 0.3$ | <img src="images/results/BaseReward_DifferentialSemiGradientExpectedSarsa_eps_0.3.png" alt="Grid" width="400"/> |
| Q-Learning     | $eps = 0.3$ | <img src="images/results/BaseReward_DifferentialSemiGradientQLearning_eps_0.3.png" alt="Grid" width="400"/>     |

For other $\varepsilon$ settings, please refer to `images/results` folder.

## Continuing Task Semi-Gradient Control: n-step
#### $n$-Step Differential Semi-gradient Sarsa

<img src="images/Differential_nStep_Semi_Gradient_Sarsa.png.png" alt="Grid" width="800"/>


## Continuing Task Semi-Gradient Control Experiments: n-step
Just like above, MountainCar environment - modified for the continuing task 
case is also used.

* Parameters
  * $40k$ steps to approximate continuing task 
  * $\varepsilon$ linearly decayed over $\frac{40k}{3}$ steps
  * $\beta$ = 0.05
  * $n$ = 4

| Agent          | Results                                                                                                               | 
|----------------|-----------------------------------------------------------------------------------------------------------------------|
| Sarsa          | <img src="images/results/BaseReward_DifferentialSemiGradient_4StepSarsa_eps_0.3.png" alt="Grid" width="400"/>         |
| Expected Sarsa | <img src="images/results/BaseReward_DifferentialSemiGradient_4StepExpectedSarsa_eps_0.3.png" alt="Grid" width="400"/> |
| Q-Learning     | <img src="images/results/BaseReward_DifferentialSemiGradient_4StepQLearning_eps_0.3.png" alt="Grid" width="400"/>     |

More experiments with different starting $\varepsilon$ are located in `images/results`

## Environment
Here the [MountainCar](https://gymnasium.farama.org/environments/classic_control/mountain_car/) environment from OpenAI's Gymnasium is used.
This is a discrete control environment where the agent is a car that must reach the flag at the top of the hill.

The discrete action space is defined as:
* 0 - Accelerate to the left
* 1 - Don’t accelerate
* 2 - Accelerate to the right

The continuous state-space is discretized into feature vectors using tile-coding from [`tiles3.py`](http://incompleteideas.net/tiles/tiles3.py-remove) - where, as in footnote (1) in the book - it is used as:
- `iht=IHT(4096)` 
- `tiles(iht,8,[8*x/(0.5+1.2),8*xdot/(0.07+0.07)],[A])`

where the number of tiles & tilings are set to 8.

The [Mountain Car]([MountainCar](https://gymnasium.farama.org/environments/classic_control/mountain_car/#mountain-car)) MDP is a deterministic MDP that consists of a car placed 
stochastically at the bottom of a sinusoidal valley, with the only 
possible actions being the accelerations that can be applied to the car 
in either direction. The goal of the MDP is to strategically accelerate 
the car to reach the goal state on top of the right hill.

The goal is to reach the flag placed on top of the right hill as quickly as 
possible, as such the agent is penalised with a reward of -1 
for each timestep.

_In these experiments the discrete action is used. The reward was modified
for the __continuing task__ case, such that when the flag is reached, a 
"fake" reward of +100 is awarded, otherwise the average reward is constant
-1.0 and there is nothing to optimize_.
