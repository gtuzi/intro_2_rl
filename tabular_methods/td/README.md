[Sutton & Barto RL Book]: http://incompleteideas.net/book/RLbook2020.pdf

# Temporal Difference and n-Step Bootstrapping


## Table of Contents
- [Introduction](#introduction)
- [Implemented Algorithms](#implemented-algorithms)
- [Execution](#execution)
- [Environments](#environments)


## Introduction
This section contains methods from Chapter 6 & 7 in [Sutton & Barto RL Book].
Expanded discussions are in [summary](summary.ipynb)

## Implemented Algorithms
- [x] Sarsa (Section: 6.4): `agents/Sarsa`
- [x] ExpectedSarsa (Section: 6.6): `agents/ExpectedSarsa`
- [x] QLearning / SarsaMax (Section: 6.5): `agents/QLearning`
- [x] nStepSarsa (Section: 7.2): `agents/nStepSarsa`
- [x] nStepsSarsaOffPolicy (Section: 7.3): `agents/nStepsSarsaOffPolicy`
- [x] QSigmaOffPolicy (Section 7.6): `agents/QSigmaOffPolicy`

## Algorithms
The following algorithms have been implemented

### Sarsa

<img src="images/pub/SarsaAlgo.png" alt="Grid" width="800"/>


### Expected Sarsa
Same as Sarsa, however for next state–action pairs it uses the _expected_ value, 
taking into account how likely each action is under the _current_ policy.

<img src="images/pub/ExpectedSarsa_UpdateRule.png" alt="Grid" width="2002"/>


### Sarsa-Max (Q Learning)

<img src="images/pub/QLearningAlgo.png" alt="Grid" width="800"/>


### nStep Sarsa

<img src="images/pub/nStepSarsa.png" alt="Grid" width="800"/>


### Offpolicy - nStepSarsa &  Q(&sigma;)
As of this writing, these implementations are not yielding good results. Further work is necessary. 


## Experiments

### Environments Setup
Different environments have been tested with the following parameters

* Num train seeds = $10$
* Num eval seeds = $100$
* $\gamma = 0.99$

--- 

* _Frozen Lake_ - stochastic: 
  * Reward
    * Reach goal: $+1$
    * Reach hole: 0
    * Reach frozen: 0
    * Slippery: True (stochastic)
    * Map: $4 \times 4$
  * Initial state-action value $q_{init} = -1$
  * Number of episodes = $10000$ 
  * Episode horizon = $100$ 
  * On-Policy Algorithms
    * $\varepsilon$ linearly annealed $[1, 0.01]$
    * $\alpha$ linearly annealed $[0.3, 0.1]$
  * Off-Policy Algorithms
    * $\varepsilon$ linearly annealed $[1, 0.01]$
    * $\alpha$ linearly annealed $[0.3, 0.1]$

---

* CliffWalking - is not stochastic
  * Reward
    * Each time step incurs $-1$ reward 
    * Player stepped into the cliff incurs $-100$ reward
  * Initial state-action value $q_{init} = 0$
  * Number of episodes = $5000$ 
  * Episode horizon = $20$ 
  * On-Policy Algorithms
    * $\varepsilon$ linearly annealed $[1, 0.01]$
    * $\alpha$ linearly annealed $[0.5, 0.01]$
  * Off-Policy Algorithms
    * $\varepsilon$ linearly annealed $[1, 0.01]$
    * $\alpha$ linearly annealed $[0.3, 0.01]$

---

* Taxi - taxi is not stochastic.
  * Reward
    * $-1$ per step unless other reward is triggered
    * $+20$ delivering passenger
    * $-10$ executing “pickup” and “drop-off” actions illegally.
  * Initial state-action value $q_{init} = 0$
  * Number of episodes = $5000$ 
  * Episode horizon = $50$ 
  * On-Policy Algorithms
    * $\varepsilon$ linearly annealed $[0.3, 0.001]$
    * $\alpha$ linearly annealed $[1, 0.2]$
  * Off-Policy Algorithms
    * $\varepsilon$ linearly annealed $[1.0, 0.001]$
    * $\alpha$ linearly annealed $[0.1, 0.001]$


##### Algorithm Parameters

All the algorithms shown share the same parameters per environment as shown.
nStepSarsa used $n=4$


#### Results
* Greedy action and policy: $a \sim \pi_h(\cdot|s) = \arg \max_{a'} Q(s, a')$
* Soft action and policy: $a \sim \pi_f(\cdot|s) = \varepsilon-\text{greedy}(Q(s, \cdot))$
* Hard sum of discounted rewards: $G_{t, \pi_h} = \sum_{k = t} \gamma^{k - t}r_t$
* Soft sum of discounted rewards: $G_{t, \pi_f} = \sum_{k = t} \gamma^{k - t}r_t$
* Hard average sum of rewards: $R_{0, \pi_h} = \sum_{t=0} r^{n}_t$
* Soft average sum of rewards: $R_{0, \pi_f} = \sum_{t=0} r^{n}_t$
* Hard average sum of discounted rewards: $\bar{G}_{0, h} = \frac{1}{N}\sum_n G^n_{0, h}$
* Soft average sum of discounted rewards: $\bar{G}_{0, f} = \frac{1}{N}\sum_n G^n_{0, f}$
* Average initial state value: $\bar{V}(s_0) = \frac{1}{N}\sum_n V(s^n_0)$
* Hard average sum of rewards: $\bar{R}_{0, \pi_h} = \frac{1}{N}\sum_{n}^{N}R_{0, \pi_h}$
* Hard average sum of rewards: $\bar{R}_{0, \pi_f} = \frac{1}{N}\sum_{n}^{N}R_{0, \pi_f}$


#### Performance Evaluation

The following table answers the question: how does each algorithm perform, wrt
1) pure returns
2) discounted returns (its objective)
3) episode length on hard simulations

Under these scenarios we concern ourselves with how well the algo works 
in an (almost) black-box fashion.


#### On-Policy Results

| Environment       | $\bar{R}_{0, h}$                                                                                                        | $\bar{G}_{0, h}$                                                                                           | Average Episode Length                                                                                                 |
|-------------------|-------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------| 
| Frozen Lake (v1)  | <img src="images/evaluation_metrics/on_policy/FrozenLake_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>   | <img src="images/evaluation_metrics/on_policy/FrozenLake_mean_hard_eval_G0.png" alt="Grid" width="400"/>   | <img src="images/evaluation_metrics/on_policy/FrozenLake_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>   |
| CliffWalking (v0) | <img src="images/evaluation_metrics/on_policy/CliffWalking_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/on_policy/CliffWalking_mean_hard_eval_G0.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/on_policy/CliffWalking_mean_hard_eval_episode_length.png" alt="Grid" width="400"/> |
| Taxi (v3)         | <img src="images/evaluation_metrics/on_policy/Taxi_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>         | <img src="images/evaluation_metrics/on_policy/Taxi_mean_hard_eval_G0.png" alt="Grid" width="400"/>         | <img src="images/evaluation_metrics/on_policy/Taxi_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>         |


#### Off-Policy Results

For off-policy, evaluation is performed by the target policy

| Environment       | $\bar{R}_{0, h}$                                                                                                         | $\bar{G}_{0, h}$                                                                                            | Average Episode Length                                                                                                  |
|-------------------|--------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------| 
| Frozen Lake (v1)  | <img src="images/evaluation_metrics/off_policy/FrozenLake_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>   | <img src="images/evaluation_metrics/off_policy/FrozenLake_mean_hard_eval_G0.png" alt="Grid" width="400"/>   | <img src="images/evaluation_metrics/off_policy/FrozenLake_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>   |
| CliffWalking (v0) | <img src="images/evaluation_metrics/off_policy/CliffWalking_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/off_policy/CliffWalking_mean_hard_eval_G0.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/off_policy/CliffWalking_mean_hard_eval_episode_length.png" alt="Grid" width="400"/> |
| Taxi (v3)         | <img src="images/evaluation_metrics/off_policy/Taxi_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>         | <img src="images/evaluation_metrics/off_policy/Taxi_mean_hard_eval_G0.png" alt="Grid" width="400"/>         | <img src="images/evaluation_metrics/off_policy/Taxi_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>         |


#### Learned Correctness Evaluation
Correctness concerns itself with how well does the algorithm learn. 
In the following table the following questions are addressed:
* Is the agent learning ? This is measured by the loss, i.e. TD error approaching 0.
* How does the expected initial state value compare to the actual discounted 
returns. Are we learning the objective function as expected. Note that here I am
looking at the _soft_ evaluation $G_{0, f}$ in order to measure the quality of 
_expectation_ of $V(s_0) \overset \cdot{=} \mathbb{E}[R_0 | s_0]$.
 _Note_ that here initial value $q_{init}$ indicates the starting levels of the 
expected values in $V$


#### On-Policy Results

| Environment       | $\bar{G}_{0, f}$ vs $\bar{V}(s_0)$                                                                  | Avg Loss                                                                                                    |
|-------------------|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| Frozen Lake (v1)  | <img src="images/learning/on_policy/FrozenLake_value_accuracy_JOINT.png" alt="Grid" width="400"/>   | <img src="images/training_metrics/on_policy/FrozenLake_mean_behavioral_loss.png" alt="Grid" width="400"/>   |
| CliffWalking (v0) | <img src="images/learning/on_policy/CliffWalking_value_accuracy_JOINT.png" alt="Grid" width="400"/> | <img src="images/training_metrics/on_policy/CliffWalking_mean_behavioral_loss.png" alt="Grid" width="400"/> |
| Taxi (v3)         | <img src="images/learning/on_policy/Taxi_value_accuracy_JOINT.png" alt="Grid" width="400"/>         | <img src="images/training_metrics/on_policy/Taxi_mean_behavioral_loss.png" alt="Grid" width="400"/>         |


#### Off-Policy Results

For off-policy, evaluation is performed by the target policy

| Environment       | $\bar{G}_{0, f}$ vs $\bar{V}(s_0)$                                                                   | Avg Loss                                                                                                 |
|-------------------|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Frozen Lake (v1)  | <img src="images/learning/off_policy/FrozenLake_value_accuracy_JOINT.png" alt="Grid" width="400"/>   | <img src="images/training_metrics/off_policy/FrozenLake_mean_target_loss.png" alt="Grid" width="400"/>   |
| CliffWalking (v0) | <img src="images/learning/off_policy/CliffWalking_value_accuracy_JOINT.png" alt="Grid" width="400"/> | <img src="images/training_metrics/off_policy/CliffWalking_mean_target_loss.png" alt="Grid" width="400"/> |
| Taxi (v3)         | <img src="images/learning/off_policy/Taxi_value_accuracy_JOINT.png" alt="Grid" width="400"/>         | <img src="images/training_metrics/off_policy/Taxi_mean_target_loss.png" alt="Grid" width="400"/>         |


## Execution
Run code in `main.py`. Each algorithm has its own `experiments` task.

## Environments
- [Gymnasium] - `FrozenLake-v1`, `CliffWalking-v0`, `Taxi-v3`