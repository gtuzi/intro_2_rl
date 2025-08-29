[Sutton & Barto RL Book]: http://incompleteideas.net/book/RLbook2020.pdf

# Monte Carlo Methods
Monte Carlo methods require only experience—sample sequences of states, actions, and rewards from actual or simulated interaction with an environment. Here we learn value functions from _sample_ returns. The estimates for each state are independent. The estimate for one state does not build upon the estimate of any other state, as is the case in DP. In other words, Monte Carlo methods do not bootstrap.


## Table of Contents
- [Background and Explanations](#background-and-explanations)
- [Implemented Algorithms](#Implemented-Algorithms)
- [On-Policy, First Visit](#on-policy-first-visit)
- [Off-Policy MC Control with Importance Sampling](#off-policy-mc-control-with-importance-sampling)
- [Environments and Agent Parameter Setup](#environments-and-agent-parameter-setup)
- [Metrics and Other Definitions](#metrics-and-other-definitions)
- [Results](#results)
  - [Performance Evaluation](#performance-evaluation)
  - [Learned Correctness](#learned-correctness)
- [Execution](#Execution)


## Background and Explanations
See expanded discussion of the algorithms and concepts covered here: [summary.ipynb](summary.ipynb).

Topics covered in the notebook:

* __Introduction Monte Carlo Control__
* __On-Policy First Visit__
* __Off-policy Prediction via Importance Sampling__
* __Off-Policy MC Control__


## Implemented Algorithms
- [x] On-policy first-visit MC control (5.4): `agents/MCOnPolicyFirstVisitGLIE`
- [x] Off-Policy MC control for estimating optimal $\pi_*$ (5.7): `agents/MCOffPolicy`


## On-Policy First Visit
<img src="images/MCOnPolicyFirstVisit.png" alt="Grid" width="700"/>

## Off-Policy MC Control with Importance Sampling
<img src="images/MCOffPolicy.png" alt="Grid" width="700"/>


---

## Environments and Agent Parameter Setup

Different environments have been tested with the following parameters

* Num train seeds = $10$
* Num eval seeds = $100$
* $\gamma = 0.99$


* __Frozen Lake__ - stochastic: 
  * Reward
    * Reach goal: $+1$
    * Reach hole: 0
    * Reach frozen: 0
    * _Slippery_: True (stochastic)  / False
    * Map: $4 \times 4$
  * Initial state-action value $q_{init} = -1$
  * Number of episodes = $500$ 
  * Episode horizon = $50$
  * $\varepsilon$ linearly annealed from starting $\varepsilon$ to $\varepsilon * 0.001$
  * On-Policy Algorithms
    * $\alpha$ linearly annealed $[0.3, 0.01]$
  * Off-Policy Algorithms
    * $\alpha$ linearly annealed $[1e-1, 1e-5]$


* __CliffWalking__
  * Reward
    * Each time step incurs $-1$ reward 
    * Player stepped into the cliff incurs $-100$ reward
  * Initial state-action value $q_{init} = 0$
  * Number of episodes = $5000$ 
  * Episode horizon = $20$
  * $\varepsilon$ linearly annealed from starting $\varepsilon$ to $\varepsilon * 0.001$
  * On-Policy Algorithms
    * $\alpha$ linearly annealed $[0.5, 0.01]$
  * Off-Policy Algorithms
    * $\alpha$ linearly annealed $[0.3, 0.01]$


* __Taxi__
  * Reward
    * $-1$ per step unless other reward is triggered
    * $+20$ delivering passenger
    * $-10$ executing “pickup” and “drop-off” actions illegally.
  * Initial state-action value $q_{init} = 0$
  * Number of episodes = $5000$ 
  * Episode horizon = $50$ 
  * $\varepsilon$ linearly annealed from starting $\varepsilon$ to $\varepsilon * 0.001$ 
  * On-Policy Algorithms
    * $\alpha$ linearly annealed $[1, 0.2]$
  * Off-Policy Algorithms
    * $\alpha$ linearly annealed $[0.1, 0.001]$

---

## Metrics and Other Definitions

* Greedy action and policy: $a \sim \pi_h(\cdot|s) = \arg \max_{a'} Q(s, a')$
* Soft action and policy: $a \sim \pi_f(\cdot|s) = \varepsilon-\text{greedy}(Q(s, \cdot))$
* Hard sum of discounted rewards: $G_{t, \pi_h} = \sum_{k = t} \gamma^{k - t}r_t$
* Soft sum of discounted rewards: $G_{t, \pi_f} = \sum_{k = t} \gamma^{k - t}r_t$
* Hard sum of raw rewards: $R_{0, \pi_h} = \sum_{t=0} r_t$
* Soft sum of raw rewards: $R_{0, \pi_f} = \sum_{t=0} r_t$
* Hard average sum of discounted rewards over N trials/seeds: $\bar{G}_{0, h} = \frac{1}{N}\sum_n G^{(n)}_{0, h}$
* Soft average sum of discounted rewards over N trials/seeds: $\bar{G}_{0, f} = \frac{1}{N}\sum_n G^{(n)}_{0, f}$
* Average initial state value over N trials/seeds: $\bar{V}(s_0) = \frac{1}{N}\sum_n V(s^{(n)}_0)$
* Hard average sum of raw rewards over N trials/seeds: $\bar{R}_{0, \pi_h} = \frac{1}{N}\sum_{n}^{N}R^{(n)}_{0, \pi_h}$
* Hard average sum of raw rewards over N trials/seeds: $\bar{R}_{0, \pi_f} = \frac{1}{N}\sum_{n}^{N}R^{(n)}_{0, \pi_f}$


## Results

### Performance Evaluation

The following table answer the question: how does each algorithm perform, wrt
1) pure returns
2) discounted returns (its objective)
3) episode length on hard (i.e. greedy) evaluations

Under these scenarios we concern ourselves with how well the algo works 
in an (almost) black-box fashion.


#### On-Policy

| Environment               | $\bar{R}_{0, h}$                                                                                                               | $\bar{G}_{0, h}$                                                                                                  | Average Episode Length                                                                                                        |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------| 
| Frozen Lake (v1)          | <img src="images/evaluation_metrics/on_policy/FrozenLake_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>          | <img src="images/evaluation_metrics/on_policy/FrozenLake_mean_hard_eval_G0.png" alt="Grid" width="400"/>          | <img src="images/evaluation_metrics/on_policy/FrozenLake_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>          |
| Frozen Lake-Slippery (v1) | <img src="images/evaluation_metrics/on_policy/FrozenLake-Slippery_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/on_policy/FrozenLake-Slippery_mean_hard_eval_G0.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/on_policy/FrozenLake-Slippery_mean_hard_eval_episode_length.png" alt="Grid" width="400"/> |
| CliffWalking (v0)         | <img src="images/evaluation_metrics/on_policy/CliffWalking_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>        | <img src="images/evaluation_metrics/on_policy/CliffWalking_mean_hard_eval_G0.png" alt="Grid" width="400"/>        | <img src="images/evaluation_metrics/on_policy/CliffWalking_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>        |
| Taxi (v3)                 | <img src="images/evaluation_metrics/on_policy/Taxi_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>                | <img src="images/evaluation_metrics/on_policy/Taxi_mean_hard_eval_G0.png" alt="Grid" width="400"/>                | <img src="images/evaluation_metrics/on_policy/Taxi_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>                |


#### Off Policy

For off-policy below are evaluated the case when behavioral agent is stateless
and samples actions uniformly, and the case when it is a soft greedy agent, 
in this case I am using MCAgent. Epsilon (i.e exploration) is kept fairly high, 
however epsilon is annealed.

#### Off-Policy: Uniform Behavioral Agent

For off-policy, evaluation is performed by the target policy

| Environment               | $\bar{R}_{0, h}$                                                                                                                                   | $\bar{G}_{0, h}$                                                                                                                      | Average Episode Length                                                                                                                            |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------| 
| Frozen Lake (v1)          | <img src="images/evaluation_metrics/off_policy_uniform_behavioral/FrozenLake_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>          | <img src="images/evaluation_metrics/off_policy_uniform_behavioral/FrozenLake_mean_hard_eval_G0.png" alt="Grid" width="400"/>          | <img src="images/evaluation_metrics/off_policy_uniform_behavioral/FrozenLake_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>          |
| Frozen Lake-Slippery (v1) | <img src="images/evaluation_metrics/off_policy_uniform_behavioral/FrozenLake-Slippery_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/off_policy_uniform_behavioral/FrozenLake-Slippery_mean_hard_eval_G0.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/off_policy_uniform_behavioral/FrozenLake-Slippery_mean_hard_eval_episode_length.png" alt="Grid" width="400"/> |
| CliffWalking (v0)         | <img src="images/evaluation_metrics/off_policy_uniform_behavioral/CliffWalking_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>        | <img src="images/evaluation_metrics/off_policy_uniform_behavioral/CliffWalking_mean_hard_eval_G0.png" alt="Grid" width="400"/>        | <img src="images/evaluation_metrics/off_policy_uniform_behavioral/CliffWalking_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>        |
| Taxi (v3)                 | <img src="images/evaluation_metrics/off_policy_uniform_behavioral/Taxi_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>                | <img src="images/evaluation_metrics/off_policy_uniform_behavioral/Taxi_mean_hard_eval_G0.png" alt="Grid" width="400"/>                | <img src="images/evaluation_metrics/off_policy_uniform_behavioral/Taxi_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>                |


#### Off-Policy: Greedy Behavioral Agent

For off-policy, evaluation is performed by the target policy

| Environment               | $\bar{R}_{0, h}$                                                                                                                                  | $\bar{G}_{0, h}$                                                                                                                     | Average Episode Length                                                                                                                           |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------| 
| Frozen Lake (v1)          | <img src="images/evaluation_metrics/off_policy_greedy_behavioral/FrozenLake_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>          | <img src="images/evaluation_metrics/off_policy_greedy_behavioral/FrozenLake_mean_hard_eval_G0.png" alt="Grid" width="400"/>          | <img src="images/evaluation_metrics/off_policy_greedy_behavioral/FrozenLake_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>          |
| Frozen Lake (v1)-Slippery | <img src="images/evaluation_metrics/off_policy_greedy_behavioral/FrozenLake-Slippery_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/off_policy_greedy_behavioral/FrozenLake-Slippery_mean_hard_eval_G0.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/off_policy_greedy_behavioral/FrozenLake-Slippery_mean_hard_eval_episode_length.png" alt="Grid" width="400"/> |
| CliffWalking (v0)         | <img src="images/evaluation_metrics/off_policy_greedy_behavioral/CliffWalking_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>        | <img src="images/evaluation_metrics/off_policy_greedy_behavioral/CliffWalking_mean_hard_eval_G0.png" alt="Grid" width="400"/>        | <img src="images/evaluation_metrics/off_policy_greedy_behavioral/CliffWalking_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>        |
| Taxi (v3)                 | <img src="images/evaluation_metrics/off_policy_greedy_behavioral/Taxi_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>                | <img src="images/evaluation_metrics/off_policy_greedy_behavioral/Taxi_mean_hard_eval_G0.png" alt="Grid" width="400"/>                | <img src="images/evaluation_metrics/off_policy_greedy_behavioral/Taxi_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>                |


### Learned Correctness
Correctness concerns itself with how well does the algorithm learn. 
In the following table the following questions are addressed:
* Is the agent learning ? This is measured by the loss, i.e. TD error approaching 0.
* How does the expected initial state value compare to the actual discounted 
returns. Are we learning the objective function as expected. Note that here I am
looking at the _soft_ evaluation $G_{0, f}$ in order to measure the quality of 
_expectation_ of $V(s_0) \overset \cdot{=} \mathbb{E}[R_0 | s_0]$.
 _Note_ that here initial value $q_{init}$ indicates the starting levels of the 
expected values in $V$


#### On-Policy

| Environment               | $\bar{G}_{0, f}$ vs $\bar{V}(s_0)$                                                                         | Avg Loss                                                                                                           |
|---------------------------|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| Frozen Lake (v1)          | <img src="images/learning/on_policy/FrozenLake_value_accuracy_JOINT.png" alt="Grid" width="400"/>          | <img src="images/training_metrics/on_policy/FrozenLake_mean_behavioral_loss.png" alt="Grid" width="400"/>          |
| Frozen Lake (v1)-Slippery | <img src="images/learning/on_policy/FrozenLake-Slippery_value_accuracy_JOINT.png" alt="Grid" width="400"/> | <img src="images/training_metrics/on_policy/FrozenLake-Slippery_mean_behavioral_loss.png" alt="Grid" width="400"/> |
| CliffWalking (v0)         | <img src="images/learning/on_policy/CliffWalking_value_accuracy_JOINT.png" alt="Grid" width="400"/>        | <img src="images/training_metrics/on_policy/CliffWalking_mean_behavioral_loss.png" alt="Grid" width="400"/>        |
| Taxi (v3)                 | <img src="images/learning/on_policy/Taxi_value_accuracy_JOINT.png" alt="Grid" width="400"/>                | <img src="images/training_metrics/on_policy/Taxi_mean_behavioral_loss.png" alt="Grid" width="400"/>                |


#### Off-Policy: Uniform Behavioral Agent 

For off-policy, evaluation is performed by the target policy

| Environment               | $\bar{G}_{0, f}$ vs $\bar{V}(s_0)$                                                                                             | Avg Loss                                                                                                                           |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| Frozen Lake (v1)          | <img src="images/learning/off_policy_uniform_behavioral/FrozenLake_value_accuracy_JOINT.png" alt="Grid" width="400"/>          | <img src="images/training_metrics/off_policy_uniform_behavioral/FrozenLake_mean_target_loss.png" alt="Grid" width="400"/>          |
| Frozen Lake (v1)-Slippery | <img src="images/learning/off_policy_uniform_behavioral/FrozenLake-Slippery_value_accuracy_JOINT.png" alt="Grid" width="400"/> | <img src="images/training_metrics/off_policy_uniform_behavioral/FrozenLake-Slippery_mean_target_loss.png" alt="Grid" width="400"/> |
| CliffWalking (v0)         | <img src="images/learning/off_policy_uniform_behavioral/CliffWalking_value_accuracy_JOINT.png" alt="Grid" width="400"/>        | <img src="images/training_metrics/off_policy_uniform_behavioral/CliffWalking_mean_target_loss.png" alt="Grid" width="400"/>        |
| Taxi (v3)                 | <img src="images/learning/off_policy_uniform_behavioral/Taxi_value_accuracy_JOINT.png" alt="Grid" width="400"/>                | <img src="images/training_metrics/off_policy_uniform_behavioral/Taxi_mean_target_loss.png" alt="Grid" width="400"/>                |


#### Off-Policy: Greedy Behavioral Agent 


| Environment               | $\bar{G}_{0, f}$ vs $\bar{V}(s_0)$                                                                                            | Avg Loss                                                                                                                          |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| Frozen Lake (v1)          | <img src="images/learning/off_policy_greedy_behavioral/FrozenLake_value_accuracy_JOINT.png" alt="Grid" width="400"/>          | <img src="images/training_metrics/off_policy_greedy_behavioral/FrozenLake_mean_target_loss.png" alt="Grid" width="400"/>          |
| Frozen Lake (v1)-Slippery | <img src="images/learning/off_policy_greedy_behavioral/FrozenLake-Slippery_value_accuracy_JOINT.png" alt="Grid" width="400"/> | <img src="images/training_metrics/off_policy_greedy_behavioral/FrozenLake-Slippery_mean_target_loss.png" alt="Grid" width="400"/> |
| CliffWalking (v0)         | <img src="images/learning/off_policy_greedy_behavioral/CliffWalking_value_accuracy_JOINT.png" alt="Grid" width="400"/>        | <img src="images/training_metrics/off_policy_greedy_behavioral/CliffWalking_mean_target_loss.png" alt="Grid" width="400"/>        |
| Taxi (v3)                 | <img src="images/learning/off_policy_greedy_behavioral/Taxi_value_accuracy_JOINT.png" alt="Grid" width="400"/>                | <img src="images/training_metrics/off_policy_greedy_behavioral/Taxi_mean_target_loss.png" alt="Grid" width="400"/>                |


## Execution
Run code in `main.py`. Each algorithm has its own `experiments` task.