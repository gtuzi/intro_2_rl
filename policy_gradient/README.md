[Sutton & Barto RL Book]: http://incompleteideas.net/book/RLbook2020.pdf

# Policy Gradient Methods


## Table of Contents
- [Introduction](#Introduction)
- [Implemented Algorithms](#implemented-algorithms)
- [Detailed Development](#explanations-development-and-experimental-details)
- [REINFORCE - MC Policy Gradient](#reinforce-the-monte-carlo-policy-gradient-)
- [REINFORCE with Baseline](#reinforce-with-baseline)
- [Actor Critic Methods](#actor---critic-methods---introduction)
- [One-Step Actor Critic](#one-step-actorcritic)
- [n-Step with Eligibility Traces Actor–Critic](#n-step-with-eligibility-traces-actorcritic)
- [Policy Gradient for Continuing Problems](#policy-gradient-for-continuing-problems)
- [Policy Parameterization for _Continuous Actions_](#policy-parameterization-for-_continuous-actions_)


## Introduction
In policy gradient (PG) methods, the policy does not consult 
the action values in its decision of action. Here, a parametrized policy is 
learned which selects the action without consulting the estimated action value.
The value function may still be learned, with the aim of learning the policy
(parameters), but it is not specifically consulted in order to take the 
action.

In PG methods, learning the policy parameter is based on the  gradient of the 
scalar performance measure wherein these methods aim to
_maximize_ its value, via gradient _ascent_.

All PG methods follow this general schema - independent of whether they learn 
a state/action value function or not. Methods which do learn value functions
are usually called _actor-critic_, where actor refers to the policy and the 
critic the state or (most often) action value function.


## Implemented Algorithms
- [x] REINFORCE: `agents/Reinforce`
- [x] REINFORCE with Baseline: `agents/ReinforceBaseline`
- [x] One-Step Actor Critic: `agents/OneStepAC`
- [x] n-Step Actor Critic with Eligibility Traces: `agents/ACWithEligibilityTraces`
- [x] Actor Critic with Eligibility Traces, Continuing Task: `agents/ACWithEligibilityTracesContinuing`
- [x] REINFORCE for Continuous Action: `agents/ReinforceContinuousAction`
- [x] REINFORCE with Baseline for Continuous Action: `agents/ReinforceBaselineContinuousAction`
- [x] Actor Critic with Eligibility Traces for Continuous Action: `agents/ACWithEligibilityTracesContinuousAction`
- [x] Actor Critic with Eligibility Traces for Continuous Action, Continuous Task: `agents/ACWithEligibilityTracesContinuousActionContinuingTask`


## Explanations, Development, and Experimental Details
Full development and discussion visit the notebook [here](summary.ipynb)


## REINFORCE: The Monte Carlo Policy Gradient 

REINFORCE uses the complete return from time $t$, which includes all
future rewards up until the end of the episode. In this sense REINFORCE is a Monte
Carlo algorithm and is well defined only for the episodic case with all updates made in
retrospect after the episode is completed.

<img src="images/pub/reinforce.png" alt="Grid" width="450"/>

The implementation is in `agents/Reinforce`


## REINFORCE with Baseline

The PG theorem can be generalized to include a baseline $b(s)$:

$$
\nabla J(\mathbf{\theta}) \propto \sum_s \mu(s) \sum_a \Bigl(q_{\pi}(s, a) - b(s) \Bigr) \nabla \pi(a | s, \mathbf{\theta}) 
$$

The new update rule for REINFORCE with baseline is:

$$
\mathbf{\theta}_{t+1} = \mathbf{\theta}_t + \alpha \Bigl(G_t - b(S_t) \Bigr) \nabla \log \pi(A_t | S_t, \mathbf{\theta})
$$

A common choice for the baseline is the state value function 
$\hat{v}(S_t, \mathbf{w}_t)$ where the parameter $\mathbf{w} \in \mathbb{R}^d$ 
iss learned and updated like in the linear approximation methods (refer to 
those methods). So in this case we would have 2 sets of parameters 
we're learning, $\mathbf{\theta}$ and $\mathbf{w}$.  Because REINFORCE is a Monte-Carlo method, i.e. we use complete 
returns generated _at the end of the episode_, to learn the policy parameters
$\mathbf{\theta}$, we can also use the same method for learning the parameters
of the state value function $\mathbf{w}$.

<img src="images/pub/reinforce_with_baseline.png" alt="Grid" width="450"/>

The implementation is in `agents/ReinforceBaseline`


## Actor - Critic Methods - Introduction
In REINFORCE with baseline, we use a state value function (estimate) of the 
state _before_ the action is taken, i.e. $S_t$. This estimate sets a baseline 
for the ensuing return $R_{t+1}$, but it cannot be used to evaluate 
the action $A_t$. In Actor-Critic (AC) methods we evaluate the following 
state ($S_{t+1}$) as well. The estimated value of the second state, when
discounted and added to the reward, constitutes the one-step return $G_{t:t+1}$ 
which is a useful estimate of the actual return and thus is a way of 
assessing the action. With this formulation, we can modulate the bias furthermore
by using $n$-step returns and eligibility traces. When the
state-value function is used to assess actions in this way it is called a 
_critic_, and the overall policy-gradient method is termed an _actor–critic_
method.


## One-Step Actor–Critic
They are the analog of the TD methods, such as TD(0), Sarsa(0), and Q-learning.
They are fully online and incremental, yet avoid the complexities of  
eligibility traces. One-step actor–critic methods replace the full return of REINFORCE with the 
one-step estimate of the return, and use a learned state value function as
the baseline.

The pseudo-code for the episododict algorithm is shown below. 
It is fully online, incremental algorithm, with states, 
actions, and rewards processed as they occur and then never 
revisited.

<img src="images/pub/OneStep_AC.png" alt="Grid" width="450"/>

The implementation is in `agents/OneStepAC`


## n-Step with Eligibility Traces Actor–Critic
We can replace the one-step target $G_t$ with $G_{t:t+n}$ or the lambda return
$G_t^{\lambda}$. Then using the eligibility traces we can incorportate these 
methods into the algorithm below. 

<img src="images/pub/AC_with_eligibility_traces.png" alt="Grid" width="450"/>


The implementation is in `agents/ACWithEligibilityTraces`



## Policy Gradient for Continuing Problems
For the continuing setting we need to redefine the objective function (
performance function) in terms of the average rate of reward per step. 
The pseudo-code of which is shown below:


<img src="images/pub/AC_with_eligibility_traces_continuing.png" alt="Grid" width="450"/>


The implementation is in `agents/ACWithEligibilityTracesContinuing`

## Experiments
In the following experiments, for each algorithm presented several
hyperparameters are shown. Some of the hyper-parameters cause simulation
failures, in which case they are not plotted. While the hyperparameters
are not perfectly fine-tuned, they should give a starting point in further
improvements for that particular algorithm. Also, note that some algorithms
perform better for certain evnironments than others (as it is to be expected).


### Environments 
The gymnasium environments used for experiments are shown in the tables, for
each algorithm. For the episodic environments 10 training seeds with 10 
evaluation seeds were used, while 10 training seeds with 5 for evaluation \
were used for continuing task.


#### Reward Shaping
For the Acrobot environment, this reward shaper was used to help the algorithm
learn (particularly helpful for the continuing case).


```python
def acrobot_reward_shaper(reward: float, state: np.ndarray, **kwargs) -> float:

    """
    A reward shaper for Acrobot that provides a dense reward based on height
    and penalizes excessive velocity to encourage smoother control.

    The state is: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), vel1, vel2]
    The height of the foot is: -cos(theta1) - cos(theta1 + theta2)
    """
   
    # If the original reward is 0 (or > -1), the goal has been reached. Return a large bonus.
    if reward > -1.0:
        return 10.0

    # The state vector components
    cos_theta1 = state[0]
    sin_theta1 = state[1]
    cos_theta2 = state[2]
    sin_theta2 = state[3]
    vel1 = state[4]
    vel2 = state[5]

    # Calculate the height of the foot using the angle sum identity for cosine
    height_of_foot = -cos_theta1 - (
                cos_theta1 * cos_theta2 - sin_theta1 * sin_theta2)

    # Penalty for high angular velocity  to encourage the agent to be 
    # more controlled and stable.
    velocity_penalty_weight = 0.001
    velocity_penalty = -velocity_penalty_weight * (vel1 ** 2 + vel2 ** 2)

    # The final reward is the height reward plus the stability penalty
    return float(height_of_foot + velocity_penalty)

```

5 training seeds and 10 evaluation seeds were used.


### Episodic Results


#### Performance Evaluation

The following table answer the question: how does each algorithm perform, wrt
1) pure returns
2) discounted returns, the algorithm's objective
3) episode length on hard, i.e. greedy evaluations


#### REINFORCE

| Environment | $\bar{R}_{0, h}$                                                                                                       | $\bar{G}_{0, h}$                                                                                          | Average Episode Length                                                                                                |
|-------------|------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------| 
| MountainCar | <img src="images/evaluation_metrics/reinforce/MountainCar_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/reinforce/MountainCar_mean_hard_eval_G0.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/reinforce/MountainCar_mean_hard_eval_episode_length.png" alt="Grid" width="400"/> |
| LunarLander | <img src="images/evaluation_metrics/reinforce/LunarLander_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/reinforce/LunarLander_mean_hard_eval_G0.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/reinforce/LunarLander_mean_hard_eval_episode_length.png" alt="Grid" width="400"/> |
| CartPole    | <img src="images/evaluation_metrics/reinforce/CartPole_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>    | <img src="images/evaluation_metrics/reinforce/CartPole_mean_hard_eval_G0.png" alt="Grid" width="400"/>    | <img src="images/evaluation_metrics/reinforce/CartPole_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>    |
| Acrobot     | <img src="images/evaluation_metrics/reinforce/Acrobot_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>     | <img src="images/evaluation_metrics/reinforce/Acrobot_mean_hard_eval_G0.png" alt="Grid" width="400"/>     | <img src="images/evaluation_metrics/reinforce/Acrobot_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>     |


#### REINFORCE - with Baseline

| Environment | $\bar{R}_{0, h}$                                                                                                               | $\bar{G}_{0, h}$                                                                                                  | Average Episode Length                                                                                                        |
|-------------|--------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------| 
| MountainCar | <img src="images/evaluation_metrics/reinforcebaseline/MountainCar_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/reinforcebaseline/MountainCar_mean_hard_eval_G0.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/reinforcebaseline/MountainCar_mean_hard_eval_episode_length.png" alt="Grid" width="400"/> |
| LunarLander | <img src="images/evaluation_metrics/reinforcebaseline/LunarLander_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/reinforcebaseline/LunarLander_mean_hard_eval_G0.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/reinforcebaseline/LunarLander_mean_hard_eval_episode_length.png" alt="Grid" width="400"/> |
| CartPole    | <img src="images/evaluation_metrics/reinforcebaseline/CartPole_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>    | <img src="images/evaluation_metrics/reinforcebaseline/CartPole_mean_hard_eval_G0.png" alt="Grid" width="400"/>    | <img src="images/evaluation_metrics/reinforcebaseline/CartPole_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>    |
| Acrobot     | <img src="images/evaluation_metrics/reinforcebaseline/Acrobot_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>     | <img src="images/evaluation_metrics/reinforcebaseline/Acrobot_mean_hard_eval_G0.png" alt="Grid" width="400"/>     | <img src="images/evaluation_metrics/reinforcebaseline/Acrobot_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>     |


#### ActorCritic - 1 step

| Environment | $\bar{R}_{0, h}$                                                                                                       | $\bar{G}_{0, h}$                                                                                          | Average Episode Length                                                                                                |
|-------------|------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------| 
| MountainCar | <img src="images/evaluation_metrics/onestepac/MountainCar_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/onestepac/MountainCar_mean_hard_eval_G0.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/onestepac/MountainCar_mean_hard_eval_episode_length.png" alt="Grid" width="400"/> |
| LunarLander | <img src="images/evaluation_metrics/onestepac/LunarLander_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/onestepac/LunarLander_mean_hard_eval_G0.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/onestepac/LunarLander_mean_hard_eval_episode_length.png" alt="Grid" width="400"/> |
| CartPole    | <img src="images/evaluation_metrics/onestepac/CartPole_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>    | <img src="images/evaluation_metrics/onestepac/CartPole_mean_hard_eval_G0.png" alt="Grid" width="400"/>    | <img src="images/evaluation_metrics/onestepac/CartPole_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>    |
| Acrobot     | <img src="images/evaluation_metrics/onestepac/Acrobot_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>     | <img src="images/evaluation_metrics/onestepac/Acrobot_mean_hard_eval_G0.png" alt="Grid" width="400"/>     | <img src="images/evaluation_metrics/onestepac/Acrobot_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>     |


#### ActorCritic with Eligibility Traces

| Environment | $\bar{R}_{0, h}$                                                                                                         | $\bar{G}_{0, h}$                                                                                            | Average Episode Length                                                                                                  |
|-------------|--------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------| 
| MountainCar | <img src="images/evaluation_metrics/ACEligTrace/MountainCar_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/ACEligTrace/MountainCar_mean_hard_eval_G0.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/ACEligTrace/MountainCar_mean_hard_eval_episode_length.png" alt="Grid" width="400"/> |
| LunarLander | <img src="images/evaluation_metrics/ACEligTrace/LunarLander_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/ACEligTrace/LunarLander_mean_hard_eval_G0.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/ACEligTrace/LunarLander_mean_hard_eval_episode_length.png" alt="Grid" width="400"/> |
| CartPole    | <img src="images/evaluation_metrics/ACEligTrace/CartPole_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>    | <img src="images/evaluation_metrics/ACEligTrace/CartPole_mean_hard_eval_G0.png" alt="Grid" width="400"/>    | <img src="images/evaluation_metrics/ACEligTrace/CartPole_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>    |
| Acrobot     | <img src="images/evaluation_metrics/ACEligTrace/Acrobot_mean_hard_eval_sum_raw_rewards.png" alt="Grid" width="400"/>     | <img src="images/evaluation_metrics/ACEligTrace/Acrobot_mean_hard_eval_G0.png" alt="Grid" width="400"/>     | <img src="images/evaluation_metrics/ACEligTrace/Acrobot_mean_hard_eval_episode_length.png" alt="Grid" width="400"/>     |


#### Learned Correctness
Correctness concerns itself with how well does the algorithm learn. 
In the following table the following questions are addressed:
* Is the agent learning ? This is measured by the loss specific to each algorithm
  * REINFORCE with Baseline: average $\delta$ (refer to pseudocode box)
  * AC: TD-error
* How does the expected initial state value compare to the actual discounted 
returns. Are we learning the objective function as expected. Note that here I am
looking at the _soft_ evaluation $G_{0, f}$ in order to measure the quality of 
_expectation_ of $V(s_0) \overset \cdot{=} \mathbb{E}[R_0 | s_0]$.
 _Note_ that here initial value $q_{init}$ indicates the starting levels of the 
expected values in $V$


#### REINFORCE

| Environment | $\bar{G}_{0, f}$ vs $\bar{V}(s_0)$                                                                 | Avg Loss                                                                                                   |
|-------------|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| MountainCar | <img src="images/learning/reinforce/MountainCar_value_accuracy_JOINT.png" alt="Grid" width="400"/> | <img src="images/training_metrics/reinforce/MountainCar_mean_behavioral_loss.png" alt="Grid" width="400"/> |
| LunarLander | <img src="images/learning/reinforce/LunarLander_value_accuracy_JOINT.png" alt="Grid" width="400"/> | <img src="images/training_metrics/reinforce/LunarLander_mean_behavioral_loss.png" alt="Grid" width="400"/> |
| CartPole    | <img src="images/learning/reinforce/CartPole_value_accuracy_JOINT.png" alt="Grid" width="400"/>    | <img src="images/training_metrics/reinforce/CartPole_mean_behavioral_loss.png" alt="Grid" width="400"/>    |
| Acrobot     | <img src="images/learning/reinforce/Acrobot_value_accuracy_JOINT.png" alt="Grid" width="400"/>     | <img src="images/training_metrics/reinforce/Acrobot_mean_behavioral_loss.png" alt="Grid" width="400"/>     |


#### REINFORCE with Baseline

| Environment | $\bar{G}_{0, f}$ vs $\bar{V}(s_0)$                                                                         | Avg Loss                                                                                                           |
|-------------|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| MountainCar | <img src="images/learning/reinforcebaseline/MountainCar_value_accuracy_JOINT.png" alt="Grid" width="400"/> | <img src="images/training_metrics/reinforcebaseline/MountainCar_mean_behavioral_loss.png" alt="Grid" width="400"/> |
| LunarLander | <img src="images/learning/reinforcebaseline/LunarLander_value_accuracy_JOINT.png" alt="Grid" width="400"/> | <img src="images/training_metrics/reinforcebaseline/LunarLander_mean_behavioral_loss.png" alt="Grid" width="400"/> |
| CartPole    | <img src="images/learning/reinforcebaseline/CartPole_value_accuracy_JOINT.png" alt="Grid" width="400"/>    | <img src="images/training_metrics/reinforcebaseline/CartPole_mean_behavioral_loss.png" alt="Grid" width="400"/>    |
| Acrobot     | <img src="images/learning/reinforcebaseline/Acrobot_value_accuracy_JOINT.png" alt="Grid" width="400"/>     | <img src="images/training_metrics/reinforcebaseline/Acrobot_mean_behavioral_loss.png" alt="Grid" width="400"/>     |


#### ActorCritic - 1 step

| Environment | $\bar{G}_{0, f}$ vs $\bar{V}(s_0)$                                                                 | Avg Loss                                                                                                   |
|-------------|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| MountainCar | <img src="images/learning/onestepac/MountainCar_value_accuracy_JOINT.png" alt="Grid" width="400"/> | <img src="images/training_metrics/onestepac/MountainCar_mean_behavioral_loss.png" alt="Grid" width="400"/> |
| LunarLander | <img src="images/learning/onestepac/LunarLander_value_accuracy_JOINT.png" alt="Grid" width="400"/> | <img src="images/training_metrics/onestepac/LunarLander_mean_behavioral_loss.png" alt="Grid" width="400"/> |
| CartPole    | <img src="images/learning/onestepac/CartPole_value_accuracy_JOINT.png" alt="Grid" width="400"/>    | <img src="images/training_metrics/onestepac/CartPole_mean_behavioral_loss.png" alt="Grid" width="400"/>    |
| Acrobot     | <img src="images/learning/onestepac/Acrobot_value_accuracy_JOINT.png" alt="Grid" width="400"/>     | <img src="images/training_metrics/onestepac/Acrobot_mean_behavioral_loss.png" alt="Grid" width="400"/>     |


#### ActorCritic with Eligibility Traces

| Environment | $\bar{G}_{0, f}$ vs $\bar{V}(s_0)$                                                                   | Avg Loss                                                                                                     |
|-------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| MountainCar | <img src="images/learning/ACEligTrace/MountainCar_value_accuracy_JOINT.png" alt="Grid" width="400"/> | <img src="images/training_metrics/ACEligTrace/MountainCar_mean_behavioral_loss.png" alt="Grid" width="400"/> |
| LunarLander | <img src="images/learning/ACEligTrace/LunarLander_value_accuracy_JOINT.png" alt="Grid" width="400"/> | <img src="images/training_metrics/ACEligTrace/LunarLander_mean_behavioral_loss.png" alt="Grid" width="400"/> |
| CartPole    | <img src="images/learning/ACEligTrace/CartPole_value_accuracy_JOINT.png" alt="Grid" width="400"/>    | <img src="images/training_metrics/ACEligTrace/CartPole_mean_behavioral_loss.png" alt="Grid" width="400"/>    |
| Acrobot     | <img src="images/learning/ACEligTrace/Acrobot_value_accuracy_JOINT.png" alt="Grid" width="400"/>     | <img src="images/training_metrics/ACEligTrace/Acrobot_mean_behavioral_loss.png" alt="Grid" width="400"/>     |


### Continuing Task Results

The concepts of hard and soft evaluation from the episodic case are carried
over for the continuing task. For the continuing task we are interested 
in maximizing the reward ratio. The reward the agent is trained on is the shaped
reward.

* Raw reward is the reward generated by the environment at time $t$: $r_t$
* Shaped reward is the modified reward to help the agent learn, at time $t$: $g_t = F(r_t, s_t)$
* Rate of raw reward: $R = \frac{1}{T}\sum_{t=1}^{T}r_t$, where $T \rightarrow \infty$
* Rate of shaped reward: $G = \frac{1}{T}\sum_{t=1}^{T}g_t$, where $T \rightarrow \infty$
* Average rate of raw rewards over N trials/seeds: $\bar{R} = \frac{1}{N}\sum_{n}^{N}R^{(n)}$
* Average rate of shaped rewards over N trials/seeds: $\bar{G} = \frac{1}{N}\sum_{n}^{N}G^{(n)}$
* Agent estimated rate of shaped rewards $G_{\pi}$
* Hard eval: $h$
* Soft eval: $f$


#### Performance Evaluation

Here we're interested in the final performance of the agent. While "raw" 
rewards are the user's metric, it is interesting to note how the agent
also performs with the shaped reward, as it is the reward the agent maximizes.


#### ActorCritic with Eligibility Traces

| Environment | $\bar{R}_h$                                                                                                                     | $\bar{G}_{h}$                                                                                                                       | 
|-------------|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Acrobot     | <img src="images/evaluation_metrics/ACwEligTraceContinuing/Acrobot_mean_hard_eval_mean_raw_reward.png" alt="Grid" width="400"/> | <img src="images/evaluation_metrics/ACwEligTraceContinuing/Acrobot_mean_hard_eval_mean_shaped_rewards.png" alt="Grid" width="400"/> |


#### Learned Correctness
Here the learned expected reward is compared against the actual shaped reward.

#### ActorCritic with Eligibility Traces

| Environment | $\bar{G}_h$ vs $\bar{V}_{\pi}$                                                                                                  | 
|-------------|---------------------------------------------------------------------------------------------------------------------------------|
| Acrobot     | <img src="images/evaluation_metrics/ACwEligTraceContinuing/Acrobot_mean_hard_eval_mean_raw_reward.png" alt="Grid" width="400"/> |



## Policy Parameterization for _Continuous Actions_

For continuous actions (i.e. infinite actions) we learn the statistics of the
distribution of said action. For example, the action set might be the real numbers, with actions chosen
from a normal (Gaussian) distribution. To produce a policy parameterization, 
the policy can be defined as the normal probability density over a 
real-valued scalar action, with mean and standard deviation given by 
parametric function approximators that depend on the state.

$$
\pi(a | s, \mathbf{\theta}) \overset \cdot{=} \frac{1}{\sigma(s, \mathbf{\theta})} \exp \Bigl(- \frac{\bigl(a - \mu(s, \mathbf{\theta}) \bigr)^2}{2\sigma(s, \mathbf{\theta})^2} \Bigr)
$$


where: $\pi: \mathcal{S} \times\mathbb{R} ^{d'} \rightarrow \mathbb{R}$ 
and $\sigma: \mathcal{S} \times\mathbb{R} ^{d'} \rightarrow \mathbb{R}^{+}$
are two _parametrized function approximators_. With these definitions, 
all the algorithms from discrete action, can be used to 
learn continuous action selection.

###### Experiments

The following algorithms were adopted for continuous action policies. The 
environment used here was the continuous action [MountainCar](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/)

Note that the continuous action environment has a different reward from
the discrete action environment. From the website:

"A negative reward of $-0.1 * action^2$ is received at each timestep to 
penalise for taking actions of large magnitude. If the mountain car reaches 
the goal then a positive reward of +100 is added to the negative 
reward for that timestep."

| Algorithms                             | Train                                                                                                            | 
|----------------------------------------|------------------------------------------------------------------------------------------------------------------|
| Episodic: REINFORCE                    | <img src="images/ReinforceContinuousAction_G0_train.png" alt="Grid" width="450"/>                                |
| Episodic: REINFORCE with Baseline      | <img src="images/ReinforceBaselineContinuousAction_G0_train.png" alt="Grid" width="450"/>                        |
| Episodic: AC with Eligibility Traces   | <img src="images/ACWithEligibilityTracesContinuousAction_G0_train.png" alt="Grid" width="450"/>                  |
| Continuing: AC with Eligibility Traces | <img src="images/ACWithEligibilityTracesContinuousActionContinuingTask_avg_R_train.png" alt="Grid" width="450"/> |

For the REINFORCE algorithms, the return was mean-centered, as it significantly
improved learning.