[Sutton & Barto RL Book]: http://incompleteideas.net/book/RLbook2020.pdf


# Multi-armed Bandits


## Table of Contents
- [Introduction](#introduction)
  - [Non-Associative Bandits](#non-associative-bandits)
  - [The $k$-armed Bandit Problem](#the-k-armed-bandit-problem)
- [Implementation: Action-value Functions](#implementation-action-value-functions)
- [Implementation: Policies](#implementation-policies)
- [Implementation: Run scripts](#implementation-run-scripts)
- [Experiments List](#experiments-list)
- [Bandit Algorithm](#bandit-algorithm)
- [10-armed Test Bed](#10-armed-testbed)


## Introduction
The most important feature distinguishing reinforcement learning from other 
types of learning is that it uses training information that _evaluates_ the 
actions taken rather than _instructs_ by giving correct actions.This is 
what creates the need for active exploration, for an explicit search for 
good behavior. Purely _evaluative_ feedback indicates how good the action 
taken was, but not whether it was the best or the worst action possible. 
Purely _instructive_ feedback, on the other hand, indicates the correct 
action to take, independently of the action actually taken. This kind of 
feedback is the basis of supervised learning. In their pure forms, these 
two kinds of feedback are quite distinct: evaluative feedback depends 
entirely on the action taken, whereas instructive feedback is independent of 
the action taken.

### Non-Associative Bandits
The non-associative setting, which does not involve learning to act in 
more than one situation, is the one in which most prior work involving 
evaluative feedback has been done, and it avoids much of the complexity 
of the full reinforcement learning problem. Studying this case enables us to 
see most clearly how evaluative feedback differs from, and yet can be 
combined with, instructive feedback.


### The $k$-armed Bandit Problem
Consider the following learning problem. You are faced repeatedly with a 
choice among $k$ different options, or _actions_. After each choice you 
receive a numerical _reward_ chosen from a __stationary probability 
distribution__ that depends on the action you selected. The objective is to 
maximize the expected total reward over some time period. This is the original 
form of the _$k$-armed bandit_ problem, so named by analogy to a slot
machine, or “_one_-armed bandit,” except that it has $k$ levers instead of one. 
Each action selection is like a play of one of the slot machine’s levers, 
and the rewards are the payoffs for hitting the jackpot. Through repeated 
action selections you are to maximize your winnings by concentrating your 
actions on the best levers. Today the term “bandit problem” is sometimes 
used for a generalization of the problem described above.


## Implementation: Action-Value Functions
- [x] QMonteCarlo: `nonassocative_value_functions.py/QMonteCarlo`
- [x] QCoefficientMovingAverage: `nonassocative_value_functions.py/QCoefficientMovingAverage`

## Implementation: Policies
- [x] $\varepsilon$-greedy: `nonassociative_policies.py/EpsGreedyPolicy`
- [x] UCB1: `nonassociative_policies.py/UCB1Policy`
- [x] Naiive Preference: `nonassociative_policies.py/NaiivePreferencePolicy`
- [x] Softmax Exploration: `nonassociative_policies.py/SoftmaxExplorationPolicy`
- [x] Bernoulli Greedy: `nonassociative_policies.py/BernoulliGreedy`
- [x] Bernoulli with Thompson Sampling: `nonassociative_policies.py/BernoulliThompsonSampling`

## Implementation: Run scripts
These are the scripts to generate experiment results
- [x] Action-value function methods: `eps_greedy_main.py`
- [x] UCB1 method: `ucb_main.py`
- [x] Additional methods, not in the book: `additional_algorithms_main.py`

## Details and explanations
Theoretical details, experiment explanations and other information
is located in this [notebook](summary.ipynb)


## Experiments List
* [Ex 1 - stationary environment, sample average action value estimation, $\varepsilon$-greedy](#experiment-1-stationary-testbed-sample-average-action-value-varepsilon-greedy-selection)
* [Ex 2 - stationary environment, constant step size for action value, $\varepsilon$-greedy selection](#experiment-2-stationary-testbed-constant-step-size-for-action-value-varepsilon-greedy-selection)
* [Ex 3 - stationary environment, exponential average (const step size) action value estimation, $\varepsilon$-greedy](#experiment-3-non-stationary-testbed-constant-step-size-for-action-value-varepsilon-greedy-selection)
* [Ex 4 - stationary environment, sample average action value estimation, $\varepsilon$-greedy](#experiment-4-non-stationary-testbed-sample-average-action-value-estimation-varepsilon-greedy-selection)
* [Ex 5 - stationary environment, comparing initial action values, sample average action value estimation, $\varepsilon$-greedy](#experiment-5-stationary-testbed-initial-values-comparison-sample-average-action-value-estimation-varepsilon-greedy-selection)
* [Ex 6 - stationary environment, $\varepsilon$-greedy vs. UCB1](#experiment-6-stationary-testbed-varepsilon-greedy-vs-ucb1)
* [Ex 7 - Stationary environment, gradient method - naiive preference, baseline evaluation](#experiment-7-stationary-testbed-gradient-method---naiive-preference-baseline-evaluation)
* [Ex 8 - Non-Stationary environment, Softmax Exploration](#experiment-8-non-stationary-testbed-softmax-exploration)
* [Ex 9 - Stationary environment, Bernoulli-Greedy](#experiment-9-stationary-environment-bernoulli-greedy)
* [Ex 10 - Stationary environment, Bernoulli Thompson Sampling](#experiment-10-stationary-environment-bernoulli-thompson-sampling)
* [Ex 11 - Non-Stationary Environment, Bernoulli Thompson Sampling](#experiment-11-non-stationary-environment-bernoulli-thompson-sampling)


## Bandit Algorithm
Pseudocode for a complete bandit algorithm using incrementally computed sample
averages and $\varepsilon$-greedy action selection is shown in the box below. 

<img src="images/simple_bandit_algorithm.png" alt="Grid" width="650"/>


## 10-armed Testbed
To roughly assess the relative effectiveness of the greedy and 
$\varepsilon$-greedy action-value methods, we compare them numerically on a 
suite of test problems. This is a set of $2000$ randomly generated $k$-armed 
bandit problems with $k = 10$. For each bandit problem, the action values, 
$q_{*}(a)$, where $a = 1, . . ., 10$, are selected according to a 
normal (Gaussian) distribution with mean $0$ and variance $1$. 

This means that for a bandit $j$ we have set its mean $\mu_j$ as 
$\mu_j = \mathbb{E}[R_t | A_t = a_j] \sim \mathcal{N}(0, 1)$. 

Then, during the simulation, when $a_j$ is selected, the observed reward is 
sampled as: $R_t \sim\mathcal{N}(\mu_j, 1)$.
 

<img src="images/10_armed_testbed.png" alt="Grid" width="650"/>

When the selection method selects action $A_t$ at time step $t$, the actual 
reward, $R_t$, is selected from a normal distribution with 
mean $q_{*}(A_t)$ and variance 1. This suite of test tasks is called
the _10-armed testbed_. For any learning method, we measure its performance
and behavior as it improves with experience over $1000$ time steps when applied 
to one of the bandit problems. This makes up one run. Repeating this for 
$2000$ independent runs, each with a different bandit problem, we obtained 
measures of the learning algorithm’s average behavior.

###### Task Extension
To extend and allow the testing of other learning algorithms, the test bed 
implemented supports two variants:

* _Stationary_: each bandit has fixed mean / std dev.
* _Non-stationary_: each bandit's mean follows a (normal) random walk

#### Experiment 1: Stationary Testbed, sample average action value,  $\varepsilon$-greedy selection.
The following implemented experiments relate to Figure 2.2 in the book. Here
action value is estimated using the following update rule:

$$
Q_{n+1} = Q_n + \frac{1}{n}\Bigl[R_n - Q_n \Bigr]
$$

which I have called this the monte-carlo estimate, 
$Q_{MC}(a)$, since it's an unbiased sample estimate of the action value. 
Action selection follows $\varepsilon$-greedy algorithm. Initial action value 
$Q_0 = 0$

Value function is implemented in `nonassociative_value_functions.py/QMonteCarlo`, 
while the incremental averaging method is implemented in 
`tools/moving_averages.py/CummulativeMovingAverage`.


| Average Rewards                                                     | Average Regret                                                      |
|---------------------------------------------------------------------|---------------------------------------------------------------------|
| <img src="images/rewards_experiment_1.png" alt="Grid" width="450"/> | <img src="images/regrets_experiment_1.png" alt="Grid" width="450"/> |

In this experiment, the sampled rewards from each bandit had a variance of 1. 
When we compare these graphs with those presented in the book we notice that 
the variance of the rewards is higher than those presented in the book. However, 
if we were to set the variance of the bandit to $0$, we obtain exactly the 
graphs of the book.

#### Experiment 2: Stationary Testbed, constant step size for action value, $\varepsilon$-greedy selection.
This experiment is set up the same as Experiment 1, with the exception that the 
update of the action value $Q$ is done using a fixed step $\alpha = 0.1$, with 
the following update value:

$$
Q_{n+1} = Q_n + \alpha\Bigl[R_n - Q_n \Bigr]
$$

Action selection follows $\varepsilon$-greedy algorithm. Initial action value 
$Q_0 = 0$


| Average Rewards                                                     | Average Regret                                                      |
|---------------------------------------------------------------------|---------------------------------------------------------------------|
| <img src="images/rewards_experiment_2.png" alt="Grid" width="450"/> | <img src="images/regrets_experiment_2.png" alt="Grid" width="450"/> |


Value function is implemented in `nonassociative_value_functions.py/QCoefficientMovingAverage`, 
while the incremental averaging method is implemented in 
`tools/moving_averages.py/ExponentialMovingAverage`, which uses
a fixed coefficient in this experiment.

Here we notice that purely greedy and slightly greedy perform almost equally, 
and all perform better than the more agressive exploratory strategy 
$\varepsilon = 0.3$

#### Experiment 3: Non-stationary Testbed, constant step size for action value, $\varepsilon$-greedy selection.
In this experiment the test bed is non-stationary. For each trial and each 
bandit, they are instantiated as normal random walk objects 
(`tools/random_walks.py/NormalRandomWalk`), each initialized with a random 
reward (mean) and randomness variance (random walk variance) of $0.1$, more
agressive than that mentioned in the book - refer 
to Exercise 2.5 - to make the non-stationarity and the results more obvious. 
Initial action values $Q_0(a) = 0$ for all actions,  and step size $\alpha = 0.1$


| Average Rewards                                                     | Average Regret                                                      |
|---------------------------------------------------------------------|---------------------------------------------------------------------|
| <img src="images/rewards_experiment_3.png" alt="Grid" width="450"/> | <img src="images/regrets_experiment_3.png" alt="Grid" width="450"/> |


#### Experiment 4: Non-stationary Testbed, sample-average action value estimation, $\varepsilon$-greedy selection.
Here I'm showing the difficulties of using sample-averages for estimating 
action values. As was shown below, under this approach, remote samples have as
much weight as more recent ones. Using the same test bed paramters as Excercise 
3, here are the plots.


| Average Rewards                                                     | Average Regret                                                      |
|---------------------------------------------------------------------|---------------------------------------------------------------------|
| <img src="images/rewards_experiment_4.png" alt="Grid" width="450"/> | <img src="images/regrets_experiment_4.png" alt="Grid" width="450"/> |


We can see that recency-weighted exponential averaging for action value 
estimation (exercise 3) is more efficient than sample-averaging (exercise 4).


#### Experiment 5: Stationary Testbed, initial values comparison, sample-average action value estimation, $\varepsilon$-greedy selection.

Here we're comparing the effects of optimism in the initial action value estimations.

| Average Rewards                                                     | Average Regret                                                      |
|---------------------------------------------------------------------|---------------------------------------------------------------------|
| <img src="images/rewards_experiment_5.png" alt="Grid" width="450"/> | <img src="images/regrets_experiment_5.png" alt="Grid" width="450"/> |

As we can see, optimistic initial estimations prompt for more exploration even 
when compared to $\varepsilon > 0$.


#### Experiment 6: Stationary Testbed, $\varepsilon$-greedy vs. UCB1.

| Average Rewards                                                     | Average Regret                                                      |
|---------------------------------------------------------------------|---------------------------------------------------------------------|
| <img src="images/rewards_experiment_6.png" alt="Grid" width="450"/> | <img src="images/regrets_experiment_6.png" alt="Grid" width="450"/> |


#### Experiment 7: Stationary Testbed, Gradient Method - Naiive Preference, Baseline Evaluation
In this experiment we're evaluating the gradient method over different learning 
steps and the use of baseline. Here the means of the bandits $q_*(a)$ were 
sampled around +4 (refer to Figure 2.5 in the book). 

| Average Rewards                                                     | Average Regret                                                      |
|---------------------------------------------------------------------|---------------------------------------------------------------------|
| <img src="images/rewards_experiment_7.png" alt="Grid" width="450"/> | <img src="images/regrets_experiment_7.png" alt="Grid" width="450"/> |

As we can see, using the baseline yields more efficient learning.


#### Experiment 8: Non-Stationary Testbed, Softmax Exploration
The estimated average used in the softmax function can be estimated
as a simple average (unbiased), or as a recency-weighted exponential average 
(refer to the action value section above). 


| Average Rewards                                                     | Average Regret                                                      |
|---------------------------------------------------------------------|---------------------------------------------------------------------|
| <img src="images/rewards_experiment_8.png" alt="Grid" width="450"/> | <img src="images/regrets_experiment_8.png" alt="Grid" width="450"/> |


As we saw in the action-value strategy for the non-stationary test bed, 
recency weighted averages deal with non-stationarity much better. Also, greedy 
strategy fairs worse than the exploratory strategy.


#### Experiment 9: Stationary Environment, Bernoulli-Greedy
In the following experiment the testbed is composed of $k=10$ bernoulli 
(density) bandits. Their success rate $\mu_{k}$ is sampled randomly from 
$\mu_k \sim \mathcal{U}[0.1, 0.9]$. The success rate remains stationary through
the simulation (i.e. non-stationary). Several combinations of initial 
$\alpha$s and $\beta$s were tried. 1000 steps and 2000 trials were run.


| Average Rewards                                                     | Average Regret                                                      |
|---------------------------------------------------------------------|---------------------------------------------------------------------|
| <img src="images/rewards_experiment_9.png" alt="Grid" width="450"/> | <img src="images/regrets_experiment_9.png" alt="Grid" width="450"/> |


#### Experiment 10: Stationary Environment, Bernoulli Thompson Sampling
Same setting as in experiment 9 for BernoulliTS algo.


| Average Rewards                                                      | Average Regret                                                       |
|----------------------------------------------------------------------|----------------------------------------------------------------------|
| <img src="images/rewards_experiment_10.png" alt="Grid" width="450"/> | <img src="images/regrets_experiment_10.png" alt="Grid" width="450"/> |


#### Experiment 11: Non-Stationary Environment, Bernoulli Thompson Sampling
In this experiment the true success rates of a bandit change over time 
(random walk with normal distribution) with a variance of 0.02. The rest of the
setup is the same as in experiment 9 


| Average Rewards                                                      | Average Regret                                                       |
|----------------------------------------------------------------------|----------------------------------------------------------------------|
| <img src="images/rewards_experiment_11.png" alt="Grid" width="450"/> | <img src="images/regrets_experiment_11.png" alt="Grid" width="450"/> |

As we can see here, Algorithm 2 doesn't deal too well with non-stationary environment
