[Sutton & Barto RL Book]: http://incompleteideas.net/book/RLbook2020.pdf

# Multi-armed Bandits
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


## Non-Associative Bandits
The non-associative setting, which does not involve learning to act in 
more than one situation, is the one in which most prior work involving 
evaluative feedback has been done, and it avoids much of the complexity 
of the full reinforcement learning problem. Studying this case enables us to 
see most clearly how evaluative feedback differs from, and yet can be 
combined with, instructive feedback.


## The k-armed Bandit Problem
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

### Action Values
In our $k$-armed bandit problem, each of the $k$ actions has an expected or 
mean reward given that that action is selected; let us call this the 
_value_ of that action. We denote the action selected on time step $t$ as 
$A_t$, and the corresponding reward as $R_t$.

The (true) _value_, then, of an arbitrary action $a$, denoted $q_{*}(a)$, is the 
expected reward given that $a$ is selected:

$$
q_{*}(a) \overset \cdot{=} \mathbb{E}[R_t | A_t = a]
$$

If you knew the (true) value of each action, then it would be trivial to 
solve the $k$ -armed bandit problem: you would always select the action with 
the highest value. We assume that you _do not know_ the action values with 
certainty, although you may have _estimates_. We denote the estimated value 
of action $a$ at time step $t$ as $Q_{t}(a)$. We would like 
$Q_{t}(a)$ to be close to $q_{*}(a)$.


### Exploration vs. Exploitation
If you maintain estimates of the action values, then at any time step there is 
at least one action whose estimated value is greatest. We call these the 
_greedy_ actions. When you select one of these actions, we say that you are 
__exploiting__ your current knowledge of the values of the actions. If instead 
you select one of the nongreedy actions, then we say you are __exploring__, 
because this enables you to improve your estimate of the nongreedy action’s
value. Exploitation is the right thing to do to maximize the expected reward on the one
step, but exploration may produce the greater total reward in the long run.
Because it is not possible both to explore and to exploit with any single 
action selection, one often refers to the “conflict” between exploration 
and exploitation.


### Action - Value Methods
These are methods for estimating the values of actions and
for using the estimates to make action selection decisions, which are 
collectively call _action-value methods_. The __true__ value of an action is 
the _mean reward_ when that action is selected. One natural way to _estimate_ 
this is by averaging the rewards actually received:

$$
Q_{t}(a) \overset \cdot{=} \frac{\text{sum of rewards when $a$ taken prior to $t$}}{\text{number of times $a$ taken prior to $t$}} = \frac{\sum_{i=1}^{t-1}R_i \cdot \mathbb{1}_{A_i = a}}{\sum_{i=1}^{t-1} \mathbb{1}_{A_i = a}} 
$$

If the denominator is zero, we define $Q_t(a)$ as some default value, 
such as $0$. As the denominator goes to infinity, by the law of large 
numbers, $Q_t(a)$ converges to $q_{*}(a)$. This is called the 
_sample-average_ method for estimating action values because each estimate 
is an average of the sample of relevant rewards. 

### Action Selection 
The simplest action selection rule is to select one of the actions with the 
highest estimated value, that is, one of the _greedy_ actions.
If there is more than one greedy action, then a selection is made among 
them in some arbitrary way, perhaps randomly. Greedy action selection method is
writen as:

$$
A_t \overset \cdot{=} \text{arg max}_a Q_t(a)
$$

with ties broken arbitrarily. Greedy action selection always exploits current 
knowledge to maximize immediate reward; it spends no time at all sampling 
apparently inferior actions to see if they might really be better. 
A simple alternative is to behave greedily most of the time, but every once 
in a while, say with small probability $\varepsilon$, instead select randomly from among 
all the actions with equal probability, independently of the action-value 
estimates. These near-greedy action selection rules are called
_$\varepsilon$-greedy_ methods. In the limit as the number of steps increases,
every action will be sampled an infinite number of times, 
thus ensuring that all the $Q_t(a)$ converge to their respective $q_{*}(a)$.
This implies that the probability of selecting the optimal action 
converges to greater than $1 - \varepsilon$, that is, to near certainty. 
These are just asymptotic guarantees, however, and say little about the 
practical effectiveness of the methods. $\varepsilon$-greedy algorithm can be formulated as:

$$
A_t \leftarrow 
\begin{cases}
\text{arg max}_a Q_t(a) &\quad \text{with $p = 1 - \varepsilon$} \\
a \sim \mathcal{U} (\{a_0, a_1, ...,a_{|\mathcal{A}| - 1}\}) &\quad \text{with $p = \varepsilon$}
\end{cases}
$$

### Action Value
The naiive estimation of the action value $Q(a)$, is to take the sample 
average of the observed rewards. But instead of keep track of all the samples
and computing the average for each selection step - in our decision - we can 
approach it in an incremental fashion.

For a reward $R_i$ observed after taking action $A_{i} = a$, and $Q_n$ be $a$'s
value estimate after it has been selected $n-1$ times, we have:

$$
Q_n \overset \cdot{=} \frac{R_1 + R_2 + ... + R_{n-1}}{n - 1}
$$

This formulation requires that we keep a record of all the times $a$ has been 
selected, and at each step, we estimate $Q_n$. 

###### Incremental Approach
The incremental approach for computing $Q_n$ can be done as follows:

$$
\begin{align*}
Q_{n + 1} &= \frac{1}{n}\sum_{i = 1}^{n} R_i \\
&= \frac{1}{n}\Bigl(R_n + (n-1) \frac{1}{n-1} \sum_{i=1}^{n-1} R_i \Bigr) \\
&=\frac{1}{n}\Bigl(R_n + (n-1)Q_n  \Bigr) \\
&=\frac{1}{n}\Bigl(R_n + nQ_n - Q_n  \Bigr) \\
&= Q_n + \frac{1}{n}\Bigl[R_n - Q_n \Bigr]
\end{align*}
$$

The update rule:

$$
Q_{n+1} = Q_n + \frac{1}{n}\Bigl[R_n - Q_n \Bigr]
$$

is a specific form of a more general update rule:

$$
Q_{n+1} = Q_n + \alpha \Bigl[R_n - Q_n \Bigr]
$$

Where $Q$ can be any sample estimate (not just action value). This update 
form is used extensively in RL. $R_n - Q_n$ is the error in estimate, and it is
reduced by taking a step towards the target. $\alpha$ is called the _step size_.

In the step size in the incremental implementation above $\alpha = \frac{1}{n}$
changes with every action selection (even though in the naiive approach above
it's not action dependent). But we can generalize step size more broadly as 
$\alpha_t(a)$

### Bandit Algorithm
Pseudocode for a complete bandit algorithm using incrementally computed sample
averages and $\varepsilon$-greedy action selection is shown in the box below. 

<img src="images/simple_bandit_algorithm.png" alt="Grid" width="650"/>


## Regret
When measuring the performance of the algorithms, besides the sum of rewards
we've obtained, we are also interested in measuring how efficient the approach
is. One way to do this is via the _regret_ metric. This metric measures
the "true" rewards "left on the table", i.e. if we were to know the true value
of the action (in practice we can't, but in simulations we can), how much we 
gave up. Ideally as the selection strategy learns, the rewards we leave on the 
table approaches zero.

The random variable of regret $Z_t$ is defined  as the difference between the 
optimal reward $q^{*}(A_i = a^*)$ minus  the true value of the selected 
action random variable $A_i = a$, defined as 
$q^*(A_i = a) = \mathbb{E}[R_i | A_i = a]$:

$$
\begin{align*}
Z_t &= \sum_{i = 1}^{t}\mathbb{}(q^{*}(A_i = a^{*}) - q^*(A_i = a))
\end{align*}
$$

To get an estimate of the regret $\mathbb{E}[Z_t]$ for a given strategy, 
we compute the sample average $\bar{Z}_t$ induced by this strategy over $N$ trials:

$$
\begin{align*}
\bar{Z}_t &= \frac{1}{N}\sum_{k=1}^{N}Z_t^{(k)} \\
&= \frac{1}{N}\sum_{k=1}^{N} \sum_{i=1}^{t} \Bigr( q^*(a^{*(k)}) - q^*(A_i = a^{(k)})\Bigl)
\end{align*}
$$

Under this notation, for these experiments the optimal action is changing
for each trial, hence the definition of the optimal action for trial
$k$ is $a^{*(k)}$. 

In (non-stationary) cases, where the optimal action changes over time, 
we would have:

$$
\begin{align*}
\bar{Z}_t &= \frac{1}{N}\sum_{k=1}^{N}Z_t^{(k)} \\
&= \frac{1}{N}\sum_{k=1}^{N} \sum_{i=1}^{t} \Bigr( q^*(a_{i}^{*(k)}) - q^*(A_i = a^{(k)})\Bigl)
\end{align*}
$$

where $a_{i}^{*(k)}$ is the optimal action for trial $k$ at step $i$.
Of course, we have access to the true reward values for any 
arbitrary action via the known environment. 

In the experiments, I'm tracking the _per-step_ or _average regret_, defined
as $\frac{\mathbb{E}[Z_t]}{t}$, as an efficiency metric. 
Ideally, this metric should tend towards $0$ in the limit.


## 10-armed Testbed
To roughly assess the relative effectiveness of the greedy and 
$\varepsilon$-greedy action-value methods, we compare them numerically on a 
suite of test problems. This is a set of $2000$ randomly generated $k$-armed 
bandit problems with $k = 10$. For each bandit problem, the action values, 
$q_{*}(a)$, where $a = 1, . . ., 10$, are selected according to a 
normal (Gaussian) distribution with mean $0$ and variance $1$.

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

###### Experiment 1: Stationary Testbed, sample average action value,  $\varepsilon$-greedy selection.
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

###### Experiment 2: Stationary Testbed, constant step size for action value, $\varepsilon$-greedy selection.
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

### Non-Stationary Tasks
We often encounter reinforcement learning problems that are effectively
nonstationary. In such cases it makes sense to give more weight to recent rewards than
to long-past rewards. One of the most popular ways of doing this is to use a constant
step-size parameter.

I introduced the constant step size above, but let's dive into a bit more 
detail here. 

The update rule presented earlier is:

$$
Q_{n+1} \overset \cdot{=} Q_n + \alpha[R_n - Q_n] 
$$

for $\alpha \in (0, 1]$


$$
\begin{align*}
Q_{n+1} &\overset \cdot{=} Q_n + \alpha[R_n - Q_n] \\
&= \alpha R_n + (1 - \alpha)Q_n \\
&= \alpha R_n + (1 - \alpha)[\alpha R_{n-1} + (1 - \alpha)Q_{n - 1}] \\
&= \alpha R_n + (1 - \alpha)\alpha R_{n-1} + (1 - \alpha)^{2}Q_{n-1} \\
&= \alpha R_n + (1 - \alpha)\alpha R_{n-1} +  (1 - \alpha)^2\alpha R_{n-2} + ... + (1 - \alpha)^{n-1}\alpha R_{1} + (1 - \alpha)^{n}Q_{1} \\
&= (1 - \alpha)^n Q_1 + \sum_{i = 1}^{n}\alpha(1 - \alpha)^{n-i}R_{i}
\end{align*}
$$

This is called the _weighted average_ because the sum of the weights
$(1 - a)^n + \sum_{i = 1}^{n}\alpha (1 - \alpha)^{n - 1} = 1$. 

The weight $\alpha(1 - \alpha)^{n-i}$ given to $R_i$ depends on how many steps 
ago this reward was observed. Since $(1 - \alpha) < 1$, the weight given to 
the past rewards decreases - decays exponentially - over the steps. That is why
this average is also called _exponential recency-weighted average_. As we can
see, using a fixed step size recent rewards are weighted more, allowing for 
quicker adaptation to the non-stationary environment parameters.

