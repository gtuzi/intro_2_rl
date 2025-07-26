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

#### Action Selection 
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
apparently inferior actions to see if they might really be better. A simple 
alternative is to behave greedily most of the time, but every once in a while,
say with small probability $\varepsilon$, instead select randomly from among 
all the actions with equal probability, independently of the action-value 
estimates. These near-greedy action selection rules are called
_$\varepsilon$-greedy_ methods. In the limit as the number of steps increases,
every action will be sampled an infinite number of times, 
thus ensuring that all the $Q_t(a)$ converge to their respective $q_{*}(a)$.
This implies that the probability of selecting the optimal action 
converges to greater than $1 - \varepsilon$, that is, to near certainty. 
These are just asymptotic guarantees, however, and say little about the 
practical effectiveness of the methods.


#### Experiment: 10-armed Testbed
To roughly assess the relative effectiveness of the greedy and 
$\varepsilon$-greedy action-value methods, we compare them numerically on a 
suite of test problems. This is a set of $2000$ randomly generated $k$-armed 
bandit problems with $k = 10$. For each bandit problem, the action values, 
$q_{*}(a)$, where $a = 1, . . . , 10$, are selected according to a 
normal (Gaussian) distribution with mean $0$ and variance $1$.

<img src="../bandits_wip/images/10_armed_testbed.png" alt="Grid" width="650"/>

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

###### Experiment 1: Stationary Testbed with $\varepsilon$-greedy selection.

The following implemented experiments relate to Figure 2.2 in the book. Here 
the regret is being plotted, where random variable of regret $Z_t$ is defined 
as the difference between the optimal reward $q^{*}(A_i = a^*)$ minus 
the true value of the selected action random variable $A_i = a$, defined as 
$q^*(A_i = a) = \mathbb{E}[R_i | A_i = a]$:

$$
\begin{align*}
Z_t &= \sum_{i = 1}^{t}\mathbb{}(q^{*}(A_i = a^{*}) - q^*(A_i = a))
\end{align*}
$$

To get the estimate of the regret $\mathbb{E}[Z_t]$ for a given strategy, 
we compute the sample average $\bar{Z}_t$ induced by this strategy over $N$ trials:

$$
\begin{align*}
\bar{Z}_t &= \frac{1}{N}\sum_{k=1}^{N}Z_t^{(k)} \\
&= \frac{1}{N}\sum_{k=1}^{N} \sum_{i=1}^{t} \Bigr( q^*(a^{*(k)}) - q^*(A_i = a^{(k)})\Bigl)
\end{align*}
$$

Note that for these experiments, the optimal action is changing
for each trial, hence the definition of the optimal action for trial
$k$ is $a^{*(k)}$.

For these experiments I am reporting the _per-step_ or _average regret_, defined
as $\frac{\mathbb{E}[Z_t]}{t}$. Ideally, this metric should tend towards $0$ 
in the limit.



