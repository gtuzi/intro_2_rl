[Sutton & Barto RL Book]: http://incompleteideas.net/book/RLbook2020.pdf


# Eligibility Traces

Eligibility traces (ET)s unify and generalize TD and Monte Carlo methods. When TD
methods are augmented with ETs, they produce a family of methods spanning
a spectrum that has Monte Carlo methods at one end ($ \lambda = 1$) and one-step TD methods
at the other ($\lambda = 0$). In between are intermediate methods that are often better than
either extreme method. ETs also provide a way of implementing Monte Carlo
methods online and on continuing problems without episodes. 
What ETs offer is a short-term memory vector mechanism, 
a memory vector, the _eligibility trace_ $\mathbf{z}_t \in \mathbb{R}^d$, 
that parallels the $n$-step TD method's long-term weight vector 
$\mathbf{w}_t \in \mathbb{R}^d$.

Some of the advantages of ETs are:
* Only a single trace vector is required rather than a store of the last $n$ feature vectors
* Learning also occurs continually and uniformly in time rather than being delayed and
then catching up at the end of the episode.
* Learning can occur and affect behavior immediately after a state is 
encountered rather than being delayed $n$ steps.

Monte-Carlo and TD methods evaluate a state's value on events that occur
following the state over multiple stepts. This is the _forward view_. With 
ETs it is possible to leverage the TD error and looking backward
to evaluate a state's value. This approach is called the _backward view_.

---

## The $\lambda$-return
The $n$-step __return__ is defined as:
$$
\begin{align*}
G_{t:t+n} &\overset{\cdot}{=} \sum_{i=t}^{t+n-1}\gamma^{i-t}R_{i+1} + \gamma^{n}\hat{v}(S_{t+n}, \mathbf{w}_{t+n-1}), \quad 0 \le t \le T - n 
\end{align*}
$$

where for $t \gt T$ the $n$-step return is just the sum of discounted rewards up
to the end of the episode at $t = T$. Here $\hat{v}(s, \mathbf{w})$ is the
approximate value of $s$ evaluated by function (approximator) $\hat{v}$ 
parametrized by $\mathbf{w}$.

Another valid target return for the update can be the average over 
returns from different $n$s. For example a target update can be
defined as the sum $\frac{1}{2} G_{t:t+2} + \frac{1}{2}G_{t:t+4}$. 
Any set of $n$-step returns - even infinite - can be averaged this way as 
long as their weights (coefficients) sum to $1$.

An update that averages simpler component updates is called 
a _compound update_. A compound update can only be done
when the longest of its component updates is complete. The TD($\lambda$) is
one particular way of combining all $n$-step updates, where each 
component is weighted proportionally to 
$\lambda^{n - 1}, \space \lambda \in [0, 1)$ and normalized 
by $1 - \lambda$ to ensure the sum of the weights add up to $1$.

<img src="images/td_lambda_backup_diagram.png" alt="Grid" width="550"/>

The resulting target return $G^{\lambda}$ is called the _$\lambda$-return_:

$$
\begin{align*}
G^{\lambda}_{t} \overset{\cdot}{=} (1 - \lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_{t:t+n}
\end{align*}
$$

In this formulation, the $1$-step is given the largest weight $1 - \lambda$, 
next the $2$-step with weight $(1 - \lambda)\lambda$, and so forth - the weight
decays with $\lambda$. 

For any $n \ge T$ all $n$-step return is the  conventional $G_t$. 
So for the remaining traces of same return, their cummulative weight 
is $\lambda^{T - t - 1}$. An alternative view of weight distribution 
is given below:

<img src="images/td_lambda_weights.png" alt="Grid" width="550"/>

So we can separate the terms after terminal state from the main summation 
of the $\lambda$-return as follows:

$$
\begin{align*}
G^{\lambda}_{t} = (1 - \lambda)\sum_{n=1}^{T - t - 1}\lambda^{n-1}G_{t:t+n} + \lambda^{T - t - 1} G_t
\end{align*}
$$

Here we note that when $\lambda = 1$, we get the conventional MC return 
$G_t^{\lambda=1} = G_t$, while for $\lambda = 0$ we get 
$G_t^{\lambda=0} = G_{t:t+1}$, the one-step TD($0$) return. The $\lambda$ 
parameter affects how far into the future the $\lambda$-return is determined by. 
The $\lambda$-return gives us an alternative way of moving smoothly between MC
and one-step TD methods, comparable with the $n$-step bootstrapping methods, 
where we vary the steps.


##### Exercise 12.1
Note the recursive relationship of the $n$-step return:
$$
\begin{align*}
G_{t:t+n} &\overset{\cdot}{=} \sum_{i=t}^{t+n-1}\gamma^{i-t}R_{i+1} + \gamma^{n}\hat{v}(S_{t+n}, \mathbf{w}_{t+n-1}) = R_{t+1} + \gamma G_{t+1:t+n},  \quad 0 \le t \le T - n
\end{align*}
$$

Replacing it into the $\lambda$-return we also get the recursive relationship as follows:

$$
\begin{align*}
G^{\lambda}_{t} &\overset{\cdot}{=} (1 - \lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_{t:t+n} \\
&= (1 - \lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\bigl(R_{t+1} + \gamma G_{t+1:t+n}\bigr) \\
&= (1 - \lambda)\sum_{n=1}^{\infty}\bigl[\lambda^{n-1} R_{t+1} + \lambda^{n-1}\gamma G_{t+1:t+n}\bigr] \\
&=(1 - \lambda)\Bigl[\sum_{n=1}^{\infty}\lambda^{n-1} R_{t+1} + \gamma \sum_{n=1}^{\infty}\lambda^{n-1}  G_{t+1:t+n} \Bigr] \\
&=(1 - \lambda)\sum_{n=1}^{\infty}\lambda^{n-1} R_{t+1} + \gamma \Bigl[(1 - \lambda) \sum_{n=1}^{\infty}\lambda^{n-1}  G_{t+1:t+n} \Bigr] \\
&= R_{t+1} + \gamma\Bigl[(1 - \lambda)\sum_{n=1}^{\infty}\lambda^{n-1}  G_{t+1:t+n} \Bigr] \tag*{(sum of infinite geometric series cancels out coefficient)} \\
&= R_{t+1} + \gamma G^{\lambda}_{t+1}
\end{align*}
$$


### Offline $\lambda$-return Algorithm

The _offline_ algorithms makes no changes to the weight vector during the 
episode. At the end of the episode, a sequence of offline updates are applied
according to the semi-gradient update rule, using the $\lambda$-return as 
the target

$$
\begin{align*}
\mathbf{w}_{t+1} = \mathbf{w}_{t} + \alpha\bigl[G^{\lambda}_t - \hat{v}(S_t, \mathbf{w}_t) \bigr]\nabla_{\mathbf{w}}\hat{v}(S_t, \mathbf{w}_t), \;\; t = 0, ..., T-1 
\end{align*}
$$

In this approach, for each state visited, we look forward in time to
all the future rewards and decide how best to combine them. This is the 
(theoretical) _forward_ view of the learning algorithm. 
After looking forward from and updating one particular state, 
we move on to the next state and do not
bother with the already "updated" states. Future states however, are 
repeatedly affected from the point of view of the previous states 
(recall that for approximate methods, an update to a state affects others, 
unlike the independence we get in the tabular methods where the update
of a state does not affect the values others).

---

## TD($\lambda$)
Ties the theoretical forward view with the backward view using ETs. 
It improves over the offline (forward) view in the following ways:

* updates the weight vector $\mathbf{w}$ in every step, rather at the end of the episode
* computations are distributed equally in time, as opposed to at the end
* can be applied to continuing rather than just episodic problems

#### Semigradient TD($\lambda$) with function approximation
In this setting, ET is a vector $\mathbf{z} \in \mathbb{R}^{d}$
with the same number of components as the weight vector $\mathbf{w}$. 
The weight vector is a long-term memory accumulating accross the episode,
whereas ETs are short-term memory lasting less than the episode, and assist
in the learning process. They affect the weight vector, which in turn affects
the state value.

In TD($\lambda$) ETs are initialized to zero at the beggining of the episode,
it is incremented by the gradient at each step, and it is decayed by
a $\gamma \lambda$ factor over time.

$$
\begin{align*}
\mathbf{z}_{-1} &\overset{\cdot}{=} \mathbf{0} \\
\mathbf{z}_{t} &= \gamma \lambda \mathbf{z}_{t-1} + \nabla_{\mathbf{w}} \hat{v}(S_t, \mathbf{w}_t), \;\; 0 \le t \le T   
\end{align*}
$$

$\lambda$ is the same parameters as in the $\lambda$-return and is also referred
to as _trace-decay_ parameter. The ET keeps track of which components of 
$\mathbf{w}$ have contributed positively or negatively to _recent_ state 
valuations. Recency in this context is defined in terms of $\gamma \lambda$.
In linear function approximation, the gradient of the value 
function $\hat{v}(S_t, \mathbf{w}_t)$ is simply the feature vector 
$\mathbf{x}_t$ - so the ET is the accumulation of past (decaying) features.

_Eligibility of the trace $\mathbf{z}$ denotes the eligibility of each component of $\mathbf{w}$
for undergoing (learning) changes if a reinforcing event happens._ The reinforcing
event under consideration are the momentary 1-step TD errors 
$\delta_t \overset{\cdot}{=} R_{t + 1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t)$.
The update rule is defined as: 
$$
\begin{align*}
\mathbf{w}_{t+1} \overset{\cdot}{=} \mathbf{w}_{t} + \alpha \delta_{t} \mathbf{z}_t
\end{align*}
$$

So here we have the update depending on the value of the current TD erro and
the accumulation of features of past events, i.e. _backward_ view. The 
prediction algorithm is given as:

<img src="images/td_lambda_prediction_algo.png" alt="Grid" width="550"/>

At each moment we look at the current TD error  and assign it backward to 
each prior state according to how much that state contributed  to the current 
eligibility trace at that time.


###### Intuition behind the backward view of TD($\lambda$)
