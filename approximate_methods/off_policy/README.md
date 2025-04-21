[Sutton & Barto RL Book]: http://incompleteideas.net/book/RLbook2020.pdf

# *WIP* Off-policy Methods with Approximation

_Notes on this README.md are, __for the moment__, being developed. The ongoing 
work is also copied onto [summary.ipynb](summary.ipynb) - which 
will be their final destination. Please
refer to this document for the complete rendering of the formulas._

## Introduction
The extension to function approximation is significantly different and 
harder for off-policy learning than it is for  on-policy learning. 
The tabular off-policy methods readily extend to semi-gradient algorithms, 
but these algorithms do not converge as robustly as  they do under on-policy 
training. In off-policy learning we try to learn a value function for a target policy
$\pi$, given data due to a different behavior policy $b$.

For the _prediction_ case, both policies are static and given, and we try
to learn/approximate the value functions $\hat{q} \approx q_{pi}$ 
or $\hat{v} \approx v_{\pi}$. For the _control_ case we learn action values 
$\hat{q}$ and both policies change, where the target policy $\pi$ being 
greedy wrt to $\hat{q}$ and behavioral $b$ is $\varepsilon$-soft wrt $\hat{q}$.

#### Challenges with Off-Policy
Two challenges present themselves in the off-policy with approximation
* _Target_ (value) of the update $\rightarrow$ dealt with importance sampling (IS)
* _Distribution_ of the updates $\rightarrow$ IS for semi-gradient methods, true gradients without IS

### Semi-Gradient Methods
The tabular IS methods can be extended for the off-policy semi-gradient case, 
in dealing with the learning target. These methods, however, do not address
the update distribution.

IS used in the tabular form, are adopted here by replacing the tabular 
update of $Q$ or $V$ to approximated method parametrized by $\mathbf{w}$.

The per-step IS ratio is defined as:

$$
    \rho_{t} \overset{\cdot}{=} \rho_{t:t} =\frac{\pi(A_t | S_t)}{b(A_t | S_t)}
$$

#### One-Step

The following is the development for the one-step case, both for prediction
($\hat{v}$) and control ($\hat{q}$)

###### Prediction 
TD-errors are:

* Episodic

$\delta_{t} \overset{\cdot}{=} R_{t+1} + \gamma\hat{v}(S_{t+1}, \mathbf{w}_{t}) - \hat{v}(S_{t}, \mathbf{w}_{t})$


* Continuing 

$\delta_{t} \overset{\cdot}{=} R_{t+1} - \bar{R}_t + \gamma\hat{v}(S_{t+1}, \mathbf{w}_{t}) - \hat{v}(S_{t}, \mathbf{w}_{t})$

###### Control
TD-errors are as follows for each control aglorithm:
* __Sarsa__
  * Episodic: 
$\delta_{t} \overset{\cdot}{=} R_{t+1} + \gamma\hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_{t}) - \hat{q}(S_{t}, A_{t}, \mathbf{w}_{t})$
  * Continuing: 
$\delta_{t} \overset{\cdot}{=} R_{t+1} - \bar{R}_t + \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_{t}) - \hat{q}(S_{t}, A_{t}, \mathbf{w}_{t})$

* __Expected Sarsa__
  * Episodic: 
  $\delta_{t} \overset{\cdot}{=} R_{t+1} + \gamma \sum_a \pi(a | S_{t+1}) \hat{q}(S_{t+1}, a, \mathbf{w}_{t}) - \hat{q}(S_{t}, A_{t}, \mathbf{w}_{t})$
  * Continuing: 
  $\delta_{t} \overset{\cdot}{=} R_{t+1} - \bar{R}_t + \sum_a \pi(a | S_{t+1}) \hat{q}(S_{t+1}, a, \mathbf{w}_{t}) - \hat{q}(S_{t}, A_{t}, \mathbf{w}_{t})$

* __Q-Learning__
  * Episodic:
  $\delta_{t} \overset{\cdot}{=} R_{t+1} + \gamma \max_a \hat{q}(S_{t+1}, a, \mathbf{w}_{t}) - \hat{q}(S_{t}, A_{t}, \mathbf{w}_{t})$
  * Continuing: 
  $\delta_{t} \overset{\cdot}{=} R_{t+1} - \bar{R}_t + \max_a \hat{q}(S_{t+1}, a, \mathbf{w}_{t}) - \hat{q}(S_{t}, A_{t}, \mathbf{w}_{t})$

###### Update
The update procedure is as follows:

* $\mathbf{w}_{t+1} = \mathbf{w}_{t} + \alpha \rho_{t}\delta_{t}\nabla_{\mathbf{w}}\hat{v}(S_{t}, \mathbf{w}_{t}) $ (prediction)
* $\mathbf{w}_{t+1} = \mathbf{w}_{t} + \alpha \delta_{t}\nabla_{\mathbf{w}}\hat{q}(S_{t}, A_{t}, \mathbf{w}_{t}) $ (control)

Note that for the _control_ case ($\hat{q}$) we __do not use__
IS ratio. This is because we're estimating the value for $(S_t, A_t)$ following 
$\pi$. This means, we're at $A_t$ already, and we care how the ratios of the subsequent
actions, i.e. $A_{t+1:t+n}$ are taken not how we got to $A_t$. So, for the 
one-step case, the control algorithm is the same as the on-policy.

Another way to look at it is when we consider the following identity 
$v(s) = \mathbb{E}_{\pi(\cdot|s)}[\mathbb{E}_{P(\cdot|s, a)}[r + \gamma v_{\pi}(s')]] = \sum_{a}\pi(a|s) \mathbb{E}_{P(\cdot|s, a)}[r + \gamma v_{\pi}(s')]$, 
therefore the appropriate IS adjustment is  needed for the sampled actions $a \sim b(\cdot|s)$, 
in order to obtain the sample-estimate of  the expectation $\mathbb{E}_{\pi(\cdot|s)}$. 

For the $q(s, a) = \mathbb{E}_{P(\cdot|s, a)}[r + \gamma v_{\pi}(s')]
= \mathbb{E}_{P(\cdot|s, a)}[r + \gamma  \mathbb{E}_{\pi(\cdot|s'), P}[q_{\pi}(s', a')]]$ we're 
already starting our analysis at $a$,  i.e. there is no $\mathbb{E}_{\pi(\cdot|s)}$ 
expectation that depends on the $a$ under consideration; and we're 
bootsrapping  $q(s', a')$, i.e. not estimating it.


#### Multi-Step
In the multi-step case the one-step algorithms are extended as follows:

###### Target

__Prediction__
* $G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1}R_{t+n} + \gamma^{n}\hat{v}(S_{t+n}, \mathbf{w}_{t+n - 1})$ (episodic)
* $G_{t:t+n} = R_{t+1} - \bar{R}_{t} + - \bar{R}_{t+1} + ... + R_{t+n} - \bar{R}_{t+n-1} + \hat{v}(S_{t+n}, \mathbf{w}_{t+n - 1})$ (continuing)

__Control__

_Sarsa_
* $G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1}R_{t+n} + \gamma^{n}\hat{q}(S_{t+n}, A_{t+n}, \mathbf{w}_{t+n - 1})$ (episodic)
* $G_{t:t+n} = R_{t+1} - \bar{R}_{t} + R_{t+2} - \bar{R}_{t+1} + ... + R_{t+n} - \bar{R}_{t+n-1} + \hat{q}(S_{t+n}, A_{t+n}, \mathbf{w}_{t+n - 1})$ (continuing)

_Expected Sarsa_
* $G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1}R_{t+n} + \gamma^{n}\sum_{a}\pi(a | S_{t+n})\hat{q}(S_{t+n}, a, \mathbf{w}_{t+n - 1})$ (episodic)
* $G_{t:t+n} = R_{t+1} - \bar{R}_{t} + R_{t+2} - \bar{R}_{t+1} + ... + R_{t+n} - \bar{R}_{t+n-1} + \sum_{a}\pi(a | S_{t+n}) \hat{q}(S_{t+n}, a, \mathbf{w}_{t+n - 1})$ (continuing)

_Q-Learning_
* $G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1}R_{t+n} + \gamma^{n}\max_{a}\hat{q}(S_{t+n}, a, \mathbf{w}_{t+n - 1})$ (episodic)
* $G_{t:t+n} = R_{t+1} - \bar{R}_{t} + R_{t+2} - \bar{R}_{t+1} + ... + R_{t+n} - \bar{R}_{t+n-1} + \max_{a}\hat{q}(S_{t+n}, a, \mathbf{w}_{t+n - 1})$ (continuing)


__Update__ 
The update rule for the $n$-Step are as follows:

* $\mathbf{w}_{t+1} = \mathbf{w}_{t} + \alpha (\prod_{k=\textbf{t}}^{t+n-1} \rho_{k})[G_{t:t+n} - \hat{v}(S_{t}, \mathbf{w}_{t+n-1})] \nabla_{\mathbf{w}}\hat{v}(S_{t}, \mathbf{w}_{t+n-1})$ (prediction)
* $\mathbf{w}_{t+1} = \mathbf{w}_{t} + \alpha (\prod_{k=\textbf{t+1}}^{t+n} \rho_{k}) [G_{t:t+n} - \hat{q}(S_{t}, A_{t}, \mathbf{w}_{t+n-1})] \nabla_{\mathbf{w}}\hat{q}(S_{t}, A_{t}, \mathbf{w}_{t+n-1})$ (control)

Here also, just like in the one-step setting, for action-value
we do not IS weigh the $R_t$, since we're estimating $q(s, a)$, but we do weigh
the subsequent rewards as their choice depends on the appropriate (IS adjusted)
probability $\pi$. This makes sense for the tabular case, where each $(S_t, A_t)$
update is independent of others. But for the function approximation approach
this assumption does not hold. So for $\hat{v}$ the IS weights are 
synchronized with the actions which generated sampled rewards, whereas 
for $\hat{q}$ they are shifted one step forward, i.e. synchronized 
the next rewards, i.e. $R_{t+1:t+n}$ and the last $\hat{q}(S_{t+n}, A_{t+n})$

Note that $\rho_k = 1$ for $k \ge T$ and $G_{t:t+n} = G_t$ for $t+n \ge T$

### Off-Policy Divergence
One issue with off-policy with function approximation is the update 
distribution divergence. An example of such divergence is Baird's example 
(refer to `bairds_main.py` for the implementation).

In this example, the _dashed_ line (actions) take the system to one of the upper 
states which land into any of the with equal probability. The _solid_ line
(the other action) takes the system to the seventh state. In this example
the behavioral policy $b$ takes _dashed_ action with $\frac{6}{7}$ probability
and _solid_ action with $\frac{1}{7}$ so that the next state distribution
under $b$ is uniform, i.e. $b(\cdot|S_{t+1}) \sim U$. 

The target $\pi$ policy takes only the _solid_ line action with probability 1. 
The reward is 0 on all transitions.

<img src="images/Bairds_counterexample.png" alt="Grid" width="400"/>

Using the semi-gradient update for offline learning as follows:

##### TD(0) - Sarsa
Here I am using the behavioral policy $b$ to generate the experiences.
On-Policy for the Sarsa case means that we force $\rho = 1$

* $\mathbf{w}_{t+1} = \mathbf{w}_{t} + \alpha \rho_{t}(R_{t+1} + \gamma\hat{v}(S_{t+1}, \mathbf{w}_{t}) - \hat{v}(S_{t}, \mathbf{w}_{t}))\nabla_{\mathbf{w}}\hat{v}(S_{t}, \mathbf{w}_{t}) $

| Off-Policy                                                                    | On-Policy                                                                    |
|-------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| <img src="images/results/Bairds_Sarsa_OffPolicy.png" alt="Grid" width="400"/> | <img src="images/results/Bairds_Sarsa_OnPolicy.png" alt="Grid" width="400"/> |


##### DP
For the DP case, we have access to the environment dynamics $P$. 
On-Policy for the DP case means that we use $\pi = b$ probabilities.
Also note that $P(r \ne 0, \cdot | \cdot) = 0$. 

* $\mathbf{w}_{k + 1} = \mathbf{w}_{k} + \frac{\alpha}{|\mathcal{S}|}\sum_{s}([\mathbb{E}_{S_{t+1} \sim P(\cdot | S_t = s, a \sim \pi(\cdot|S_t = s))}[R_{t} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_k) | S_t = s] - \hat{v}(S_t = s, \mathbf{w}_k)]\nabla_{\mathbf{w}}\hat{v}(S_t = s, \mathbf{w}_k))$

Went a little verbose here for clarity.

| Off-Policy                                                                 | On-Policy                                                                 |
|----------------------------------------------------------------------------|---------------------------------------------------------------------------|
| <img src="images/results/Bairds_DP_OffPolicy.png" alt="Grid" width="400"/> | <img src="images/results/Bairds_DP_OnPolicy.png" alt="Grid" width="400"/> |


As we can see above, for the off policy cases, weights diverge, whereas 
the on-policy there is a solution found.

##### TD(0) - Q-Learning
For Q-Learning I'm keeping a separate set of weight vectors for each actions
as $\mathbf w^{T}_{a}$. The (explicit) update rule is then as follows:

* $\mathbf{w}_{A_t, t+1} = \mathbf{w}_{A_t, t} + \alpha (R_{t+1} + \gamma \max_a \hat{q}(S_{t+1}, a, \mathbf{w}_{a, t}) - \hat{q}(S_{t}, A_t, \mathbf{w}_{A_t, t}))\nabla_{\mathbf{w}_{A_t}}\hat{q}(S_{t}, A_t, \mathbf{w}_{A_t, t}) $

We also note here that the weights diverge, for both actions


| <img src="images/results/Bairds_QLearning_SolidAction_Weight.png" alt="Grid" width="400"/> | <img src="images/results/Bairds_QLearning_DashAction_Weight.png" alt="Grid" width="400"/> |
|--------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|


### The Deadly Triad
Likelihood of divergence rises under these three conditions

* Function approximation: generalizing from a state space
* Bootstrapping: updates based on estimates, as opposed to fully relying on samples (eg MC)
* Off-Policy Training: Training on a distribution of transitions other than that produced
by the target policy.Sweeping through the state space and updating all states
uniformly, as in dynamic programming, does not respect the target policy and is
an example of on-policy training.


### Bellman Error
In addressing the stability of off-policy learning, we may want to pursue the 
robust guarantees of SGD. And to achieve this we may want to re-formulate
the objective function in such a way that instead of using the diverging 
semi-gradient approach above, we use an objective function which generates a 
full gradient. For this, one of the sought after approaches considered is 
the _Bellman error_.

The Bellman Equation value function is defined as follows 
$v_{\pi}(s) = \sum_{a}\pi(a | s)\sum_{s', r}P(s', r | s, a)[r - \gamma v_{\pi}(s')]$, for all $s \in \mathcal{S}$.

The only solution to the Bellman Equation is the true value function $v_{\pi}$.
Any approximation of it $\hat{v}_{\pi} = v_{\mathbf{w}, \pi}$ will yield an error, which is called
_Bellman Error_ (BE) $\bar{\delta}$ at state $s$ and is defined as:

$$
\begin{align*}
\bar{\delta}_{\mathbf{w}}(s) &\overset{\cdot}{=}(\sum_{a}\pi(a | s)\sum_{s', r}P(s', r | s, a)[r - \gamma v_{\mathbf{w}, \pi}(s')]) - v_{\mathbf{w}, \pi}(s) \\
&= \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\mathbf{w}, \pi}(S_{t+1}) - v_{\mathbf{w}, \pi}(S_{t}) | S_{t} = s, A_t \sim \pi] \\
&= \mathbb{E}_{\pi}[\delta_{t,  \mathbf{w}} | S_{t} = s, A_t \sim \pi]
\end{align*}
$$

where we see that __BE is the expected TD error under $\pi$__. The vector of 
BE's at _all states_ is the vector: $\bar{\delta}_{\mathbf{w}} \in \mathbb{R}^{|\mathcal{S}|}$
called the Bellman error vector. The norm of this vector, is the mean square BE

$$
  \overline{BE}(\mathbf{w}) = \lVert \bar{\delta}_{\mathbf{w}} \lVert^{2}_{\mu}
$$
where $\mu$ is the _stationary distribution_ of states under $\pi$ (c.f. (9.3)) - 
which denotes the fraction of time spent on a state over an entire episode.

Based on $(11.14)$ in the book:

$$
\begin{align*}
  \overline{BE}(\mathbf{w}) &= \sum_{s}\mu(s)[\bar{\delta}_{\mathbf{w}}(s)]^2 \\
  &= \mathbb{E}_{\mu}[[\bar{\delta}_{\mathbf{w}}(s)]^2]
\end{align*}
$$


It is not possible to reduce $\overline{BE}(\mathbf{w}) = 0$ where $v_{\pi} = v_{\mathbf{w}}$
but for linear approximation there is a $\mathbf{w}$ for which $\overline{BE}$
is minimized.

### Gradient Descend on Bellman Error
SGD is appealing for true gradient methods because of its robust convergence 
guarantees. 

##### TD Error as Objective
Temporal difference learning uses TD error

$\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_{t}, \mathbf{w}_t)$

A candidate objective function could be $\overline{TDE}(\mathbf{w})$ called
the _mean squared TD error_:

$$
  \begin{align*}
    \overline{TDE}(\mathbf{w}) &= \sum_{s}\mu(s)\mathbb{E}[\delta_{t}^2 | S_t = s, A_t \sim \pi] \\
    &= \sum_{s}\mu(s)\mathbb{E}[\rho_t \delta_{t}^2 | S_t = s, A_t \sim b] \\
    &= \mathbb{E}_{b}[\rho_t \delta_{t}^2]
  \end{align*}
$$

where for $A_t \sim b$, $\mu(s)$ is the on-policy state distribution under $b$ -
which is often defined to be the fraction of time spent on $s$. 
The last equation is of the form needed for SGD; it gives the objective as 
an expectation that can be sampled from experience

$$
\begin{align*}
\mathbf{w}_t &= \mathbf{w}_t - \frac{1}{2}\alpha \nabla_{\mathbf{w}} (\rho_t \delta_{t}^2) \\
             &= \mathbf{w}_t + \alpha \rho_t \delta_t(\nabla_{\mathbf{w}}\hat{v}(S_{t}, \mathbf{w}_t) - \gamma\nabla_{\mathbf{w}}\hat{v}(S_{t+1}, \mathbf{w}_t))
\end{align*}
$$

This is a "complete" gradient and therefore a true SGD - called the _naive 
residual-gradient_. The residual-gradient converges, but it doesn't converge
to a desirable place.

##### A-Split Example
Considering the 3-state episodic Markov Reward Process shown in the image.

<img src="images/A_Split_MRP.png" alt="Grid" width="150"/>


Considering $\gamma = 1$, and on-policy ($\rho=1$) the true state 
values of this MRP should be:
* $v(A) = \frac{1}{2}$
* $v(B) = 1$
* $v(C) = 0$

Using the naive residual-gradient approach the system learns 
* $\hat{v}(A) = \frac{1}{2}$ which is correct
* $\hat{v}(B) = \frac{3}{4}$
* $\hat{v}(C) = \frac{1}{4}$

Let's list the errors at each transition to understand the results.
Each entry in the table is a TD-error ($\delta_t$)

| $\hat{v}(A) \rightarrow \hat{v}(B)$             | $\hat{v}(A) \rightarrow \hat{v}(C)$              | $\hat{v}(B) \rightarrow Term$         | $\hat{v}(C) \rightarrow Term$    |
|-------------------------------------------------|--------------------------------------------------|---------------------------------------|----------------------------------|
| $(0 + \frac{3}{4}) - \frac{1}{2} = \frac{1}{4}$ | $(0 + \frac{1}{4}) - \frac{1}{2} = -\frac{1}{4}$ | $(1 + 0) - \frac{3}{4} = \frac{1}{4}$ | $0 - \frac{1}{4} = -\frac{1}{4}$ |

The average (i.e. sample expectation) TD-error $\overline{TDE} = \frac{1}{16}$. 

Let's compare $\overline{TDE}$ wrt the real values

| $v(A) \rightarrow v(B)$               | $v(A) \rightarrow v(C)$                   | $v(B) \rightarrow Term$ | $v(C) \rightarrow Term$ |
|---------------------------------------|-------------------------------------------|-------------------------|-------------------------|
| $(0 + 1) - \frac{1}{2} = \frac{1}{2}$ | $( 0 + 0) - \frac{1}{2} = -\frac{1}{2}$   | $1 - 1 = 0$             | $0 - 0 = 0$             |

The average TD-error $\overline{TDE} = \frac{1}{8}$. 

So the values found with the residual-gradients has a lower $\overline{TDE}$ than
the true values. Therefore, the true solution has higher expected error.

From this example we can conclude that $\overline{TDE}$ is not a desirable objective.

##### Bellman Error as Objective
A better objective woule be minimizing mean square of the Bellman Error 
$\overline{BE}$. In the A-split example above $\overline{BE} = 0$. Normally
we wouldn't typically expect $\overline{BE}$ to be exactly zero. 

Let's repeat the definition of the $\overline{BE}$ from above:
$$
\begin{align*}
\overline{BE} &= \mathbb{E}_{\mu}[[\bar{\delta}_{\mathbf{w}}(s)]^2] \\
              &= \mathbb{E}_{\mu}[\mathbb{E}_{\pi}[\delta_{t, \mathbf{w}}]^2]
\end{align*}
$$

The gradient update of the weights then becomes:

$$
\begin{align*}
\mathbf{w}_{t+1} & = \mathbf{w}_{t} - \frac{1}{2} \alpha \nabla_{\mathbf{w}} (\mathbb{E}_{\pi}[\delta_{t, \mathbf{w}}]^2) \\
  &= \mathbf{w}_{t} - \frac{1}{2} \alpha \nabla_{\mathbf{w}} (\mathbb{E}_{b}[\rho_t \delta_{t, \mathbf{w}}]^2) \\
  &= \mathbf{w}_{t} - \alpha \mathbb{E}_{b}[\rho_t \delta_{t, \mathbf{w}}] \nabla_{\mathbf{w}} (\mathbb{E}_{b}[\rho_t \delta_{t, \mathbf{w}}]) \\
  &= \mathbf{w}_{t} - \alpha \mathbb{E}_{b}[\rho_t (R_{t+1} + \gamma v(S_{t+1}, \mathbf{w}) - v(S_{t}, \mathbf{w}))]\mathbb{E}_{b}[\rho_t  \nabla_{\mathbf{w}}\delta_{t, \mathbf{w}}] \\
  &= \mathbf{w}_{t} + \alpha \mathbb{E}_{b}[\rho_t (R_{t+1} + \gamma v(S_{t+1}, \mathbf{w}_t) - v(S_{t}, \mathbf{w}_t))]\mathbb{E}_{b}[\rho_t  \nabla_{\mathbf{w}}v(S_{t}, \mathbf{w}_t) - \gamma \rho_t\nabla_{\mathbf{w}}v(S_{t+1}, \mathbf{w}_t)] \\
\end{align*}
$$

here, $v_{\pi}(S_t)$ is treated as an expectation wrt to $\pi$, 
so $\mathbb{E}_{b}[v_{\pi}] = v_{\pi}$

###### Discussion

Recall the off-policy $n$-step error $-$ let's call it $\delta_{t:t+n}$ $-$ for the prediction case is:

$\delta_{t:t+n} = (\prod_{k=t}^{t+n-1} \rho_{k})[G_{t:t+n} - \hat{v}(S_{t}, \mathbf{w}_{t+n-1})] \nabla_{\mathbf{w}}\hat{v}(S_{t}, \mathbf{w}_{t+n-1})$

so for the one-step TD error is expanded as follows:
$$
\begin{align*}
\delta_{t, \mathbf{w}} &= \mathbb{E}_{\pi}[R_{t+1} + \gamma v(S_{t+1}, \mathbf{w}) - v(S_{t}, \mathbf{w}) | S_t, A_t \sim \pi] \\
    &= \rho_t \mathbb{E}_{b}[R_{t+1} + \gamma v(S_{t+1}, \mathbf{w}) - v(S_{t}, \mathbf{w}) | S_t, A_t \sim b] \\
    &= \rho_t \mathbb{E}_{b}[R_{t+1} + \gamma v(S_{t+1}, \mathbf{w})| S_t, A_t \sim b] - \rho_t \mathbb{E}_{b}[v(S_{t}, \mathbf{w}) | S_t, A_t \sim b] \\
\end{align*}
$$

So for the off-policy TD error, the importance sampling weights are used
for both the target and the estimation. But this does not _exactly_ 
follow the $\overline{BE}$ udpate above, where it is assumed that 
$\mathbb{E}_{b}[v_{\pi}] = v_{\pi}$. Note the footnote [1] in Ch. 11.5 under 
the BE development, the authors briefly mention the discrepancy. 

The problem here is the determination of whether $S_t$ sample is generated 
from $b$ $-$ which corresponds to the original definition or not. 
We can make the case similar to the control case where we don't care how we 
got to take $A_t$ for $\hat{q}(S_t, A_t)$. The authors here  are not 
clarifying why this different treatment when developing $\overline{BE}$ above.

Let's continue with the procedure as presented in the book:
$$
\begin{align*}
  \mathbf{w}_{t+1} &= \mathbf{w}_{t} + \alpha \mathbb{E}_{b}[\rho_t (R_{t+1} + \gamma v(S_{t+1}, \mathbf{w}_t)) - v(S_{t}, \mathbf{w}_t)][\nabla_{\mathbf{w}}v(S_{t}, \mathbf{w}_t) - \gamma \mathbb{E}_{b}[\rho_t\nabla_{\mathbf{w}}v(S_{t+1}, \mathbf{w}_t)]]
\end{align*}
$$

This is called as _residual-gradient algorithm_. If we were to use only the 
obtained samples, this reduces to the naive residual-gradient above (with some
minor difference wrt the treatment of $\rho_t$). This is naive because
the expectation wrt $S_{t+1}$ is multiplied together. Since we're using 
the same sample $S_{t+1}$ we get a _biased estimate_ of the expectation. To
obtain an un-biased estimate of the expectation, we would need two independently
sampled $S_{t+1}$'s. But we only get one sample in the interaction with the 
environment. There are two ways to de-bias this expectation:

* If the environment is deterministic (i.e. $S_{t+1, k} = S_{t+1, j})$, the
estimate is un-biase
* Simulated environment, where we can sample $S_{t+1}$ twice starting from $S_{t}$

For these conditions, the algorithm is guaranteed to converge to a 
minumm of $\overline{BE}$. This would work for the linear and non-linear
approximations. In the linear approximation case, the solution is unique.
However, this is not feasible in real environments.

So in this section we showed that BE can be used to formulate an objective
function which can be used to generate full gradients. 
However, the algorithm converges to a minimum under the conditions listed above. 
Moreover, the authors outline 3 additional issues with this algorithm 
(for the two conditions given above):
* Slow convergence
* Despite strong convergence guarantees of $\overline{BE}$ the predicted values 
found can still be incorrect.
* $\overline{BE}$ objctive is actually not learnable - where the authors show 
that same data distributions can generate two different objective functions.
This means that the objective function cannot be learned.


### Gradient-TD Methods
In pursuing the off-line stability, "true" SGD methods for minimizing 
mean squared Projected Bellman Error (PBE), namely $\overline{PBE}$ 
are considered. Let's start with a few definitions.

##### Projected Bellman Error
As we showed above, Bellman Error (BE) vector is defined as:

$\bar{\delta}_{\mathbf{w}}(s) \overset{\cdot}{=}(\sum_{a}\pi(a | s)\sum_{s', r}P(s', r | s, a)[r - \gamma v_{\mathbf{w}, \pi}(s')]) - v_{\mathbf{w}, \pi}(s)$

Moreover, the _Bellman operator_ $B_{\pi}: \mathbb{R}^{|\mathcal{S}|} \rightarrow \mathbb{R}^{|\mathcal{S}|}$
is defined as follows:

$B_{\pi}v(s) \overset{\cdot}{=} \sum_{a}\pi(a | s)\sum_{s', r}P(s', r | s, a)[r - \gamma v(s')]$

So we can also view the error vector in terms of $B_{\pi}$ as: 
$\bar{\delta}_{\mathbf{w}} = B_{\pi}v_\mathbf{w} - v_\mathbf{w}$

Repeated application of $B_{\pi}$ on $v_{\pi}$ converges to the true value $v_{\pi}$,
the fixed point of the operator, i.e.: $v_{\pi} = B_{\pi}v_{\pi}$. Note that
$v_{pi}$ is not an estimate of the value function (DP setting).

Now, as the operator is applied to $v_{\pi}$, there are intermediate values of 
$v_{\pi}$ which approach the final value, i.e. the fixed point of $B$. 
When we use the approximation for the value function 
$\hat{v}_{\pi}(\cdot, \mathbf{w})$, the application of the operator generates
$\mathbf{w}$ representable intermediate values of the aproximate value function.

These intermediate values is the projected Bellman error vector $\Pi\bar{\delta}_{\mathbf{w}}$.
The ($\mu$ weighted) vector norm of this error is another measure of the error
in the approximation space which is called the mean square BE - $\overline{PBE}$
defined as:

$$
   \overline{PBE}(\mathbf{w}) = \lVert \Pi \bar{\delta}_\mathbf{w} \lVert^{2}_{\mu}
$$

For a linear function approximator, the projection operation is linear, which implies
that it can be represented as an $\lvert \mathcal{S} \rvert \times \lvert \mathcal{S} \rvert$ matrix
$$
\begin{align*}
\Pi \overset{\cdot}{=} \mathbf{X}(\mathbf{X}^{T}\mathbf{D}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{D}
\end{align*}
$$

* $\mathbf{D}$ is the $\lvert \mathcal{S} \rvert \times \lvert \mathcal{S} \rvert$ diagonal matrix with
entries $\mu(s)$ along the diagonal
* $\mathbf{X}$ is the  $\lvert \mathcal{S} \rvert \times d$ whose rows are the feature 
vectors $\mathbf{x}(s)^{T}$ of size $d$, one for each state $s$.
* If the inverse does not exist, the pseudo inverse is used.

With linear function approximation there always exists an approximate value function
within the $\mathbf{w}$ space where $\overline{PBE}(\mathbf{w}) = 0$. This is 
the TD fixed point, $\mathbf{w}_{TD}$. As we have shown so far, this point is 
not always stable under the semi-gradient off-policy approach.

##### Gradient Descend in the Bellman Error
Now we turn our attention to the stability of off-policy 
training where we minimize the $\overline{PBE}$ - i.e. use it as the 
objective function. The gradient of $\overline(PBE)$ (refer to the book
for the complete derivation):

$$
\begin{align*}
\nabla\overline{PBE}(\mathbf{w}) &= 2 (\nabla_{\mathbf{w}} [\mathbf{X}^{T}\mathbf{D}\bar{\delta}_\mathbf{w}]^{T})[\mathbf{X}^{T}\mathbf{D}\mathbf{X}]^{-1}[\mathbf{X}^{T}\mathbf{D}\bar{\delta}_\mathbf{w}]
\end{align*}
$$

To turn this into an SGD method, we have to sample something on every 
time step that has this quantity as its expected value. We have $\mu$ as 
the stationary distribution of states under the behavior policy, 
where $\mathbf{D}$ is the diagonal matrix whose diagonal entries are
$\mu(s)$ induced by the behavioral policy. 
The terms above can then be written as expectations under $\mu$.

* $\mathbf{X}^{T}\mathbf{D}\bar{\delta}_\mathbf{w} = \sum_{s}\mu(s)\mathbf{x}(s)\bar{\delta}_{\mathbf{w}}(s) = \mathbb{E}[\rho_t \delta_t \mathbf{x}_t]$ of shape $d$
* $\mathbf{X}^{T}\mathbf{D}\mathbf{X} = \sum_s \mu(s)\mathbf{x}(s) \mathbf{x}(s)^{T} = \mathbb{E}[\mathbf{x}_t \mathbf{x}^{T}_{t}]$ of shape $d \times d$

The gradient of the transpose of the last term:

$$
\begin{align*}
\nabla_{\mathbf{w}} \mathbb{E}[\rho_t \delta_t \mathbf{x}_t]^{T} &= \mathbb{E}[\rho_t \nabla_{\mathbf{w}}\delta^{T}_t \mathbf{x}^{T}_t] \\
&= \mathbb{E}[\rho_t \nabla_{\mathbf{w}}(R_{t+1} + \gamma \mathbf{w}^{T} \mathbf{x}_{t+1} - \mathbf{w}^{T} \mathbf{x}_{t}) \mathbf{x}^{T}_t] 
\quad\text{(using episodic  $\delta_t$)} \\
&= \mathbb{E}[\rho_t (\gamma \mathbf{x}_{t+1} - \mathbf{x}_t)\mathbf{x}^{T}_t], \quad\text{(of shape $d$)}
\end{align*}
$$

After final substitution we get:

$$
\nabla_{\mathbf{w}}\overline{PBE}(\mathbf{w}) = 2\mathbb{E}[\rho_t (\gamma \mathbf{x}_{t+1} - \mathbf{x}_t)\mathbf{x}^{T}_t] [\mathbb{E}[\mathbf{x}_t \mathbf{x}^{T}_{t}]]^{-1}\mathbb{E}[\rho_t \delta_t \mathbf{x}_t]
$$

So in this formulation the gradient depends on the expectations of next step 
for first and last term of the gradient. So we cannot sample these expectations
and multiply them as this would give us a biased estimate. 

One approach is to sample these terms independently and then multiply them 
together to obtain the unbiased estimate of the gradient. Naiively, this is 
costly. One alternative is to estimate - and store - two of the terms, 
while the third term is sampled.

##### Gradient-TD
Gradient-TD methods estimate  and store the product of the second two factors 
of $\nabla_{\mathbf{w}}\overline{PBE}(\mathbf{w})$ of sizes $d \times d$ and 
$d$ with the resulting $\mathbf{v}$ of size $d$, defined as:

$$
\mathbf{v} \approx [\mathbb{E}[\mathbf{x}_t \mathbf{x}^{T}_{t}]]^{-1}\mathbb{E}[\rho_t \delta_t \mathbf{x}_t]
$$

Re-writing the gradient, we get 
$\nabla_{\mathbf{w}}\overline{PBE}(\mathbf{w}) = 2\mathbb{E}[\rho_t (\gamma \mathbf{x}_{t+1} - \mathbf{x}_t)\mathbf{x}^{T}_t]\mathbf{v} $


##### Small but long diversion - Linear Least Squares Problem
For a problem of the form: $\mathbf{y} = X \mathbf{w}$, where 
$\mathbf{y} \in \mathbb{R}^{m}, X \in \mathbb{R}^{m \times n}$ we want to find
$\mathbf{w}^{*} \in \mathbb{R}^{n}$ s.t. $\mathbf{y} = X \mathbf{w}^{*}$. 

But typically this is not possible because this is an over determined system 
($m > n$, i.e. more equations / samples than unknowns). So, we look for
$\mathbf{w}^{*}$ which yields best $\hat{\mathbf{y}}$ which approximates
$\mathbf{y}$ in the least squares sense, i.e.
$\mathbf{w}^{*} = \arg \min_{\mathbf{w}}\lVert \mathbf{y} - \hat{\mathbf{y}} \rVert^{2}_{2} = \arg \min_{\mathbf{w}}\lVert \mathbf{y} - X \mathbf{w}\rVert^{2}_{2}$.

The solution to this equation, i.e. $\mathbf{w}^{*}$ minimizes the squared 
error between the target and estimations. 

To achieve this we set our objective function - the squared distance - as 
$$
\begin{align*}
J(\mathbf{w}) &= \lVert \mathbf{y} - X \mathbf{w}\rVert^{2} \\
&= ( \mathbf{y} - X \mathbf{w})^{T}( \mathbf{y} - X \mathbf{w})
\end{align*}
$$

The point $\mathbf{w}^{*}$ at which this distance is at a minimum is located
where the gradient of the objective function is zero 
(first-order optimality condition)

$$
\begin{align*}
& \nabla_{\mathbf{w}}J(\mathbf{w}) = \mathbf{0} \Rightarrow \\
& 2A^{T}(\mathbf{y} - A\mathbf{x}) = \mathbf{0} \Rightarrow \\
& A^{T}\mathbf{y} = A^{T}A\mathbf{w} \Rightarrow \\
& \mathbf{w}^{*} = (A^{T}A)^{-1}A^{T}\mathbf{y} 
\end{align*}
$$

An alternative to directly finding  $\mathbf{w}^{*}$ from the closed form solution
is to use (batch) gradient descend. Note that:

$$
\begin{align*}
& J(\mathbf{w}) = \frac{1}{N}\sum^{N}_{i=1}(y_i - \mathbf{x}^{T}_i \mathbf{w})^{2} \quad\text{(normalizing the norm by number of samples)} \Rightarrow \\
& \nabla_{\mathbf{w}}J(\mathbf{w}) = -\frac{2}{N}\sum^{N}_{i = 1} (y_i - \mathbf{x}^{T}_i \mathbf{w})\mathbf{x}_i
\end{align*}
$$

The update takes the form:

$\mathbf{w}_{k+1} = \mathbf{w}_{} - \alpha \nabla_{\mathbf{w}}J(\mathbf{w}_{k}) = \mathbf{w}_{} + \frac{2\alpha}{N}\sum^{N}_{i = 1} (y_i - \mathbf{x}^{T}_i \mathbf{w})\mathbf{x}_i$

But this direct approach has several drawbacks. The dataset can be too large,
computation of the inverses (closed-form) may not be stable or feasible, or - in our case -
we don't have access to all the samples and would like to find $\mathbf{w}^{*}$
in an online fashion. An iterative approach is to use SGD, where we update
one sample at a time.

Note that:

$$
\begin{align*}
& J(\mathbf{w}) = \lim_{N \rightarrow \infty} \frac{1}{N}\sum^{N}_{i=1}(y_i - \mathbf{x}^{T}_i \mathbf{w})^{2} = \mathbb{E}_{y, \mathbf{x} \sim (\mathbf{y}, X)}[(y - \mathbf{x}^{T} \mathbf{w})^{2}] \Rightarrow \\
& \nabla_{\mathbf{w}}J(\mathbf{w}) =  \mathbb{E}_{y, \mathbf{x} \sim (\mathbf{y}, X)}[\nabla_{\mathbf{w}}(y - \mathbf{x}^{T} \mathbf{w})^{2}] = \mathbb{E}_{y, \mathbf{x} \sim (\mathbf{y}, X)}[\nabla_{\mathbf{w}}\ell(\mathbf{w})] 
\end{align*}
$$

If we define objective function in terms of samples - _sample loss_ - $\ell_t(\mathbf{w}) = (y_t - \mathbf{x}^{T}_t \mathbf{w})^{2}$,
the batch update becomes:

$ \mathbf{w}_{k+1} = \mathbf{w}_{k} - \alpha \nabla_{\mathbf{w}}\mathbb{E}_{y, \mathbf{x} \sim (\mathbf{y}, X)}[\nabla_{\mathbf{w}}\ell_t(\mathbf{w}_k)]$

In the "online" version, called Least Mean Squares (LMS) algorithm, 
the weights are updated after each sample:
$$
\mathbf{w}_{t+1} = \mathbf{w}_{t} + \alpha \nabla_\mathbf{w} \ell_t(\mathbf{w}) = \mathbf{w}_{t} + \alpha(y_t - \mathbf{x}^{T}_t\mathbf{w}_t)\mathbf{x}_t
$$

##### Estimating the second term online
Let's move some terms around:

$$
\begin{align*}
& \mathbf{v} \approx [\mathbb{E}[\mathbf{x}_t \mathbf{x}^{T}_{t}]]^{-1}\mathbb{E}[\rho_t \delta_t \mathbf{x}_t] \Rightarrow\\
& \mathbb{E}[\mathbf{x}_t \mathbf{x}^{T}_{t}]\mathbf{v} = \mathbb{E}[\rho_t \delta_t \mathbf{x}_t] \Rightarrow \\
& \mathbb{E}[(\rho_t \delta_t - \mathbf{v}^{T} \mathbf{x}_{t}) \mathbf{x}_t] = \mathbf{0} \Rightarrow \\
& \mathbb{E}[\nabla_{\mathbf{v}}(\rho_t \delta_t - \mathbf{v}^{T} \mathbf{x}_{t})^2] = \mathbf{0}
\end{align*}
$$

The last form is the optimality condition for 
$\arg \min_{\mathbf{v}}\lVert \vec{\rho}\odot \vec{\delta}  - X \mathbf{v}\rVert^{2}_{2}$
where $\vec{\rho}\odot \vec{\delta}$ is the $d$-dimensional 
off-policy vector of TD-errors, and $\odot$ is the element-wise multiplication.

Following the discussion form the linear least squares problem, 
inside the expectation, we have the gradient of the sample loss 
$\ell_t(\mathbf{v}) = (\rho_t \delta_t - \mathbf{v}^{T} \mathbf{x}_{t})^2$.
And the online update for finding the minimizer $\mathbf{v}^*$, using the SGD is:

$$
\begin{align*}
\mathbf{v}_{t+1} &= \mathbf{v}_{t} + \beta(\rho_t \delta_t - \mathbf{w}^{T}_t \mathbf{x}_t)\mathbf{x}_t \\
&= \mathbf{v}_{t} + \beta \rho_t (\delta_t - \mathbf{w}^{T}_t \mathbf{x}_t)\mathbf{x}_t \quad\text{(augmented with IS ratio)}
\end{align*}
$$

##### GTD-X
Now that we can estimate online $\mathbf{v}$, we can turn our attention to the
gradient of PBE

###### GTD2

$$
\begin{align*}
\mathbf{w}_{t+1} &= \mathbf{w}_t - \frac{1}{2} \alpha \nabla_{\mathbf{w}}\overline{PBE} \\
&= \mathbf{w}_t - \alpha \mathbb{E}[\rho_t (\gamma \mathbf{x}_{t+1} - \mathbf{x}_t)\mathbf{x}^{T}_t] [\mathbb{E}[\mathbf{x}_t \mathbf{x}^{T}_{t}]]^{-1}\mathbb{E}[\rho_t \delta_t \mathbf{x}_t] \\
&\approx  \mathbf{w}_t + \alpha \mathbb{E}[\rho_t (\mathbf{x}_t - \gamma \mathbf{x}_{t+1})\mathbf{x}^{T}_t] \mathbf{v}_t \\
&= \mathbf{w}_t + \alpha \rho_t(\mathbf{x}_t - \gamma \mathbf{x}_{t+1})\mathbf{x}^{T}_t \mathbf{v}_t
\end{align*}
$$
 
This algorithm is called _GTD2_.

###### TDC
An improved alternative, called _GTD(0)_ incorportates a gradient 
correction as follows:

$$
\begin{align*}
\mathbf{w}_{t+1} &= \mathbf{w}_t + \alpha \mathbb{E}[\rho_t (\mathbf{x}_t - \gamma \mathbf{x}_{t+1})\mathbf{x}^{T}_t] [\mathbb{E}[\mathbf{x}_t \mathbf{x}^{T}_{t}]]^{-1}\mathbb{E}[\rho_t \delta_t \mathbf{x}_t] \\
&= \mathbf{w}_t + \alpha (\mathbb{E}[\mathbf{x}_t \mathbf{x}^{T}_t] - \gamma \rho_t \mathbb{E}[\mathbf{x}_{t+1} \mathbf{x}^{T}_t]) [\mathbb{E}[\mathbf{x}_t \mathbf{x}^{T}_{t}]]^{-1}\mathbb{E}[\rho_t \delta_t \mathbf{x}_t] \\
&= \mathbf{w}_t + \alpha (\mathbb{E}[\rho_t \delta_t \mathbf{x}_t] - \gamma \rho_t \mathbb{E}[\mathbf{x}_{t+1} \mathbf{x}^{T}_t] [\mathbb{E}[\mathbf{x}_t \mathbf{x}^{T}_{t}]]^{-1}\mathbb{E}[\rho_t \delta_t \mathbf{x}_t]) \quad\text{(distributed the last 2 terms)} \\
&\approx \mathbf{w}_t + \alpha (\mathbb{E}[\rho_t \delta_t \mathbf{x}_t] - \gamma \rho_t \mathbb{E}[\mathbf{x}_{t+1} \mathbf{x}^{T}_t] \mathbf{v}_t) \quad\text{because: $\mathbf{v} \approx [\mathbb{E}[\mathbf{x}_t \mathbf{x}^{T}_{t}]]^{-1}\mathbb{E}[\rho_t \delta_t \mathbf{x}_t]$} \\
&\approx \mathbf{w}_t + \alpha \rho_t (\delta_t \mathbf{x}_t - \gamma \mathbf{x}_{t+1}\mathbf{x}^{T}_{t}\mathbf{v}_t) \quad\text{sampling}
\end{align*}
$$

An alternative name for this algorithm is _TD(0) with gradient correction_ (TDC)

Note that both algorithms are both $O(d)$ complexity if 
$\mathbf{x}^{T}_t \mathbf{v}_t$ is computed first.