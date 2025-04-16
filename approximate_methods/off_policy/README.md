[Sutton & Barto RL Book]: http://incompleteideas.net/book/RLbook2020.pdf

# *Ongoing*: Off-policy Methods with Approximation

# --- Note ---
Notes on this README.md are, __for the moment__, being developed. The ongoing 
work is also copied onto [summary.ipynb](summary.ipynb) - which 
will be their final destination. Please
refer to this document for the complete rendering of the formulas.

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

###### TD(0) - Sarsa
Here I am using the behavioral policy $b$ to generate the experiences.
On-Policy for the Sarsa case means that we force $\rho = 1$

* $\mathbf{w}_{t+1} = \mathbf{w}_{t} + \alpha \rho_{t}(R_{t+1} + \gamma\hat{v}(S_{t+1}, \mathbf{w}_{t}) - \hat{v}(S_{t}, \mathbf{w}_{t}))\nabla_{\mathbf{w}}\hat{v}(S_{t}, \mathbf{w}_{t}) $

| Off-Policy                                                                    | On-Policy                                                                    |
|-------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| <img src="images/results/Bairds_Sarsa_OffPolicy.png" alt="Grid" width="400"/> | <img src="images/results/Bairds_Sarsa_OnPolicy.png" alt="Grid" width="400"/> |


###### DP
For the DP case, we have access to the environment dynamics $P$. 
On-Policy for the DP case means that we use $\pi = b$ probabilities.
Also note that $P(r \ne 0, \cdot | \cdot) = 0$. 

* $\mathbf{w}_{k + 1} = \mathbf{w}_{k} + \frac{\alpha}{|\mathcal{S}|}\sum_{s}([\mathbb{E}_{S_{t+1} \sim P(\cdot | S_t = s, a \sim \pi(\cdot|S_t = s))}[R_{t} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_k) | S_t = s] - \hat{v}(S_t = s, \mathbf{w}_k)]\nabla_{\mathbf{w}}\hat{v}(S_t = s, \mathbf{w}_k))$

Went a little verbose here for clarity.

| Off-Policy                                                                 | On-Policy                                                                 |
|----------------------------------------------------------------------------|---------------------------------------------------------------------------|
| <img src="images/results/Bairds_DP_OffPolicy.png" alt="Grid" width="400"/> | <img src="images/results/Bairds_DP_OnPolicy.png" alt="Grid" width="400"/> |


###### TD(0) - Q-Learning
* $\mathbf{w}_{t+1} = \mathbf{w}_{t} + \alpha \rho_{t}(R_{t+1} + \gamma\hat{v}(S_{t+1}, \mathbf{w}_{t}) - \hat{v}(S_{t}, \mathbf{w}_{t}))\nabla_{\mathbf{w}}\hat{v}(S_{t}, \mathbf{w}_{t}) $

TBD