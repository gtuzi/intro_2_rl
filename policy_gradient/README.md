[Sutton & Barto RL Book]: http://incompleteideas.net/book/RLbook2020.pdf


# Policy Gradient Methods

In policy gradient (PG) methods, the policy does not consult 
the action values in its decision of action. Here, a parametrized policy is 
learned which selects the action without consulting the estimated action value.
The value function may still be learned, with the aim of learning the policy
(parameters), but it is not specifically consulted in order to take the 
action.

## Notations
The following notations will be used [Sutton & Barto RL Book]

$$
\begin{align*}
\mathbf{\theta} \in \mathbb{R} ^{d^{'}} &\quad \text{policy parameter vector} \\[0.5em] 
\pi(a | s, \mathbf{\theta}) = \text{Pr}\{A_t = a | S_t, \mathbf{\theta}_t = \mathbf{\theta} \} &\quad \text{action selection probability at time $t$ given state $s$ and parameter $\mathbf{\theta}$} \\[0.5em] 
\mathbf{w} \in \mathbb{R}^{d} &\quad \text{value function ($\hat{v}(\cdot, \mathbf{w})$ or $\hat{q}(\cdot, \cdot, \mathbf{w})$) weight vector (or parameters), if the method uses it} \\[0.5em] 
J(\mathbf{\theta}) \in \mathbb{R} &\quad \text{scalar performance measure w.r.t the policy parameters}
\end{align*}
$$

## Policy Gradient Methods Overview
In PG methods, learning the policy parameter is based on the  gradient of the 
scalar performance measure $J(\mathbf{\theta})$, wherein these methods aim to
_maximize_ its value, via gradient _ascent_ of $J$:

$$
\mathbf{\theta}_{t+1} = \mathbf{\theta}_{t} + \alpha \widehat{\nabla_{\mathbf{\theta}_t} J(\mathbf{\theta}_t)} 
$$

where 
$\widehat{\nabla_{\mathbf{\theta}_t} J(\mathbf{\theta}_t)} \in \mathbb{R}^{d ^{'}}$ 
is a stochastic _estimate_ whose _expectation_ approximates the _gradient_ 
of the performance measure $J$ with respect to the policy parameters $\mathbf{\theta}$.

All PG methods follow this general schema - independent of whether they learn 
a state/action value function or not. Methods which do learn value functions
are usually called _actor-critic_, where actor refers to the policy and the 
critic the state or (most often) action value function.

## Policy Approximation
In PG methods, the policy can be parametrized in any way, as long as for 
$\pi(a | s, \mathbf{\theta})$ there is a gradient wrt its parameters; i.e. 
as long as $\nabla_{\mathbf{\theta}}\pi(a | s, \mathbf{\theta})$ exists and is
finite for all $s \in \mathcal{S}$ and $a \in \mathcal{A}$. Typically 
the policy never becomes deterministic - $\pi(\cdot) \in \{0, 1\}$, 
in order to ensure exploration.

### Discrete Action Space
If the action space is discrete and not too large, one way to parametrize 
the policy is via numerical preferences $h(s, a, \mathbf{\theta}) \in \mathbb{R}$.

Actions with the highest preference - in each state - are given higher 
probabilities of being selected (remember, we typically do not use 
deterministic policies). One way to achieve this is via the softmax function:

$$
\pi(a' | s, \mathbf{\theta} ) = \frac{e^{h(s, a', \mathbf{\theta})}}{\sum_{a \in \mathcal{A}}e^{h(s, a, \mathbf{\theta})}}
$$

This is called _soft-max in action preferences_. Action preferences (function) can 
be parameterized in an arbitrary way; for example ANNs or linear features 
$h(s, a, \mathbf{\theta}) = \mathbf{\theta}^{T} \mathbf{x}(s, a)$.

###### Approaching determinism
One advantage of parametrizing policies in soft-max in action preferences is that
the policy can approach deterministic policy, whereas with $\varepsilon$-greedy
there's always a chance of selecting a random action with $\varepsilon$ 
probability.

###### Couldn't we select an action according to the softmax on the action values ?
This approach wouldn't allow the policy to approach determinisim. Action-value 
estimates would converge to their corresponding true values, which would 
differ by a finite amount, translating to specific probabilities other 
than 0 and 1. Action preferences are driven to produce the optimal stochastic
policy (via performance measure maximization). If the optimal policy is 
deterministic, then the preferences of the optimal actions will be driven 
infinitely higher than all suboptimal actions, as allowed by the parameters 
(the probability distribution becomes "peaky" over the course of optimization).

###### Arbitrary probabilities
Another advantage of parametrizing policies is that it allows 
the selection of actions with arbitrary probabilities. In certain problems the
best policy is a "soft" policy. For example, in a card game, you may want the 
ability to bluff, due to the partial information. Action-value methods have 
no natural way of finding stochastic optimal policies, whereas policy 
approximating methods can.

###### Flexibility in complexity
One more advantage is advantage that policy parameterization may have 
over action-value parameterization is that the policy may be a simpler 
function to approximate. Problems vary in the complexity of their policies 
and action-value functions. For some, the action-value function is simpler 
and thus easier to approximate. For others, the policy is simpler. 
In the latter case a policy-based method will typically learn faster and 
yield a superior asymptotic policy.

###### Leveraging prior knowledge
The choice of policy parameterization is sometimes a good way
of injecting prior knowledge about the desired form of the policy into 
the reinforcement learning system. This is often the most important reason 
for using a policy-based learning method. For example, one can leverage
"inductive bias" from the problem at hand in formulating their policy.


## Policy Gradient Theorem
With continuous policy parameterization the action probabilities change 
smoothly as a function of the learned parameter. The continuity of the policy dependence on the parameters that enables
policy-gradient methods to approximate gradient ascent. For the episodic
task, the performance measure to be optimized (maximized) is defined
as the value of the start state of the episode:

$$
J(\mathbf{\theta}) \overset \cdot{=} v_{\pi_{\mathbf{\theta}}}(s_0)
$$

where $\pi_{\mathbf{\theta}}$ is the true value function of $\pi_{\mathbf{\theta}}$
parametrized by $\mathbf{\theta}$. In this discussion $\gamma = 1$, i.e. 
the undiscounted episode. With function approximation it may seem challenging to change the policy parameter
in a way that ensures improvement:
* Performance depends on both action selection and distribution of states where those actions are taken.
* Action selection depend on the parameters of the policy.
* The effect of the policy on the state distribution is a function of the environment - think the underlying MDP - and it is (typically) unknown.

_How can we estimate the performance gradient with respect to the policy 
parameter when the gradient depends on the unknown effect of policy changes 
on the state distribution?_

Policy gradient theorem, provides an analytic expression for the gradient of
performance with respect to the policy parameter - and it does not involve
the derivative of the state distribution.

But before we focus on the theorem, we should explain and define its building
blocks (from previous chapters in the book and otherwise).


We have the following identities:
$$
\begin{align*}
q_{\pi}(s, a) &= \sum_{s', r} p(s', r | s, a)(r + v(s')) \\[0.5em] 
v_{\pi} &= \sum_a \pi(a | s) q_{\pi}(s, a) \\[0.5em] 
p(s' | s, a) &= \sum_r p(s', r | s, a) \\[0.5em]
\end{align*}
$$

Also, the initial state distribution is defined as:
$$
h(s) = \text{Pr{$S_0 = s$}}
$$

The average number of time *steps* spent in state $s$, in a single episode, 
is denoted as $\eta(s)$. it includes both if the episode starts in
$s$, and the transitions that are made into $s$ from a preceding 
state $\bar{s}$.

$$
\eta(s) = \text{(1-step)} h(s) + \gamma \sum_{\bar{s}} \eta(\bar{s}) \sum_a p(s| \bar{s}, a) \pi(a | \bar{s}) =  h(s) + \gamma \sum_{\bar{s}} \eta(\bar{s})p(s | \bar{s})
$$

_Quick aside_: The discount term $\gamma$ can be thought of as the probability
that the next step is __not__ terminal. Consequentially, $(1 - \gamma)$ is the 
probability that the next step is terminal. So in the definition above, 
$\gamma$ down–weights the “future‐step visits” term as if the episode might 
end with probability $1 - \gamma$

The on-policy distribution ($\mu$) - known as the stationary distribution for 
the continuing case - defined as:

$$
\mu(s) = \frac{\eta(s)}{\sum_{s'}\eta(s')}
$$

means the fraction of time spent in each state, normalized.

Moreover, for a discrete Markov chain, as referenced [here](https://en.wikipedia.org/wiki/Discrete-time_Markov_chain)
in the $n$-step transition section, the probability of going from state i 
to state j in n time steps is:

$$
p_{ij}^{(n)} = \text{Pr}\{ X_{n} = j | X_0 = i \}
$$

which in the book is defined as the probability of going from $i$ to $j$, in 
$n$ steps in the underlying MDP under policy $\pi$, and is defined as: 

$$
\text{Pr}\{ i \rightarrow j, n, \pi \} = \text{Pr}_{\pi}\{X_{n} = j | X_{0} = i \}
$$

$n$-step distribution satisfy the Chapman-Kolmogorov equation:

$$
p^{(n)}_{ij} = \sum_r p_{ir}^{(k)}p_{rj}^{n-k} = \sum_{r^{(1)}, r^{(2)}, ... r^{(n-1)}} p_{ir^{(1)}}^{(1)}p_{r^{(1)}r^{(2)}}^{(1)} ... p_{r^{(n-2)}r^{(n-1)}}^{(1)}*p_{r^{(n-1)}j}^{(1)}
$$

Moreover, using an indicator function $I_{k}^{x} = \mathbf{1}\{S_k = x\}$
we can also define the expected count as:

$$
\mathbb{E}[\sum_{k=0} ^ {\infty}I_{k}^{x}] = \sum_{k=0} ^ {\infty} \mathbb{E}[I_{k}^{x}] = \sum_{k=0} ^ {\infty} \text{Pr}\{S_k = x \}
$$

So the expected state visitation count $\eta(s)$ above can also be expressed as:

$$
\eta(s) = \sum_{k=0}^{\infty}\text{Pr}\{s' \rightarrow s, k, \pi \} = \mathbb{E}[\text{num} \{t: S_t = s \}]
$$


So now we derive the gradient of the value function 
($\nabla \overset \cdot{=} \nabla_{\mathbf{\theta}}$ below):



$$
\begin{align*}
\nabla v_{\pi}(s) &= \nabla \Bigl(\sum_a \pi(a|s) q_{\pi}(s, a) \Bigr) \\[0.5em]
&=\sum_a \bigl(\nabla \pi(a | s) q_{\pi}(s, a) + \pi(a | s)\nabla q_{\pi}(s, a)\bigr) \\[0.5em]
\end{align*}
$$


$$
\begin{align*}
&= \sum_a \Bigl(\nabla \pi(a | s) q_{\pi}(s, a) + \pi(a | s)\nabla \bigl(\sum_{s', r} p(s', r | s, a)(r + v(s')) \bigr)\Bigr) \\[0.5em]
&= \sum_a \Bigl(\nabla \pi(a | s) q_{\pi}(s, a) + \pi(a|s) \bigl(\sum_{s'} p(s' | s, a) \nabla v(s') \bigr) \Bigr) \\[0.5em]
\end{align*}
$$


$$
\begin{align*}
&= \sum_a \nabla \pi(a | s) q_{\pi}(s, a) + \sum_a \sum_{s'}\pi(a | s)p(s'|s, a) \nabla v(s') \\[0.5em]
&= \sum_a \nabla \pi(a | s) q_{\pi}(s, a) + \sum_{s'} p(s' | s) \nabla v(s') \\[0.5em]
\end{align*}
$$


$$
\begin{align*}
&= \sum_a \nabla \pi(a | s) q_{\pi}(s, a) + \sum_{s'} p(s' | s) \Bigl\{\sum_{a'} \nabla \pi(a' | s') q(s', a') + \sum_{s''} p(s'' | s') \bigl[\sum_{a''}\nabla \pi(a'' | s'') \bigr] q(s'', a'') + \sum_{s'''} p(s''' | s'')[ ...] \Bigr\} \\[0.5em]
&= \sum_a \nabla \pi(a | s) q_{\pi}(s, a) + \sum_{s'} p(s' | s)\sum_{a'} \nabla \pi(a' | s') q(s', a') +  \sum_{s'} p(s' | s) \sum_{s''}p(s'' | s') \sum_{a''} \nabla \pi(a'' | s'')q(s'', a'') + \sum_{s'} p(s' | s) \sum_{s''}p(s'' | s') \sum_{s'''}p(s''' | s'')\sum_{a'''}\nabla \pi(a''' | s''')q(s''', a''') + ... \\[0.5em]
\end{align*}
$$


$$
\begin{align*}
&= \sum_a \nabla \pi(a | s) q_{\pi}(s, a) + \sum_{a'}\sum_{s'}p(s' | s) \nabla \pi(a' | s')q(s', a') + \sum_{a''} \sum_{s', s''}p(s'|s)p(s''| s') \nabla \pi(a'' | s'') q(s'', a'') + \sum_{a'''} \sum_{s', s'', s'''}p(s'|s)p(s'' | s)p(s''' | s'') \nabla \pi(a''' | s''')q(s''', a''') + ... \\[0.5em]
&= \sum_a \sum_{s}p(s|s) \nabla \pi(a | s) q_{\pi}(s, a) + \sum_{a'}\sum_{s'} p(s'|s) \nabla \pi(a' | s')q(s', a') + \sum_{a''} \sum_{s''}p(s'' | s)\nabla \pi(a'' | s'')q(s'', a'') + \sum_{a'''}\sum_{s'''}p(s''' | s) \nabla \pi(a''' | s''')q(s''', a''') + ... \quad \text{by Chapman-Kolmogorov equation}  \\[0.5em]
\end{align*}
$$


$$
\begin{align*}
&= \sum_a \sum_{k = 0}^{\infty} p(s^{(k)} | s) \nabla\pi(a | s^{(k)})q_{\pi}(s^{(k)}, a)  \quad \text{since $\sum_{a^{(k)}}\pi(a^{(k)} | s^{(k)})$} = \sum_a \pi(a | s^{(k)})\\[0.5em]
&= \sum_a \sum_{k = 0}^{\infty} p_{ss^{(k)}} ^ {(k)}\nabla\pi(a | s^{(k)})q_{\pi}(s^{(k)}, a) \quad \text{using the standard n-step notation}
\end{align*}
$$
