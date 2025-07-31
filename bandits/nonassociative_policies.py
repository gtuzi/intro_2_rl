import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union

from tools.coefficients import Coefficient, ConstantCoefficient
from tools.moving_averages import SimpleMovingAverage, CummulativeMovingAverage
from nonassocative_value_functions import Q


def softmax(z: List[float], temp: float, scale: bool = False) -> List[float]:
    _exp = lambda v: np.clip(np.exp(v), a_min=1e-8, a_max=1e8)

    z = np.array(z)
    if scale:
        zd = np.abs(z.max() - z.min())
        z = z / zd if zd > 1 else z  # Scale to avoid overflows

    p = (_exp(z / (temp + 1e-8)) / _exp(z / (temp + 1e-8)).sum())

    eps = 1e-5
    if (p.sum() > (1 + eps)) or (p.sum() < (1 - eps)):
        print('WARNING: softmax unstable')
    if np.isnan(p).sum() != 0:
        raise ArithmeticError('Softmax error')
    return p.tolist()


def arg_select_with_random_tie_break(T: np.ndarray, val) -> int:
    a = np.random.choice(np.where(T == val)[0])  # Random break ties
    return int(a)


class Policy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, step: int, **kwargs) -> int:
        """ Perform action a~pi(n)"""
        raise NotImplementedError

    @abstractmethod
    def step(self, step: int, action: int, reward: float):
        """ Update all parameters, including value function """
        raise NotImplementedError


class UniformPolicy(Policy):
    def __init__(self, n_bandits: int, **kwargs):
        super(UniformPolicy, self).__init__(name='Uniform')
        self.action_set = [i for i in range(n_bandits)]
        self.n_bandits = n_bandits

    def __call__(self, step: int, **kwargs) -> int:
        return int(np.random.choice(self.action_set, size=1, p=[1. / self.n_bandits] * self.n_bandits))

    def step(self, step: int, action: int, reward: float):
        pass


class ActionValuePolicy(Policy, ABC):
    def __init__(self, q: Q, name: str, **kwargs):
        super(ActionValuePolicy, self).__init__(name=name)
        self._q = q

    @property
    def q(self):
        return self._q


class EpsGreedyPolicy(ActionValuePolicy):
    def __init__(self, q: Q, eps: Union[float, Coefficient], **kwargs):
        """
            Epsilon greedy policy
        :param q: action-value function
        :param eps: epsilon, exploration parameter
        """
        super(EpsGreedyPolicy, self).__init__(q=q, name='EpsGreedy', **kwargs)
        if isinstance(eps, Coefficient):
            assert eps.current_value >= 0.0, 'Epsilon cannot be negative'
            self.eps = eps
        else:
            assert eps >= 0.0, 'Epsilon cannot be negative'
            self.eps = ConstantCoefficient(eps)

    def __call__(self, step: int, **kwargs) -> int:
        """
            Perform the eps-greedy action: a~pi(n|eps).
            Note: states are NOT updated here
        :param step: simulation step
        :param kwargs: Not used
        :return:
        """

        if np.random.uniform(0.0, 1.0, size=1) >= self.eps.current_value:
            q_vals = self.q()  # All values
            anp = np.random.choice(np.where(np.array(q_vals) == np.array(q_vals).max())[0])  # Random break ties
            return int(anp)
        else:
            return int(np.random.choice(list(range(len(self.q))), size=1))

    def step(self, step: int, action: int, reward: float):
        """
            Update all parameters and value-function.
        :param step: simulation step
        :param action: action
        :param reward: reward of action
        :return: None
        """

        # print('\nstep: ', step,
        #       'a: ', str(action),
        #       'r: ', str(reward),
        #       'q: ', self.q(step=0, a=None),
        #       'err (tgt - q(a)): ', reward - self.q(step=0, a=action))

        _ = self.eps.step()
        self.q.step(step=step, action=action, reward=reward)


class UCB1Policy(ActionValuePolicy):
    def __init__(self, q: Q, c: Union[float, Coefficient] = ConstantCoefficient(0.1)):
        super(UCB1Policy, self).__init__(q=q, name='UCB')

        if isinstance(c, (float, int)):
            c = ConstantCoefficient(c)
        assert c.current_value > 0.0, 'Coefficient value cannot be <= 0.0'
        self.c = c

    def __call__(self, step: int, **kwargs) -> int:
        q = np.array(self.q())
        N = np.array(self.q.action_visit_counts)
        assert q.shape == N.shape
        # For N == 0, (N + 1e-8) would force the selection of non-visited arms
        T = q + self.c.current_value * np.sqrt(np.log(step + 1) / (N + 1e-8))
        return arg_select_with_random_tie_break(T, T.max())

    def step(self, step: int, action: int, reward: float):
        """
            Update all parameters and value-function.
        :param step: simulation step
        :param action: action
        :param reward: reward of action
        :return: None
        """
        _ = self.c.step()
        self.q.step(step=step, action=action, reward=reward)


class SoftmaxExplorationPolicy(ActionValuePolicy):
    """
        Softmax methods are based on Luce’s axiom of choice (1959)
        and pick each arm with a probability that is proportional to its average reward.
        Arms with greater empirical means are therefore picked with higher probability.
        Alternative name, Boltzman exploration

        Ref: https://arxiv.org/pdf/1402.6028.pdf
    """

    def __init__(self, q: Q, temperature: Union[float, Coefficient], **kwargs):
        super(SoftmaxExplorationPolicy, self).__init__(q=q, name='Softmax')

        if isinstance(temperature, (int, float)):
            temperature = ConstantCoefficient(temperature)
        self.temperature = temperature

    def __call__(self, step: int, **kwargs) -> int:
        p = np.array(softmax(z=self.q(), temp=self.temperature.current_value))
        anp = np.random.choice(np.where(p == p.max())[0])  # Random break ties
        return int(anp)

    def step(self, step: int, action: int, reward: float):
        self.temperature.step()
        self.q.step(step=step, action=action, reward=reward)


class NaiivePreferencePolicy(Policy):
    """
        Gradient based policy.

        Eq (2.12) in Sutton book, 2nd edition (2018):
        H(A, t+1) = H(A, t) + alpha * Advantage * (1 - pi(A))
        H(o, t+1) = H(o, t) - alpha * Advantage * p(o)

        where:
        H: preference model
        Advantage: R(t) - R_bar
    """

    def __init__(self,
                 n_actions: int,
                 preference_initial_value: float,
                 reward_initial_value: float,
                 learning_rate: Union[float, Coefficient],
                 temperature: Coefficient = ConstantCoefficient(1.0),
                 use_baseline: bool = True,
                 **kwargs):

        super(NaiivePreferencePolicy, self).__init__(name='NaiivePreference')
        if isinstance(learning_rate, (int, float)):
            learning_rate = ConstantCoefficient(learning_rate)

        self.action_set = [i for i in range(n_actions)]

        self.H = [
            SimpleMovingAverage(
                initial_value=preference_initial_value,
                coefficient=learning_rate)
            for _ in range(n_actions)
        ]

        self.R = CummulativeMovingAverage(initial_value=reward_initial_value)

        self.temperature = temperature
        self.use_baseline = use_baseline
        self.step_count = 0

    def __call__(self, step: int, **kwargs) -> int:
        """ Perform action a~pi(n)"""
        p = softmax([h.current_value for h in self.H], temp=self.temperature.current_value)
        return int(np.random.choice(self.action_set, p=p, size=1))

    def step(self, step: int, action: int, reward: float):
        """
            Update all parameters, including preference function
            H(A, t+1) = H(A, t) + alpha * Advantage * (1 - pi(A))
            H(o, t+1) = H(o, t) - alpha * Advantage * p(o)
             =
            H(*, t+1) = H(*, t) + alpha * Advantage * (Indicator(A) - pi(*))
        """
        self.step_count += 1
        advantage = (
                reward - self.R.current_value) \
            if self.use_baseline \
            else reward

        indicator = np.zeros(shape=(len(self.action_set),), dtype=np.float32)

        indicator[action] = 1.0

        p = softmax(
            [self.H[a].current_value for a in self.action_set],
            temp=self.temperature.current_value
        )

        # Note that here d[a_i] = -alpha * pi_i if a_i != A_t
        #                       =  alpha * (1 - p_i) if a_i = A_t
        d = advantage * (indicator - np.array(p))

        # Update all parameters: just adding d[a] takes care of the sign
        _ = [self.H[a].step(d[a]) for a in self.action_set]

        _ = self.temperature.step()

        if self.use_baseline:
            _ = self.R.step(reward)


class BetaDistribution:
    """
        Ref: https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
    """

    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def update_posterior(self, reward: float):
        """
            Update Beta function parameters according to
            (alpha, beta) += (reward, 1 - reward) (see ref, page 14)
        :param reward: in {0, 1}
        """
        self.alpha, self.beta = self.alpha + reward, self.beta + 1 - reward

    def sample(self, n_samples: int = 1) -> List[float]:
        '''
            Sample from Beta Distribution
        :param n_samples: number of samples
        :return: samples as list
        '''
        return np.random.beta(a=self.alpha, b=self.beta, size=(n_samples,)).tolist()

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class BernoulliPolicy(Policy):
    """
        Ref: https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
    """

    def __init__(
            self,
            n_actions: int,
            initial_alpha: float = 1.,
            initial_beta: float = 1.,
            name: str = 'bernoulli'):
        super(BernoulliPolicy, self).__init__(name)
        assert n_actions > 0, name
        self.distributions = [BetaDistribution(alpha=initial_alpha, beta=initial_beta) for _ in range(n_actions)]
        self.step_count = 0

    def step(self, step: int, action: int, reward: float):
        """ Update all parameters, including value function """
        self.step_count += 1
        eps = 1e-5
        assert ((-eps < reward < eps) or (1 - eps < reward < 1 + eps)), 'Invalid reward value.'
        self.distributions[action].update_posterior(reward)


class BernoulliThompsonSampling(BernoulliPolicy):
    """
        Algorithm 2 in reference

        Ref: https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
    """

    def __init__(
            self,
            n_actions: int,
            initial_alpha: float = 1.,
            initial_beta: float = 1.):
        super(BernoulliThompsonSampling, self).__init__(
            n_actions=n_actions,
            initial_alpha=initial_alpha,
            initial_beta=initial_beta,
            name='bernoulli_thompson_sampling')

    def __call__(self, step: int, **kwargs) -> int:
        """ Perform action a~pi(n)"""
        # We are sampling from the distribution of our estimate of what the "true" probability - in the environment -
        # of generating a successful reward.
        estimated_success_probabilities = [dist.sample(1) for dist in self.distributions]
        return int(np.argmax(estimated_success_probabilities))


class BernoulliGreedy(BernoulliPolicy):
    """
        Algorithm 1 in reference

        Ref: https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
    """

    def __init__(
            self,
            n_actions: int,
            initial_alpha: float = 1.,
            initial_beta: float = 1.):
        super(BernoulliGreedy, self).__init__(
            n_actions=n_actions,
            initial_alpha=initial_alpha,
            initial_beta=initial_beta,
            name='bernoulli_greedy')

    def __call__(self, step: int, **kwargs) -> int:
        """ Perform action a~pi(n)"""
        # These are our expectations (E[theta]) of what the bernoulli "true" probability - in the environment -
        # of generating a successful reward.
        estimated_success_probabilities = [dist.mean for dist in self.distributions]
        return int(np.argmax(estimated_success_probabilities))
