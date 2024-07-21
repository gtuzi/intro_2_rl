from typing import Dict, Union, Any, List, Tuple
import numpy as np
import scipy


class NoiseSchedule:
    def initialize(self):
        """ Initialization of exploration to starting point """
        pass

    def reset(self):
        """
            State reset to starting point (eg, mu). In random walks
            this resets the mu to initial value.
            Exploration doesn't change  (e.g. sigma/eps)
        """
        pass

    def step(self, *args, **kwargs):
        pass

    @property
    def value(self):
        raise NotImplementedError


class ConstantSchedule(NoiseSchedule):
    def __init__(self, val):
        self.val = val

    @property
    def value(self):
        return self.val


class LinearEpsSchedule(NoiseSchedule):
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1

        self.inc = (end - start) / float(steps)

        self.start = start
        self.current = start
        self.end = end

        if end > start:
            self.bound = min
        else:
            self.bound = max

    def initialize(self):
        self.current = self.start

    def step(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)

    def reset(self):
        pass

    @property
    def value(self):
        return self.current


class PDFSampler:
    def __init__(
            self,
            distribution: Union[
                scipy.stats.rv_continuous, scipy.stats.rv_discrete
            ],
            params: Dict
    ):
        """
            Initialize the PDFSampler with a given distribution
            from scipy.stats and its parameters.
        """
        self.distribution = distribution
        self.params = params
        self.is_discrete = hasattr(distribution, 'pmf')

    def sample(self, size: int = 1) -> Union[Any, List[Any]]:
        """
        Generate a sample from the distribution.
        """
        addl = {}
        if 'size' not in self.params:
            addl.update({'size': size})
        sample = self.distribution.rvs(**{**self.params, **addl})

        if size == 1:
            return sample.tolist()[0]
        else:
            return sample.tolist()

    def probability(self, x: Union[Any, List[Any]]) -> Union[Any, List[Any]]:
        """
            Return the probability (or probability density for
            continuous distributions) of the given sample.
        """
        if self.is_discrete:
            p = self.distribution.pmf(x, **self.params)
        else:
            p = self.distribution.pdf(x, **self.params)

        if isinstance(x, List):
            return p.tolist()
        return p


class Experience:
    def __init__(self, **kwargs):
        self.s = None
        self.a = None
        self.p = None
        self.r = None
        self.sp = None
        self.ap = None
        self.pp = None
        self.done = None
        self.t = None
        self.sigma = None
        self.sigmap = None
        self.rho = None
        self.rhop = None

        for k, v in kwargs.items():
            if k == 's':  # state 0
                self.s = v
            elif k == 'a':  # action
                self.a = v
            elif k == 'p':  # action probability
                self.p = v
            elif k == 'r':  # reward
                self.r = v
            elif k == 'sp':  # next state (s')
                self.sp = v
            elif k == 'ap':  # next action ~ pi(s')
                self.ap = v
            elif k == 'pp':  # next action ~ pi(s')'s probability
                self.pp = v
            elif k == 'done':
                self.done = v
            elif k == 't':
                self.t = v
            elif k == 'sigma':
                self.sigma = v
            elif k == 'sigmap':
                self.sigmap = v
            elif k == 'rho':
                self.rho = v
            elif k == 'rhop':
                self.rhop = v


class SoftPolicy:
    def get_greedy_action(self, s) -> Tuple[int, float]:
        raise NotImplementedError

    def get_sa_probability(self, s, a) -> float:
        raise NotImplementedError


class DiscreteActionAgent:
    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int
    ):
        self.obs_space_dims = obs_space_dims
        self.action_space_dims = action_space_dims

    def act(self, s) -> Tuple[int, float]:
        """ Return the action and probability """
        raise NotImplementedError

    def initialize(self):
        pass

    def reset(self):
        pass

    def step(self, *args, **kwargs):
        """ Learn: Qk+1 = somefunction(Qk) """
        pass


class QEpsGreedyAgent(DiscreteActionAgent, SoftPolicy):

    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            discount: float,
            eps: Union[float, NoiseSchedule] = 0.01
    ):
        DiscreteActionAgent.__init__(self, obs_space_dims, action_space_dims)
        SoftPolicy.__init__(self)
        self.discount = discount
        self.eps = eps
        self.Q = None
        self.Q_update_count = None

    def get_greedy_action(self, s) -> Tuple[int, float]:
        """
            Get the greedy action and it's conditional prob.
            If a single action: probability is 1.
            If multiple actions compete for being picked, they are randomly
            tie-broken. This mean's that their probability is 1/|argmax_a|
        """
        max_vals = np.amax(self.Q[s])
        idc = np.argwhere(self.Q[s] == max_vals).squeeze().tolist()
        if isinstance(idc, list):
            # Random tie-breaking
            return int(np.random.choice(idc)), 1. / len(idc)
        else:
            assert isinstance(idc, int)
            return idc, 1.

    def get_sa_probability(self, s, a) -> float:
        if isinstance(self.eps, NoiseSchedule):
            eps = self.eps.value
        else:
            eps = self.eps

        # Check if action is greedy for this state
        max_vals = np.amax(self.Q[s])
        idc = np.argwhere(self.Q[s] == max_vals).squeeze().tolist()
        if isinstance(idc, list) and (a in idc):
            return (1. - eps + eps / self.action_space_dims) * (1. / len(idc))
        elif isinstance(idc, int) and (a == idc):
            assert isinstance(idc, int)
            return 1. - eps + eps / self.action_space_dims
        # Action is not greedy.
        else:
            return eps / self.action_space_dims

    def get_sa_update_count(self, s, a) -> int:
        if self.Q_update_count is not None:
            return self.Q_update_count[s][a]

    def act(self, s) -> Tuple[int, float]:
        """
            eps-greedy policy
        :param s: state
        :return: (action, probability of action)
        """
        if isinstance(self.eps, NoiseSchedule):
            eps = self.eps.value
        else:
            eps = self.eps

        # We need to know which one is the greedy action and/or it's prob
        a_greedy, pcond_greedy = self.get_greedy_action(s)
        p_greedy = 1. - eps + eps / self.action_space_dims  # If 1 argmax_action
        # If there are > 1 argmax_a's this is scaled according to its
        # conditional uniform pmf
        p_greedy *= pcond_greedy

        greedy = np.random.choice([True, False], p=[1. - eps, eps])

        if greedy:
            return a_greedy, p_greedy
        else:
            a = int(np.random.choice(self.action_space_dims))
            # A greedy action can still be picked
            if a == a_greedy:
                return a, p_greedy
            else:
                return a, eps / self.action_space_dims

    def state_value(self, s):
        probs = [
            self.get_sa_probability(s, a)
            for a in range(self.action_space_dims)
        ]

        action_values = [
            self.Q[s][a] for a in range(self.action_space_dims)
        ]

        return sum([p * q for p, q in zip(probs, action_values)])

    def state_update_count(self, s):
        updates = [
            self.Q_update_count[s][a]
            for a in range(self.action_space_dims)
        ]

        return sum(updates)

    def optimal_state_value(self, s):
        a, p = self.get_greedy_action(s)
        return self.Q[s][a]

    def optimal_state_update_count(self, s):
        a, p = self.get_greedy_action(s)
        return self.Q_update_count[s][a]



class DiscreteActionRandomAgent(DiscreteActionAgent, SoftPolicy):
    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            distribution: scipy.stats.rv_discrete,
            distribution_args: Dict,
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        DiscreteActionAgent.__init__(self, obs_space_dims, action_space_dims)
        SoftPolicy.__init__(self)

        self.action_gen = PDFSampler(distribution, distribution_args)
        self.t = 0

    def act(self, s) -> Tuple[int, float]:
        a = self.action_gen.sample(size=1)
        p = self.action_gen.probability(a)
        return a, p

    def step(self, trajectory: List[Experience]):
        self.t += 1

    def get_sa_probability(self, s, a) -> float:
        return self.action_gen.probability(a)

    def get_greedy_action(self, s) -> Tuple[int, float]:
        return self.act(s)

    def reset(self):
        self.t = 0