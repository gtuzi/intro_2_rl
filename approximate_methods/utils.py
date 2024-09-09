from typing import Tuple, Union, Callable, Any, List
import numpy as np

from shared.policies import SoftPolicy
from shared.utils import NoiseSchedule, Experience

class DiscreteActionAgent:
    def __init__(
            self,
            feature_size: int,
            action_space_dims: int
    ):
        self.feature_size = feature_size
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


class LinearQEpsGreedyAgent(DiscreteActionAgent, SoftPolicy):

    def __init__(
            self,
            feature_size: int,
            action_space_dims: int,
            discount: float,
            feature_fn: Callable[[Any, int], np.ndarray], # state, action(int) --> np.ndarray
            eps: Union[float, NoiseSchedule] = 0.01
    ):
        DiscreteActionAgent.__init__(self, feature_size, action_space_dims)
        SoftPolicy.__init__(self)
        self.discount = discount
        self.eps = eps
        self.Q = None
        self.Q_update_count = None
        self.w = np.zeros((feature_size, 1), dtype=np.float32) #[features|actions]
        self.feature_fn = feature_fn

    def init_weights(self, *args, **kwargs):
        pass

    def action_values(self, s) -> np.ndarray:
        """ Q[s, .., a[i], ... | w] """

        # Quick shape check
        r = self.feature_fn(s, 0)
        assert 0 < len(r.shape) <= 2
        assert r.shape[0] == self.w.shape[0]
        assert r.shape[1] == 1 if len(r.shape) == 2 else None

        if len(r.shape) == 2:
            res =  np.array([
                np.dot(self.w.T, self.feature_fn(s, a)).squeeze()
                for a in range(self.action_space_dims)
            ])
        else:
            res = np.array([
                np.dot(self.w.T, self.feature_fn(s, a)[..., None]).squeeze()
                for a in range(self.action_space_dims)
            ])

        return res

    def state_action_value(self, s: Any, a: int) -> float:
        """ Q[s, a | w] """

        x = self.feature_fn(s, a)
        assert x.shape == self.w.shape

        return np.dot(self.w.T, x).squeeze()


    def get_greedy_action(self, s) -> Tuple[int, float]:
        """
            Get the greedy action and it's conditional prob.
            If a single action: probability is 1.
            If multiple actions compete for being picked, they are randomly
            tie-broken. This mean's that their probability is 1/|argmax_a|
        """
        av = self.action_values(s)
        max_vals = np.amax(av)
        idc = np.argwhere(av == max_vals).squeeze().tolist()

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
        av = self.action_values(s)
        max_vals = np.amax(av)
        idc = np.argwhere(av == max_vals).squeeze().tolist()

        if isinstance(idc, list) and (a in idc):
            return (1. - eps + eps / self.action_space_dims) * (1. / len(idc))
        elif isinstance(idc, int) and (a == idc):
            assert isinstance(idc, int)
            return 1. - eps + eps / self.action_space_dims
        # Action is not greedy.
        else:
            return eps / self.action_space_dims


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
            # The greedy action can still be picked
            if a == a_greedy:
                return a, p_greedy
            else:
                if pcond_greedy > (1. - 0.0001):
                    # We're guaranteed that the greedy action was not tie-broken
                    # so the non-greedy has eps/num_actions probability
                    return a, eps / self.action_space_dims

                # The non-greedy action here, may have been one of the
                # randomly tie-broken greedy actions. This means that the
                # probability of this action may not exactly
                # eps / action_space_dims but rather p_greedy, where we've
                # already scaled it with pcond_greedy

                av = self.action_values(s)
                max_vals = np.amax(av)
                idc = np.argwhere(av == max_vals).squeeze().tolist()
                if a in idc:
                    return a, p_greedy
                else:
                    return a, eps / self.action_space_dims


    def state_value(self, s):
        """ V[s] """
        probs = [
            self.get_sa_probability(s, a)
            for a in range(self.action_space_dims)
        ]

        av = self.action_values(s)

        return sum([p * q for p, q in zip(probs, av)])


    def optimal_state_value(self, s):
        a, _ = self.get_greedy_action(s)
        av = self.action_values(s)
        return av[a]
