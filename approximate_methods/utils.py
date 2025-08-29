from typing import Tuple, Union, Callable, Any, List
import numpy as np

from shared.utils import SoftPolicy
from shared.utils import NoiseSchedule

class DiscreteActionAgent:
    def __init__(
            self,
            feature_size: int,
            action_space_dims: int
    ):
        self.feature_size = feature_size
        self.action_space_dims = action_space_dims

    def act(self, s, **kwargs) -> Tuple[int, float]:
        """ Return the action and probability """
        raise NotImplementedError

    def initialize(self, **kwargs):
        pass

    def reset(self, **kwargs):
        pass

    def step(self, *args, **kwargs):
        """ Learn: Qk+1 = somefunction(Qk) """
        pass


class ContinuousActionAgent:
    def __init__(
            self,
            feature_size: int,
            action_size: int
    ):
        self.feature_size = feature_size
        self.action_size = action_size

    def act(self, s) -> Tuple[int, float]:
        """ Return the action and probability """
        raise NotImplementedError

    def initialize(self):
        pass

    def reset(self):
        pass

    def step(self, *args, **kwargs):
        pass


class LinearQEpsGreedyAgent(DiscreteActionAgent, SoftPolicy):

    def __init__(
            self,
            feature_size: int,
            action_space_dims: int,
            discount: Union[float, Callable[[Any, ], float]],
            feature_fn: Callable[[Any, int], np.ndarray], # state, action(int) --> np.ndarray
            eps: Union[float, NoiseSchedule] = 0.01
    ):
        DiscreteActionAgent.__init__(self, feature_size, action_space_dims)
        SoftPolicy.__init__(self)
        self.discount = discount
        self.eps = eps
        self.w = None
        self.feature_fn = feature_fn
        self.init_weights()

    def init_weights(self, *args, **kwargs):
        if 'init' in kwargs:
            self.w = kwargs['init']((self.feature_size, 1))
        else:
            self.w = np.zeros((self.feature_size, 1), dtype=np.float32) #[features|actions]

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
            Get the greedy action and its **conditional** prob.
            If a single action: conditional probability is 1.
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
            return ((1. - eps) / len(idc)) + eps / self.action_space_dims
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

        # Decide whether to be greedy or explore
        if np.random.random() > eps:
            # Get the greedy action and its conditional
            # probability (for tie-breaking)
            action, conditional_prob = self.get_greedy_action(s)
        else:
            # Choose any action uniformly at random
            action = int(np.random.choice(self.action_space_dims))

        # Look up the true probability of the chosen action using the correct method
        prob = self.get_sa_probability(s, action)

        return action, prob

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


from approximate_methods.tiles3 import IHT, tiles


# ------ Feature Extractors ------

class TileCodingFeature:
    def __init__(
            self,
            max_size: int,
            num_tiles: int,
            num_tilings: int,
            x0_low: float,
            x1_low: float,
            x0_high: float,
            x1_high: float):

        self.iht = IHT(max_size)
        self.x0_low = x0_low
        self.x1_low = x1_low
        self.x0_high = x0_high
        self.x1_high  = x1_high
        self.num_tiles = num_tiles
        self.num_tilings = num_tilings

    def __call__(
            self,
            x: Union[List, np.ndarray],
            a: int, **kwargs
    ) -> np.ndarray:

        x0, x1 = x[0], x[1]
        feats = tiles(
            ihtORsize=self.iht,
            numtilings=self.num_tilings,
            floats=[
                self.num_tiles * x0 / (self.x0_high - self.x0_low),
                self.num_tiles * x1 / (self.x1_high - self.x1_low)
            ],
            ints=[a]
        )

        feats = np.array(feats)

        res = np.zeros((self.iht.size, 1), dtype=np.float32)
        res[feats] = 1.
        return res


class TileCodingNFeature:
    def __init__(
            self,
            max_size: int,
            num_tiles: int,
            num_tilings: int,
            lows: List[float],
            highs: List[float]):

        assert isinstance(lows, (list, tuple))
        assert isinstance(highs, (list, tuple))
        assert len(lows) == len(highs)

        self.iht = IHT(max_size)
        self.lows = lows
        self.highs = highs
        self.num_tiles = num_tiles
        self.num_tilings = num_tilings

    def __call__(
            self,
            x: Union[List, np.ndarray],
            a: int, **kwargs
    ) -> np.ndarray:

        assert len(x) == len(self.lows) == len(self.highs)

        floats = [
            self.num_tiles * xx / (h - l)
            for xx, l, h in zip(x, self.lows, self.highs)
        ]

        feats = tiles(
            ihtORsize=self.iht,
            numtilings=self.num_tilings,
            floats=floats,
            ints=[a]
        )

        feats = np.array(feats)

        res = np.zeros((self.iht.size, 1), dtype=np.float32)
        res[feats] = 1.
        return res


