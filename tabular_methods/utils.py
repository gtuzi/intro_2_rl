from typing import Tuple
import numpy as np

from shared.policies import SoftPolicy
from shared.utils import *


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

    def get_greedy_action(self, s) -> Tuple[int, float]:
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
            return ((1. - eps) / len(idc)) + eps / self.action_space_dims
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

                av = self.Q[s]
                max_vals = np.amax(av)
                idc = np.argwhere(av == max_vals).squeeze().tolist()
                if a in idc:
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