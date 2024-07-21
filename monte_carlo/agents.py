from typing import List, Optional, Union
from collections import defaultdict
from utils import (
    Experience,
    NoiseSchedule,
    QEpsGreedyAgent
)


class MCOnPolicyFirstVisitGLIE(QEpsGreedyAgent):
    """
        Implementation of algorithm in 5.4: On-policy first-visit MC control
    """
    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            discount: float = 0.9,
            eps: Union[float, NoiseSchedule] = 0.01,
            qval_init: float = 0.,
            step_size: Optional[float] = None
    ):

        self.t = 0
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if not isinstance(eps, NoiseSchedule):
            assert 0 <= eps <= 1

        super().__init__(
            obs_space_dims=obs_space_dims,
            action_space_dims=action_space_dims,
            discount=discount,
            eps=eps
        )

        self.step_size = step_size
        self.qval_init = qval_init

    def initialize(self):
        if isinstance(self.eps, NoiseSchedule):
            # Reset noise to starting exploration
            self.eps.initialize()

        # Initialize Q[si][aj] = 0.
        self.Q = defaultdict(lambda: [self.qval_init] * self.action_space_dims)
        self.num_visits = defaultdict(lambda: 0)

    def step(self, trajectory: List[Experience]):
        """
            Update
        :param trajectory:
        :return:
        """
        self.t += 1

        # Needed for first visit determination
        trajectory_sa = [(exp.s, exp.a) for exp in trajectory]

        G = 0

        for ti, experience in enumerate(reversed(trajectory)):
            s, a, r = experience.s, experience.a, experience.r
            G = r + self.discount * G

            # First visit
            if not (s, a) in trajectory_sa[:-(ti + 1)]:
                self.num_visits[(s, a)] += 1

                # Moving average
                if self.step_size is None:
                    self.Q[s][a] += (1. / self.num_visits[(s, a)]) * (
                                G - self.Q[s][a])
                else:
                    # Constant-alpha method
                    self.Q[s][a] += self.step_size * (G - self.Q[s][a])

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

    def reset(self):
        self.t = 0
        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()


class MCOffPolicy(QEpsGreedyAgent):
    """
        This is the MC Off-Policy agent. It is OffPolicy because it implements
        the IS weighted updates. This agent can be used as deterministic (i.e.
        eps = 0) or as soft eps-greedy.

        Implements the "Off-Policy MC control for estimating pi~=pi*", section
        5.7 in Sutton Book.
    """

    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            discount: float = 0.9,
            eps: Union[float, NoiseSchedule] = 0.01,
            qval_init: float = 0.
    ):
        self.t = 0
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if not isinstance(eps, NoiseSchedule):
            assert 0 <= eps <= 1

        super().__init__(
            obs_space_dims=obs_space_dims,
            action_space_dims=action_space_dims,
            discount=discount,
            eps=eps
        )

        self.qval_init = qval_init

    def initialize(self):
        if isinstance(self.eps, NoiseSchedule):
            # Reset noise to starting exploration
            self.eps.initialize()
        # Initialize Q[si][aj] = 0., C[si][aj] = 0
        self.Q = defaultdict(lambda: [self.qval_init] * self.action_space_dims)
        self.C = defaultdict(lambda: [0.] * self.action_space_dims)

    def step(self, trajectory: List[Experience]):
        """
            Update agent (ie learn). This is the actual implementation
            of the Algo in sect 5.7 Sutton book.
        :param trajectory:
        :return:
        """
        self.t += 1

        G = 0
        W = 1.
        for ti, experience in enumerate(reversed(trajectory)):
            s, a, r, p = experience.s, experience.a, experience.r, experience.p

            # Monte Carlo: use the actual return (G) as target
            #              E[G(t) | S(t), a(t)]
            G = r + self.discount * G
            self.C[s][a] += W
            # Moving average
            self.Q[s][a] += (W / self.C[s][a]) * (G - self.Q[s][a])

            # For soft policy case, this will be > 0.
            # For deterministic policy (eps = 0), this will be 1.
            # In the derministic case, this is equivalent to checking
            # if argmax_a == a
            p_tgt = self.get_sa_probability(s, a)
            if p_tgt < 1e-6:
                break

            assert 0 < p_tgt <= 1.
            assert 0 < p <= 1.
            W *= p_tgt / p

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

    def reset(self):
        self.t = 0
        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()
