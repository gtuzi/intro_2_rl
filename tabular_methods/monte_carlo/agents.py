from typing import List, Optional, Union
from collections import defaultdict
from tabular_methods.utils import (
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
            update_coefficient: Union[float, NoiseSchedule],
            discount: float = 0.9,
            eps: Union[float, NoiseSchedule] = 0.01,
            q_init: float = 0.,
            seed: Optional[int] = None
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
            eps=eps,
            seed=seed
        )

        self.update_coefficient = update_coefficient
        self.qval_init = q_init
        self.trajectory = []

    def initialize(self):
        self.t = 0
        self.trajectory = []

        if isinstance(self.eps, NoiseSchedule):
            # Reset noise to starting exploration
            self.eps.initialize()

        # Initialize Q[si][aj] = 0.
        self.Q = defaultdict(lambda: [self.qval_init] * self.action_space_dims)
        self.num_visits = defaultdict(lambda: 0)

    def reset(self):
        self.t = 0
        self.trajectory = []

    def step(self, e: Experience, **kwargs) -> float:
        self.trajectory.append(e)

        loss = 0
        if e.done:
            loss = self._update()
            self.reset()
        else:
            self.t += 1

        return loss

    def _update(self) -> float:
        """
            Update
        :param trajectory:
        :return:
        """

        # Needed for first visit determination

        if isinstance(self.update_coefficient, NoiseSchedule):
            self.update_coefficient.step()
            alpha = self.update_coefficient.value
        elif self.update_coefficient is None:
            alpha = None
        else:
            alpha = self.update_coefficient

        trajectory_sa = [(exp.s, exp.a) for exp in self.trajectory]

        G = 0

        for ti, experience in enumerate(reversed(self.trajectory)):
            s, a, r = experience.s, experience.a, experience.r
            G = r + self.discount * G

            loss = 0.

            # First visit
            if not (s, a) in trajectory_sa[:-(ti + 1)]:
                self.num_visits[(s, a)] += 1

                _error = G - self.Q[s][a]

                # Moving average
                if alpha is None:
                    self.Q[s][a] += (1. / self.num_visits[(s, a)]) * _error
                else:
                    # Constant-alpha method
                    self.Q[s][a] += alpha * _error

                loss += _error

        # We've evaluated this policy, now improve
        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

        return loss


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
            update_coefficient: Union[float, NoiseSchedule] = None, #unused
            discount: float = 0.9,
            eps: Union[float, NoiseSchedule] = 0.01,
            q_init: float = 0.,
            seed: Optional[int] = None
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
            eps=eps,
            seed=seed
        )

        self.qval_init = q_init
        self.trajectory = []

    def initialize(self):
        self.t = 0
        self.trajectory = []

        if isinstance(self.eps, NoiseSchedule):
            # Reset noise to starting exploration
            self.eps.initialize()
        # Initialize Q[si][aj] = 0., C[si][aj] = 0
        self.Q = defaultdict(lambda: [self.qval_init] * self.action_space_dims)
        self.C = defaultdict(lambda: [0.] * self.action_space_dims)

    def reset(self):
        self.t = 0
        self.trajectory = []

    def step(self, e: Experience, **kwargs) -> float:

        self.trajectory.append(e)

        loss = 0
        if e.done:
            loss = self._update()
            self.reset()
        else:
            self.t += 1

        return loss

    def _update(self) -> float:
        """
            Update agent (ie learn). This is the actual implementation
            of the Algo in sect 5.7 Sutton book.
        :param trajectory:
        :return:
        """

        G = 0
        W = 1.
        loss = 0.

        for ti, experience in enumerate(reversed(self.trajectory)):
            s, a, r, p = experience.s, experience.a, experience.r, experience.p

            # Monte Carlo: use the actual return (G) as target
            #              E[G(t) | S(t), a(t)]
            G = r + self.discount * G
            self.C[s][a] += W

            err = G - self.Q[s][a]
            loss += err

            # Moving average
            self.Q[s][a] += (W / self.C[s][a]) * err

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

        # We've evaluated this policy, now improve
        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

        return loss

