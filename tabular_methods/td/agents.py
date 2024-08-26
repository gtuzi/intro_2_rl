from typing import Union
from collections import defaultdict

import numpy as np

from tabular_methods.utils import Experience, NoiseSchedule, QEpsGreedyAgent


class Sarsa(QEpsGreedyAgent):
    """
        Section: 6.4, Sutton, 2020 book
    """
    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            update_coefficient: float,
            discount: float = 0.9,
            eps: Union[float, NoiseSchedule] = 0.01,
            qval_init: float = 0.
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)
        assert 0. < update_coefficient < 1.

        if not isinstance(eps, NoiseSchedule):
            assert 0 <= eps <= 1

        super().__init__(
            obs_space_dims=obs_space_dims,
            action_space_dims = action_space_dims,
            discount=discount,
            eps=eps
        )

        self.t = 0
        self.update_coefficient = update_coefficient
        self.qval_init = qval_init

    def initialize(self):
        if isinstance(self.eps, NoiseSchedule):
            # Reset noise to starting exploration
            self.eps.initialize()

        # Initialize Q[si][aj] = qval_init
        self.Q = defaultdict(lambda: [self.qval_init] * self.action_space_dims)
        self.Q_update_count = defaultdict(lambda: [0] * self.action_space_dims)

    def step(self, experience: Experience, **kwargs):
        s, a, r, sp, ap, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap, experience.done
        )

        tgt = r + self.discount * self.Q[sp][ap] * (1 - done)
        td_error = tgt - self.Q[s][a]

        self.Q[s][a] += self.update_coefficient * td_error
        self.Q_update_count[s][a] += 1

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0
        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()


class ExpectedSarsa(QEpsGreedyAgent):
    """
        Section: 6.6, Sutton, 2020 book
    """
    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            update_coefficient: float,
            discount: float = 0.9,
            eps: Union[float, NoiseSchedule] = 0.01,
            qval_init: float = 0.
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)
        assert 0. < update_coefficient < 1.

        if not isinstance(eps, NoiseSchedule):
            assert 0 <= eps <= 1

        super().__init__(
            obs_space_dims=obs_space_dims,
            action_space_dims=action_space_dims,
            discount=discount,
            eps=eps
        )

        self.t = 0
        self.update_coefficient = update_coefficient
        self.qval_init = qval_init

    def initialize(self):
        if isinstance(self.eps, NoiseSchedule):
            # Reset noise to starting exploration
            self.eps.initialize()

        # Initialize Q[si][aj] = qval_init
        self.Q = defaultdict(lambda: [self.qval_init] * self.action_space_dims)
        self.Q_update_count = defaultdict(lambda: [0] * self.action_space_dims)

    def step(self, experience: Experience, **kwargs):
        s, a, r, sp, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.done
        )

        qp_expected = sum(
            [
                self.get_sa_probability(s=sp, a=_a) * self.Q[sp][_a]
                for _a in range(self.action_space_dims)
            ]
        )

        tgt = r + self.discount * qp_expected * (1 - done)

        td_error = tgt - self.Q[s][a]

        self.Q[s][a] += self.update_coefficient * td_error
        self.Q_update_count[s][a] += 1

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0
        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()


class QLearning(QEpsGreedyAgent):
    """
        aka: SarsaMax
        Section: 6.5, Sutton, 2020 book
    """

    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            update_coefficient: float,
            discount: float = 0.9,
            eps: Union[float, NoiseSchedule] = 0.01,
            qval_init: float = 0.
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)
        assert 0. < update_coefficient < 1.

        if not isinstance(eps, NoiseSchedule):
            assert 0 <= eps <= 1

        super().__init__(
            obs_space_dims=obs_space_dims,
            action_space_dims=action_space_dims,
            discount=discount,
            eps=eps
        )

        self.t = 0
        self.update_coefficient = update_coefficient
        self.qval_init = qval_init

    def initialize(self):
        if isinstance(self.eps, NoiseSchedule):
            # Reset noise to starting exploration
            self.eps.initialize()

        # Initialize Q[si][aj] = qval_init
        self.Q = defaultdict(lambda: [self.qval_init] * self.action_space_dims)
        self.Q_update_count = defaultdict(lambda: [0] * self.action_space_dims)

    def step(self, experience: Experience, **kwargs):
        s, a, r, sp, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.done
        )

        # Directly estimate q*
        tgt = r + self.discount * max(self.Q[sp]) * (1 - done)
        td_error = tgt - self.Q[s][a]

        self.Q[s][a] += self.update_coefficient * td_error
        self.Q_update_count[s][a] += 1

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0
        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()


class nStepSarsa(QEpsGreedyAgent):
    """
            Section: 7.2, Sutton, 2020 book
    """

    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            n: int,
            update_coefficient: float,
            discount: float = 0.9,
            eps: Union[float, NoiseSchedule] = 0.01,
            qval_init: float = 0.
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)
        assert 0. < update_coefficient < 1.
        assert n > 0

        if not isinstance(eps, NoiseSchedule):
            assert 0 <= eps <= 1

        super().__init__(
            obs_space_dims=obs_space_dims,
            action_space_dims=action_space_dims,
            discount=discount,
            eps=eps
        )

        self.t = 0
        self.n = n
        self.update_coefficient = update_coefficient
        self.qval_init = qval_init
        self.trajectory = []

    def initialize(self):
        # The agent here is completely dumb
        if isinstance(self.eps, NoiseSchedule):
            # Reset noise to starting exploration
            self.eps.initialize()

        # Initialize Q[si][aj] = qval_init
        self.Q = defaultdict(lambda: [self.qval_init] * self.action_space_dims)
        self.Q_update_count = defaultdict(lambda: [0] * self.action_space_dims)

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0

        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()

        self.trajectory = []

    def step(self, e: Experience):
        self.trajectory.append(e)
        tau = self.t - self.n + 1

        # If the episode ends before n-steps have been rolled out
        if e.done and (tau < 0):
            tau = 0

        if tau >= 0:
            self.update(tau)

        self.t += 1

    def update(self, tau: int):
        # ------ Policy Evaluation ------- #
        # starting from min(n-steps, T/done) back
        tau_end = min(tau+self.n, len(self.trajectory))

        target = sum(
            [
                # Notationally, in the book, for a[t], reward is r[t+1].
                # So while the book starts the accumulation of rewards at
                # tau+1, this means that tau+1 indexes the
                # (s[tau], a[tau], r[tau+1], s[tau+1]) experience
                (self.discount ** i) * e.r
                for i, e in enumerate(self.trajectory[tau:tau_end])
            ]
        )

        experience_tau = self.trajectory[tau]
        experience_tau_end = self.trajectory[tau_end-1]

        if not experience_tau_end.done:
            # Episode not terminated
            target += (self.discount ** self.n) * self.Q[
                experience_tau_end.sp][experience_tau_end.ap]

        # This is still a TD method
        td_error = target - self.Q[experience_tau.s][experience_tau.a]

        self.Q[experience_tau.s][experience_tau.a] += (
                self.update_coefficient * td_error
        )

        self.Q_update_count[experience_tau.s][experience_tau.a] += 1

        # ------ Policy Improvement ------- #
        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()


class nStepsSarsaOffPolicy(QEpsGreedyAgent):
    """
        Algorithm in Section 7.3 in Sutton 2020 book.
    """
    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            n: int,
            update_coefficient: float,
            discount: float = 0.9,
            eps: Union[float, NoiseSchedule] = 0.01,
            qval_init: float = 0.
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)
        assert 0. < update_coefficient < 1.
        assert n > 0

        if not isinstance(eps, NoiseSchedule):
            assert 0 <= eps <= 1

        super().__init__(
            obs_space_dims=obs_space_dims,
            action_space_dims=action_space_dims,
            discount=discount,
            eps=eps
        )

        self.t = 0
        self.n = n
        self.update_coefficient = update_coefficient
        self.qval_init = qval_init
        self.trajectory = []
        self.td_errors = None

    def initialize(self):
        # The agent here is completely dumb
        if isinstance(self.eps, NoiseSchedule):
            # Reset noise to starting exploration
            self.eps.initialize()

        # Initialize Q[si][aj] = qval_init
        self.Q = defaultdict(lambda: [self.qval_init] * self.action_space_dims)
        self.Q_update_count = defaultdict(lambda: [0] * self.action_space_dims)

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0

        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()

        self.trajectory = []

    def step(self, e: Experience):
        self.trajectory.append(e)
        tau = self.t - self.n + 1

        # If the episode ends before n-steps have been rolled out
        if e.done and (tau < 0):
            tau = 0

        if tau >= 0:
            self.update(tau)

        self.t += 1

    def update(self, tau: int):
        # ------ Policy Evaluation ------- #
        # starting from min(n-steps, T/done) back
        tau_end = min(tau+self.n, len(self.trajectory))

        target = sum(
            [
                # Notationally, in the book, for a[t], reward is r[t+1].
                # So while the book starts the accumulation of rewards at
                # tau+1, this means that tau+1 indexes the
                # (s[tau], a[tau], r[tau+1], s[tau+1]) experience
                (self.discount ** i) * e.r
                for i, e in enumerate(self.trajectory[tau:tau_end])
            ]
        )

        tau_end_rho = min(tau + self.n, len(self.trajectory) - 1)

        rho = [
            self.get_sa_probability(e.s, e.a) / e.p
            for e in self.trajectory[tau:tau_end_rho]
        ]

        if len(rho) > 0:
            rho = np.prod(rho)
        else:
            rho = 1.

        experience_tau = self.trajectory[tau]
        experience_tau_end = self.trajectory[tau_end-1]

        if not experience_tau_end.done:
            # Episode not terminated
            target += (self.discount ** self.n) * self.Q[
                experience_tau_end.sp][experience_tau_end.ap]

        # This is still a TD method
        td_error = rho * (target - self.Q[experience_tau.s][experience_tau.a])
        self.Q[experience_tau.s][experience_tau.a] += (
                self.update_coefficient * td_error)
        self.Q_update_count[experience_tau.s][experience_tau.a] += 1

        # ------ Policy Improvement ------- #
        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()


class QSigmaOffPolicy(QEpsGreedyAgent):
    """
        Algorithm in Section 7.6 in Sutton 2020 book.
    """

    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            n: int,
            update_coefficient: float,
            discount: float = 0.9,
            eps: Union[float, NoiseSchedule] = 0.01,
            qval_init: float = 0.
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)
        assert 0. < update_coefficient < 1.
        assert n > 0

        if not isinstance(eps, NoiseSchedule):
            assert 0 <= eps <= 1

        super().__init__(
            obs_space_dims=obs_space_dims,
            action_space_dims=action_space_dims,
            discount=discount,
            eps=eps
        )

        self.t = 0
        self.n = n
        self.update_coefficient = update_coefficient
        self.qval_init = qval_init
        self.trajectory = []

    def initialize(self):
        # The agent here is completely dumb
        if isinstance(self.eps, NoiseSchedule):
            # Reset noise to starting exploration
            self.eps.initialize()

        # Initialize Q[si][aj] = qval_init
        self.Q = defaultdict(lambda: [self.qval_init] * self.action_space_dims)
        self.Q_update_count = defaultdict(lambda: [0] * self.action_space_dims)

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0

        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()

        self.trajectory = []

    def step(self, e: Experience):
        self.trajectory.append(e)
        tau = self.t - self.n + 1

        # If the episode ends before n-steps have been rolled out
        if e.done and (tau < 0):
            tau = 0

        if tau >= 0:
            self.update(tau)

        self.t += 1

    def update(self, tau):
        T = np.inf
        last_experience = self.trajectory[-1]

        # if t + 1 < T. "done" is int
        if last_experience.done:
            T = last_experience.t + 1
            G = last_experience.r
        else:
            G = self.Q[last_experience.sp][last_experience.ap]

        # t + 1 or T index is not included
        for k in reversed(range(tau, min(last_experience.t + 1, T))):
            # k in the book is pegged to the t+1 in the sarsa experience
            # here k is pegged to t in the sarsa experience
            ek: Experience = self.trajectory[k]
            if ek.done:
                G = ek.r
            else:
                V = sum(
                    [
                        self.get_sa_probability(ek.sp, a) * self.Q[ek.sp][a]
                        for a in range(self.action_space_dims)
                     ]
                )

                # r is t+1
                prob_p = self.get_sa_probability(ek.sp, ek.ap)
                q_p = self.Q[ek.sp][ek.ap]
                G = ek.r + self.discount * (ek.rhop * ek.sigmap + (1 - ek.sigmap) * prob_p) * (G - q_p) + self.discount * V

        e_tau = self.trajectory[tau]
        q_tau = self.Q[e_tau.s][e_tau.a]
        self.Q[e_tau.s][e_tau.a] = q_tau + self.update_coefficient * (G - q_tau)
        self.Q_update_count[e_tau.s][e_tau.a] += 1
