from typing import Union, Optional
from collections import defaultdict

import numpy as np

from tabular_methods.utils import (
    Experience,
    NoiseSchedule,
    QEpsGreedyAgent
)


class Sarsa(QEpsGreedyAgent):
    """
        Section: 6.4, Sutton, 2020 book
    """
    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            update_coefficient: Union[float, NoiseSchedule],
            discount: float = 0.9,
            eps: Union[float, NoiseSchedule] = 0.01,
            qval_init: float = 0.,
            seed: Optional[int] = None
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if not isinstance(update_coefficient, NoiseSchedule):
            assert 0. < update_coefficient <= 1.

        if not isinstance(eps, NoiseSchedule):
            assert 0 <= eps <= 1

        super().__init__(
            obs_space_dims=obs_space_dims,
            action_space_dims = action_space_dims,
            discount=discount,
            eps=eps,
            seed=seed
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

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0

    def step(self, experience: Experience, **kwargs) -> float:
        s, a, r, sp, ap, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap, experience.done
        )

        if isinstance(self.update_coefficient, NoiseSchedule):
            self.update_coefficient.step()
            alpha = self.update_coefficient.value
        else:
            alpha = self.update_coefficient

        tgt = r + self.discount * self.Q[sp][ap] * (1 - done)
        td_error = tgt - self.Q[s][a]

        self.Q[s][a] += alpha * td_error
        self.Q_update_count[s][a] += 1

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

        return td_error


class ExpectedSarsa(QEpsGreedyAgent):
    """
        Section: 6.6, Sutton, 2020 book
    """
    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            update_coefficient: Union[float, NoiseSchedule],
            discount: float = 0.9,
            eps: Union[float, NoiseSchedule] = 0.01,
            qval_init: float = 0.,
            seed: Optional[int] = None
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if not isinstance(update_coefficient, NoiseSchedule):
            assert 0. < update_coefficient <= 1.

        if not isinstance(eps, NoiseSchedule):
            assert 0 <= eps <= 1

        super().__init__(
            obs_space_dims=obs_space_dims,
            action_space_dims=action_space_dims,
            discount=discount,
            eps=eps,
            seed=seed
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

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0

    def step(self, experience: Experience, **kwargs) -> float:
        s, a, r, sp, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.done
        )

        if isinstance(self.update_coefficient, NoiseSchedule):
            self.update_coefficient.step()
            alpha = self.update_coefficient.value
        else:
            alpha = self.update_coefficient

        qp_expected = sum(
            [
                self.get_sa_probability(s=sp, a=_a) * self.Q[sp][_a]
                for _a in range(self.action_space_dims)
            ]
        )

        tgt = r + self.discount * qp_expected * (1 - done)

        td_error = tgt - self.Q[s][a]

        self.Q[s][a] += alpha * td_error
        self.Q_update_count[s][a] += 1

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

        return td_error


class QLearning(QEpsGreedyAgent):
    """
        aka: SarsaMax
        Section: 6.5, Sutton, 2020 book
    """

    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            update_coefficient: Union[float, NoiseSchedule],
            discount: float = 0.9,
            eps: Union[float, NoiseSchedule] = 0.01,
            qval_init: float = 0.,
            seed: Optional[int] = None
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if not isinstance(update_coefficient, NoiseSchedule):
            assert 0. < update_coefficient <= 1.

        if not isinstance(eps, NoiseSchedule):
            assert 0 <= eps <= 1

        super().__init__(
            obs_space_dims=obs_space_dims,
            action_space_dims=action_space_dims,
            discount=discount,
            eps=eps,
            seed=seed
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

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0

    def step(self, experience: Experience, **kwargs) -> float:
        s, a, r, sp, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.done
        )

        if isinstance(self.update_coefficient, NoiseSchedule):
            self.update_coefficient.step()
            alpha = self.update_coefficient.value
        else:
            alpha = self.update_coefficient

        # Directly estimate q*
        tgt = r + self.discount * max(self.Q[sp]) * (1 - done)
        td_error = tgt - self.Q[s][a]

        self.Q[s][a] += alpha * td_error
        self.Q_update_count[s][a] += 1

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

        return td_error


class nStepSarsa(QEpsGreedyAgent):
    """
            Section: 7.2, Sutton, 2020 book
    """

    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            n: int,
            update_coefficient: Union[float, NoiseSchedule],
            discount: float = 0.9,
            eps: Union[float, NoiseSchedule] = 0.01,
            qval_init: float = 0.,
            seed: Optional[int] = None
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)
        assert n > 0

        if not isinstance(update_coefficient, NoiseSchedule):
            assert 0. < update_coefficient <= 1.

        if not isinstance(eps, NoiseSchedule):
            assert 0 <= eps <= 1

        super().__init__(
            obs_space_dims=obs_space_dims,
            action_space_dims=action_space_dims,
            discount=discount,
            eps=eps,
            seed=seed
        )

        self.t = 0
        self.n = n
        self.update_coefficient = update_coefficient
        self.qval_init = qval_init
        self.trajectory = { }

    def initialize(self):
        self.t = 0
        self.trajectory = {}

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
        self.trajectory = { }

    def step(self, e: Experience) -> float:
        self.trajectory[self.t] = e
        tau = self.t - self.n + 1

        loss = 0
        if tau >= 0:
            loss = self.update(tau)
            _ = self.trajectory.pop(tau)

        while e.done and self.trajectory:
            tau += 1
            if tau >= 0:
                loss = self.update(tau)
                self.trajectory.pop(tau)

        if e.done:
            self.t = 0
        else:
            self.t += 1

        return loss

    def update(self, tau: int):
        if isinstance(self.update_coefficient, NoiseSchedule):
            self.update_coefficient.step()
            alpha = self.update_coefficient.value
        else:
            alpha = self.update_coefficient

        T = np.inf
        if self.trajectory[self.t].done:
            # T is the now time, terminal R (T+1) is here.
            # So wrt the book, "T" here is "T-1" in the book
            T = self.t

        G = [
            # Notationally, in the book, for a[t], reward is r[t+1].
            # So while the book starts the accumulation of rewards at
            # tau+1, this means that tau+1 indexes the
            # (s[tau], a[tau], r[tau+1], s[tau+1]) experience
            (self.discount ** i) * self.trajectory[t].r
            for i, t in enumerate(range(tau, min(tau + self.n, T + 1)))
        ]
        assert 0 < len(G) <= self.n
        G = sum(G)

        if (
                (tau + self.n - 1 <= self.t)
                and
                (not self.trajectory[tau + self.n - 1].done
        )):
            # Include (S, A) which generated terminal S'
            sp = self.trajectory[tau + self.n - 1].sp
            ap = self.trajectory[tau + self.n - 1].ap
            G = G + (self.discount ** self.n) * self.Q[sp][ap]

        s = self.trajectory[tau].s
        a = self.trajectory[tau].a
        td_error = G - self.Q[s][a]
        self.Q[s][a] = self.Q[s][a] + alpha * td_error
        self.Q_update_count[s][a] += 1

        # ------ Policy Improvement ------- #
        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

        return td_error


class nStepsSarsaOffPolicy(QEpsGreedyAgent):
    """
        Algorithm in Section 7.3 in Sutton 2020 book.
    """
    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            n: int,
            update_coefficient: Union[float, NoiseSchedule],
            discount: float = 0.9,
            eps: Union[float, NoiseSchedule] = 0.01,
            qval_init: float = 0.,
            seed: Optional[int] = None
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)
        assert n > 0

        if not isinstance(update_coefficient, NoiseSchedule):
            assert 0. < update_coefficient <= 1.

        if not isinstance(eps, NoiseSchedule):
            assert 0 <= eps <= 1

        super().__init__(
            obs_space_dims=obs_space_dims,
            action_space_dims=action_space_dims,
            discount=discount,
            eps=eps,
            seed=seed
        )

        self.t = 0
        self.n = n
        self.update_coefficient = update_coefficient
        self.qval_init = qval_init
        self.trajectory = { }
        self.td_errors = None

    def initialize(self):
        self.t = 0
        self.trajectory = {}

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
        self.trajectory = { }

    def step(self, e: Experience) -> float:
        self.trajectory[self.t] = e
        tau = self.t - self.n + 1

        loss = 0
        if tau >= 0:
            loss = self.update(tau)
            _ = self.trajectory.pop(tau)

        while e.done and self.trajectory:
            tau += 1
            if tau >= 0:
                loss = self.update(tau)
                self.trajectory.pop(tau)

        if e.done:
            self.t = 0
        else:
            self.t += 1

        return loss

    def update(self, tau: int) -> float:

        if isinstance(self.update_coefficient, NoiseSchedule):
            self.update_coefficient.step()
            alpha = self.update_coefficient.value
        else:
            alpha = self.update_coefficient

        T = np.inf
        if self.trajectory[self.t].done:
            # T is the now time, terminal R (T+1) is here.
            # So wrt the book, "T" here is "T-1" in the book
            T = self.t

        G = [
            # Notationally, in the book, for a[t], reward is r[t+1].
            # So while the book starts the accumulation of rewards at
            # tau+1, this means that tau+1 indexes the
            # (s[tau], a[tau], r[tau+1], s[tau+1]) experience
            (self.discount ** i) * self.trajectory[t].r
            for i, t in enumerate(range(tau, min(tau + self.n, T + 1)))
        ]
        assert 0 < len(G) <= self.n
        G = sum(G)

        if (
                (tau + self.n - 1 <= self.t)
                and
                (not self.trajectory[tau + self.n - 1].done
        )):
            # Include (S, A) which generated terminal S'
            sp = self.trajectory[tau + self.n - 1].sp
            ap = self.trajectory[tau + self.n - 1].ap
            G = G + (self.discount ** self.n) * self.Q[sp][ap]

        rho = [
            self.get_sa_probability(
                s=self.trajectory[t].sp,
                a=self.trajectory[t].ap
            ) / (self.trajectory[t].pp + 1e-8)
            for t in range(tau, min(tau + self.n, T))
        ]

        if tau < T:
            assert 0 < len(rho) <= self.n
            rho = np.prod(rho)
        elif tau == T:
            assert len(rho) == 0
            rho = 1
        else:
            raise Exception(
                "tau has gone beyond n-step horizon. Must not happen")

        if (tau + self.n - 1 <= self.t) and (
                not self.trajectory[tau + self.n - 1].done
        ):
            # Include (S, A) which generated terminal S'
            sp = self.trajectory[tau + self.n - 1].sp
            ap = self.trajectory[tau + self.n - 1].ap
            G = G + (self.discount ** self.n) * self.Q[sp][ap]

        s = self.trajectory[tau].s
        a = self.trajectory[tau].a
        td_error = G - self.Q[s][a]
        self.Q[s][a] = self.Q[s][a] + alpha * rho * td_error
        self.Q_update_count[s][a] += 1

        # ------ Policy Improvement ------- #
        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

        return td_error


class nStepsQSigmaOffPolicy(QEpsGreedyAgent):
    """
        Algorithm in Section 7.6 in Barto & Sutton 2nd edition, 2020 book.
    """

    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            n: int,
            update_coefficient: Union[float, NoiseSchedule],
            discount: float = 0.9,
            eps: Union[float, NoiseSchedule] = 0.01,
            qval_init: float = 0.,
            seed: Optional[int] = None
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)
        assert n > 0

        if not isinstance(update_coefficient, NoiseSchedule):
            assert 0. < update_coefficient <= 1.

        if not isinstance(eps, NoiseSchedule):
            assert 0 <= eps <= 1

        super().__init__(
            obs_space_dims=obs_space_dims,
            action_space_dims=action_space_dims,
            discount=discount,
            eps=eps,
            seed=seed
        )

        self.t = 0
        self.n = n
        self.update_coefficient = update_coefficient
        self.qval_init = qval_init
        self.trajectory = { }

    def initialize(self):
        self.t = 0
        self.trajectory = { }

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
        self.trajectory = { }

    def step(self, e: Experience) -> float:
        self.trajectory[self.t] = e
        tau = self.t - self.n + 1
        loss = 0
        if tau >= 0:
            loss = self.update(tau)
            _ = self.trajectory.pop(tau)

        while e.done and self.trajectory:
            tau += 1
            if tau >= 0:
                loss = self.update(tau)
                self.trajectory.pop(tau)

        if e.done:
            self.t = 0
        else:
            self.t += 1

        return loss

    def update(self, tau) -> float:
        if isinstance(self.update_coefficient, NoiseSchedule):
            self.update_coefficient.step()
            alpha = self.update_coefficient.value
        else:
            alpha = self.update_coefficient

        T = np.inf
        # "done" is int
        if self.trajectory[self.t].done:
            # Our T is book's T-1, where our
            # terminal e[T].r = R[T+1] corresponding to
            # the book's R[T]
            # T = self.t
            pass
        else:
            # if t + 1 < T
            G = self.Q[self.trajectory[self.t].sp][self.trajectory[self.t].ap]

        for k in reversed(range(tau, self.t + 1)):
            # [tau, t]
            ek: Experience = self.trajectory[k]
            done = ek.done # done is done[t+1], so done[k]

            if done:
                # Start from terminal R if we're done
                G = ek.r
            else:
                Vbar = sum(
                    [
                        self.get_sa_probability(ek.sp, a) * self.Q[ek.sp][a]
                        for a in range(self.action_space_dims)
                     ]
                )

                # r is t+1
                prob_p = self.get_sa_probability(ek.sp, ek.ap)
                tdp_error = G - self.Q[ek.sp][ek.ap]
                qsig_term = ek.rhop * ek.sigmap + (1 - ek.sigmap) * prob_p
                G = ek.r + self.discount * qsig_term * tdp_error + self.discount * Vbar

        # Update the tau'th sample
        e_tau = self.trajectory[tau]
        q_tau = self.Q[e_tau.s][e_tau.a]

        td_error = G - q_tau
        self.Q[e_tau.s][e_tau.a] = q_tau + alpha * td_error
        self.Q_update_count[e_tau.s][e_tau.a] += 1

        # ------ Policy Improvement ------- #
        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()


        # print(f"error: {td_error:.1e}, alpha: {alpha:.1e}")

        return td_error

