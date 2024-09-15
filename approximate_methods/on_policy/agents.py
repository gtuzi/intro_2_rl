from typing import Union, Callable, Any
import numpy as np

from approximate_methods.utils import (
    LinearQEpsGreedyAgent,
    NoiseSchedule,
    Experience)
from shared.utils import LinearEpsSchedule


class SemiGradientSarsa(LinearQEpsGreedyAgent):
    def __init__(
            self,
            feature_size: int,
            action_space_dims: int,
            update_coefficient: Union[float, NoiseSchedule],
            feature_fn: Callable[[Any, int], np.ndarray], # state, action(int) --> np.ndarray
            discount: float = 0.9,
            eps: Union[float, LinearEpsSchedule] = 0.01
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if isinstance(update_coefficient, float):
            assert 0. < update_coefficient < 1.
        else:
            assert isinstance(update_coefficient, LinearEpsSchedule)

        if not isinstance(eps, NoiseSchedule):
            assert 0 <= eps <= 1

        super().__init__(
            feature_size=feature_size,
            action_space_dims=action_space_dims,
            feature_fn=feature_fn,
            discount=discount,
            eps=eps)

        self.t = 0
        self.update_coefficient = update_coefficient


    def initialize(self):
        if isinstance(self.eps, NoiseSchedule):
            # Reset noise to starting exploration
            self.eps.initialize()

        if isinstance(self.update_coefficient, LinearEpsSchedule):
            self.update_coefficient.initialize()


    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0
        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()

        if isinstance(self.update_coefficient, LinearEpsSchedule):
            self.update_coefficient.reset()


    def step(self, experience: Experience, **kwargs):
        # ap is already taken from eps-greedy call
        s, a, r, sp, ap, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap, experience.done
        )

        tgt = r + self.discount * self.state_action_value(sp, ap) * (1 - done)
        td_error = tgt - self.state_action_value(s, a)

        # Grad_wi(sum(xi * wi)) = xi
        grad_w = self.feature_fn(s, a)

        if isinstance(self.update_coefficient, LinearEpsSchedule):
            alpha = self.update_coefficient.value
            self.update_coefficient.step()
        elif isinstance(self.update_coefficient, float):
            alpha = self.update_coefficient
        else:
            raise Exception("Invalid type for update_coefficient")

        self.w += alpha * td_error * grad_w

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()


class nStepSemiGradientSarsa(LinearQEpsGreedyAgent):
    def __init__(
            self,
            feature_size: int,
            action_space_dims: int,
            n: int,
            update_coefficient: Union[float, NoiseSchedule],
            feature_fn: Callable[[Any, int], np.ndarray], # state, action(int) --> np.ndarray
            discount: float = 0.9,
            eps: Union[float, LinearEpsSchedule] = 0.01
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)
        assert n > 0

        if isinstance(update_coefficient, float):
            assert 0. < update_coefficient < 1.
        else:
            assert isinstance(update_coefficient, LinearEpsSchedule)

        if not isinstance(eps, NoiseSchedule):
            assert 0 <= eps <= 1

        super().__init__(
            feature_size=feature_size,
            action_space_dims=action_space_dims,
            feature_fn=feature_fn,
            discount=discount,
            eps=eps)

        self.t = 0
        self.n = n
        self.update_coefficient = update_coefficient
        self.trajectory = []


    def initialize(self):
        if isinstance(self.eps, NoiseSchedule):
            # Reset noise to starting exploration
            self.eps.initialize()

        if isinstance(self.update_coefficient, LinearEpsSchedule):
            self.update_coefficient.initialize()


    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0
        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()

        if isinstance(self.update_coefficient, LinearEpsSchedule):
            self.update_coefficient.reset()

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

        # --- Policy Evaluation --- #

        # starting from min(n-steps, T/done) back
        tau_end = min(tau + self.n, len(self.trajectory))

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
        experience_tau_end = self.trajectory[tau_end - 1]  # tau + n - 1

        if not experience_tau_end.done:
            # Episode not terminated
            # (tau + n) - th td step portion of the target
            qh = self.state_action_value(
                experience_tau_end.sp,
                experience_tau_end.ap
            )
            target += (self.discount ** self.n) * qh


        # --- Policy Improvement --- #
        # This is still a TD method, so we still have a TD error
        td_error = target - self.state_action_value(
            experience_tau.s, experience_tau.a)

        # gradient of the state-action value function
        # Grad_wi(sum(xi * wi)) = xi
        grad_w = self.feature_fn(experience_tau.s, experience_tau.a)

        if isinstance(self.update_coefficient, LinearEpsSchedule):
            alpha = self.update_coefficient.value
            self.update_coefficient.step()
        elif isinstance(self.update_coefficient, float):
            alpha = self.update_coefficient
        else:
            raise Exception("Invalid type for update_coefficient")

        self.w += alpha * td_error * grad_w

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

