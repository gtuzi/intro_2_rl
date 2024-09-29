from typing import Union, Callable, Any
import numpy as np

from approximate_methods.utils import (
    LinearQEpsGreedyAgent,
    NoiseSchedule,
    Experience)
from shared.utils import LinearSchedule


class SemiGradientSarsa(LinearQEpsGreedyAgent):
    def __init__(
            self,
            feature_size: int,
            action_space_dims: int,
            update_coefficient: Union[float, NoiseSchedule],
            feature_fn: Callable[[Any, int], np.ndarray], # state, action(int) --> np.ndarray
            discount: float = 0.9,
            eps: Union[float, LinearSchedule] = 0.01
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if isinstance(update_coefficient, float):
            assert 0. < update_coefficient < 1.
        else:
            assert isinstance(update_coefficient, LinearSchedule)

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

        if isinstance(self.update_coefficient, LinearSchedule):
            self.update_coefficient.initialize()

        self.init_weights()


    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0
        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()

        if isinstance(self.update_coefficient, LinearSchedule):
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

        if isinstance(self.update_coefficient, LinearSchedule):
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
            eps: Union[float, LinearSchedule] = 0.01
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)
        assert n > 0

        if isinstance(update_coefficient, float):
            assert 0. < update_coefficient < 1.
        else:
            assert isinstance(update_coefficient, LinearSchedule)

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

        if isinstance(self.update_coefficient, LinearSchedule):
            self.update_coefficient.initialize()

        self.init_weights()

        self.trajectory = []


    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0
        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()

        if isinstance(self.update_coefficient, LinearSchedule):
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

        if isinstance(self.update_coefficient, LinearSchedule):
            alpha = self.update_coefficient.value
            self.update_coefficient.step()
        elif isinstance(self.update_coefficient, float):
            alpha = self.update_coefficient
        else:
            raise Exception("Invalid type for update_coefficient")

        self.w += alpha * td_error * grad_w

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()


class DifferentialSemiGradientSarsa(LinearQEpsGreedyAgent):
    def __init__(
            self,
            feature_size: int,
            action_space_dims: int,
            update_coefficient: Union[float, NoiseSchedule],
            estimated_reward_update_coefficient: Union[float, NoiseSchedule],
            feature_fn: Callable[[Any, int], np.ndarray], # state, action(int) --> np.ndarray
            eps: Union[float, NoiseSchedule] = 0.01
    ):

        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if isinstance(update_coefficient, float):
            assert 0. < update_coefficient < 1.
        else:
            assert isinstance(update_coefficient, LinearSchedule)


        if isinstance(estimated_reward_update_coefficient, float):
            assert 0. < estimated_reward_update_coefficient < 1.
        else:
            assert isinstance(estimated_reward_update_coefficient, LinearSchedule)

        if not isinstance(eps, NoiseSchedule):
            assert 0 <= eps <= 1

        super().__init__(
            feature_size=feature_size,
            action_space_dims=action_space_dims,
            feature_fn=feature_fn,
            discount=None,  # gamma not used in this agent
            eps=eps)

        self.t = 0
        self.reward_estimate = 0 # r_hat
        self.estimated_reward_update_coefficient = estimated_reward_update_coefficient
        self.update_coefficient = update_coefficient

    def initialize(self):
        if isinstance(self.eps, NoiseSchedule):
            # Reset noise to starting exploration
            self.eps.initialize()

        if isinstance(self.update_coefficient, NoiseSchedule):
            self.update_coefficient.initialize()

        if isinstance(self.estimated_reward_update_coefficient, NoiseSchedule):
            self.estimated_reward_update_coefficient.initialize()

        self.reward_estimate = 0
        self.init_weights()


    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0

        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()

        if isinstance(self.update_coefficient, NoiseSchedule):
            self.update_coefficient.reset()

        if isinstance(self.estimated_reward_update_coefficient, NoiseSchedule):
            self.estimated_reward_update_coefficient.reset()


    def step(self, experience: Experience, **kwargs):
        # ap is already taken from eps-greedy call
        s, a, r, sp, ap, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap, experience.done
        )

        # ---- Policy Evaluation ---- #
        # Expecting continuing task
        assert done == 0

        delta = r - self.reward_estimate + self.state_action_value(sp, ap) - self.state_action_value(s, a)

        if isinstance(self.estimated_reward_update_coefficient, LinearSchedule):
            beta = self.estimated_reward_update_coefficient.value
            self.estimated_reward_update_coefficient.step()
        elif isinstance(self.estimated_reward_update_coefficient, float):
            beta = self.estimated_reward_update_coefficient
        else:
            raise Exception("Invalid type for estimated reward update_coefficient")

        self.reward_estimate += beta * delta

        # ---- Policy Improvement ---- #
        grad_w = self.feature_fn(s, a)  # grad_wi(sum(xi * wi)) = xi

        if isinstance(self.update_coefficient, LinearSchedule):
            alpha = self.update_coefficient.value
            self.update_coefficient.step()
        elif isinstance(self.update_coefficient, float):
            alpha = self.update_coefficient
        else:
            raise Exception("Invalid type for update_coefficient")

        self.w += alpha * delta * grad_w

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()


class DifferentialSemiGradientQLearning(DifferentialSemiGradientSarsa):
    def __init__(
            self,
            feature_size: int,
            action_space_dims: int,
            update_coefficient: Union[float, NoiseSchedule],
            estimated_reward_update_coefficient: Union[float, NoiseSchedule],
            feature_fn: Callable[[Any, int], np.ndarray], # state, action(int) --> np.ndarray
            eps: Union[float, NoiseSchedule] = 0.01
    ):

        super().__init__(
            feature_size=feature_size,
            action_space_dims=action_space_dims,
            update_coefficient=update_coefficient,
            estimated_reward_update_coefficient=estimated_reward_update_coefficient,
            feature_fn=feature_fn,
            eps=eps)

        self.max_reward = -np.inf


    def initialize(self):
        super().initialize()
        self.max_reward = -np.inf

    def step(self, experience: Experience, **kwargs):
        # ap is already taken from eps-greedy call
        s, a, r, sp, ap, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap, experience.done
        )

        # ---- Policy Evaluation ---- #
        assert done == 0  # expecting continuing task

        # max_a'{Q(s', a')}
        max_val = max(
            [
                self.state_action_value(sp, a)
                for a in range(self.action_space_dims)
            ]
        )

        if r > self.max_reward:
            self.max_reward = r

        delta = r - self.max_reward + max_val - self.state_action_value(s, a)

        if isinstance(
                self.estimated_reward_update_coefficient,
                LinearSchedule
        ):
            beta = self.estimated_reward_update_coefficient.value
            self.estimated_reward_update_coefficient.step()
        elif isinstance(self.estimated_reward_update_coefficient, float):
            beta = self.estimated_reward_update_coefficient
        else:
            raise Exception(
                "Invalid type for estimated reward update_coefficient")

        # self.reward_estimate += beta * delta

        # ---- Policy Improvement ---- #
        grad_w = self.feature_fn(s, a)  # grad_wi(sum(xi * wi)) = xi

        if isinstance(self.update_coefficient, LinearSchedule):
            alpha = self.update_coefficient.value
            self.update_coefficient.step()
        elif isinstance(self.update_coefficient, float):
            alpha = self.update_coefficient
        else:
            raise Exception("Invalid type for update_coefficient")

        self.w += alpha * delta * grad_w

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()


class DifferentialSemiGradient_nStepSarsa(LinearQEpsGreedyAgent):
    """
        Implements algorithm in 10.5 in Sutton, 2020 book
    """

    def __init__(
            self,
            feature_size: int,
            action_space_dims: int,
            update_coefficient: Union[float, NoiseSchedule],
            estimated_reward_update_coefficient: Union[float, NoiseSchedule],
            feature_fn: Callable[[Any, int], np.ndarray], # state, action(int) --> np.ndarray
            nstep_sarsa: int,
            eps: Union[float, NoiseSchedule] = 0.1
    ):

        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if isinstance(update_coefficient, float):
            assert 0. < update_coefficient < 1.
        else:
            assert isinstance(update_coefficient, LinearSchedule)

        if isinstance(estimated_reward_update_coefficient, float):
            assert 0. < estimated_reward_update_coefficient < 1.
        else:
            assert isinstance(
                estimated_reward_update_coefficient,
                LinearSchedule
            )

        if not isinstance(eps, NoiseSchedule):
            assert 0 <= eps <= 1

        super().__init__(
            feature_size=feature_size,
            action_space_dims=action_space_dims,
            feature_fn=feature_fn,
            discount=None,  # gamma not used in this agent
            eps=eps)

        self.trajectory = []
        self.t = 0
        self.nstep_sarsa = nstep_sarsa
        self.reward_estimate = 0  # r_hat
        self.reward_estimate_unbiased_trick = 0
        self.estimated_reward_update_coefficient = estimated_reward_update_coefficient
        self.update_coefficient = update_coefficient


    def initialize(self):
        if isinstance(self.eps, NoiseSchedule):
            # Reset noise to starting exploration
            self.eps.initialize()

        if isinstance(self.update_coefficient, NoiseSchedule):
            self.update_coefficient.initialize()

        if isinstance(self.estimated_reward_update_coefficient, NoiseSchedule):
            self.estimated_reward_update_coefficient.initialize()

        self.t = 0
        self.trajectory = []
        self.reward_estimate = 0
        self.reward_estimate_unbiased_trick = 0
        self.init_weights()

    def reset(self):
        # Weights are not cleared. Reset does not unlearn
        self.t = 0
        self.trajectory = []

        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()

        if isinstance(self.update_coefficient, NoiseSchedule):
            self.update_coefficient.reset()

        if isinstance(self.estimated_reward_update_coefficient, NoiseSchedule):
            self.estimated_reward_update_coefficient.reset()

    def step(self, experience: Experience, **kwargs):

        self.trajectory.append(experience)
        tau = self.t - self.nstep_sarsa + 1

        # If the episode ends before n-steps have been rolled out
        if experience.done and (tau < 0):
            tau = 0

        if tau >= 0:
            self.update(tau)

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

        self.t += 1

    def update(self, tau, **kwargs):

        """
            In the book, for step "t", the experience is
            formated as (R[t+1], S[t+1], A[t], S[t]). So, for (10.14),
            given that our trajectory[t] = (R[t+1], S[t+1], A[t], S[t]),
            we sum the rewards over trajectory over tau --> tau + n - 1
        """

        rdiff = [
            self.trajectory[i].r - self.reward_estimate
            for i in range(tau, tau + self.nstep_sarsa)
        ]

        delta = (sum(rdiff) +
                 self.state_action_value(
                     self.trajectory[tau + self.nstep_sarsa - 1].sp,
                     self.trajectory[tau + self.nstep_sarsa - 1].ap
                 ) -
                 self.state_action_value(
                     self.trajectory[tau].s,
                     self.trajectory[tau].a
                 )
        )

        if isinstance(self.estimated_reward_update_coefficient, LinearSchedule):
            beta = self.estimated_reward_update_coefficient.value
            self.estimated_reward_update_coefficient.step()
        elif isinstance(self.estimated_reward_update_coefficient, float):
            beta = self.estimated_reward_update_coefficient
        else:
            raise Exception("Invalid type for estimated reward update_coefficient")

        # Compensate for the slowiness (i.e. nonstationarity) of the reward update
        # Ref. Section 2.7 in book
        self.reward_estimate_unbiased_trick += beta * (1. - self.reward_estimate_unbiased_trick)
        self.reward_estimate += (beta / self.reward_estimate_unbiased_trick) * delta

        grad_w = self.feature_fn(
            self.trajectory[tau].s, self.trajectory[tau].a)

        if isinstance(self.update_coefficient, LinearSchedule):
            alpha = self.update_coefficient.value
            self.update_coefficient.step()
        elif isinstance(self.update_coefficient, float):
            alpha = self.update_coefficient
        else:
            raise Exception("Invalid type for update_coefficient")

        self.w += alpha * delta * grad_w
