from typing import Union, Callable, Any, Optional
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from approximate_methods.utils import (
    LinearQEpsGreedyAgent,
    NoiseSchedule,
    Experience)
from shared.utils import LinearSchedule

# ************* Semi-Gradient ************* #

#######################################
# ----------- 1-step ---------------- #
#######################################

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

        self._writer: Optional[SummaryWriter] = None

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    @writer.setter
    def writer(self, w: SummaryWriter):
        if w is not None:
            assert isinstance(w, SummaryWriter)
        self._writer = w

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

        log_step = None
        if 'log_step' in kwargs:
            log_step = kwargs['log_step']

        next_qhat = self.state_action_value(sp, ap) * (1 - done)
        qhat = self.state_action_value(s, a)

        tgt = r + self.discount * next_qhat
        td_error = tgt - qhat

        # Grad_wi(sum(xi * wi)) = xi
        grad_w = self.feature_fn(s, a)

        if isinstance(self.update_coefficient, LinearSchedule):
            alpha = self.update_coefficient.value
            self.update_coefficient.step()
        elif isinstance(self.update_coefficient, float):
            alpha = self.update_coefficient
        else:
            raise Exception("Invalid type for update_coefficient")

        update = alpha * td_error * grad_w
        self.w += update

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

        if (self._writer is not None) and (log_step is not None):
            root_name = f'on_policy/semi_gradient/sarsa/'

            self._writer.add_scalar(root_name + 'target', tgt, log_step)
            self._writer.add_scalar(root_name + 'td_error', td_error, log_step)
            self._writer.add_scalar(root_name +'q', qhat, log_step)
            self._writer.add_scalar(root_name + 'q_next', next_qhat, log_step)
            self._writer.add_scalar(root_name + 'alpha', alpha, log_step)

            if isinstance(self.eps, NoiseSchedule):
                e = self.eps.value
            else:
                e = self.eps

            self._writer.add_scalar(root_name + 'epsilon', e, log_step)
            self.writer.add_histogram(root_name + 'grad_w', grad_w, log_step)
            self.writer.add_histogram(root_name + 'update', update, log_step)


class SemiGradientExpectedSarsa(SemiGradientSarsa):
    """
        A natural extension of approximate method Sarsa to Expected Sarsa,
        following the discussion of the tabular methods of Chapter 6 (6.6) and
        Chapter 7 (7.2; note eq. 7.8)
    """

    def step(self, experience: Experience, **kwargs):
        # ap is already taken from eps-greedy call
        s, a, r, sp, ap, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap, experience.done
        )

        log_step = None
        if 'log_step' in kwargs:
            log_step = kwargs['log_step']

        # For Expected Sarsa, next step state-action value is the expectation
        # over actions, i.e. V(s'). Refer to Chapter 7, eq. 7.8) for the
        # tabular case
        v_next = self.state_value(sp) * (1 - done)
        qhat = self.state_action_value(s, a)

        tgt = r + self.discount * v_next
        td_error = tgt - qhat

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


        if (self._writer is not None) and (log_step is not None):
            self._writer.add_scalar('target', tgt, log_step)
            self._writer.add_scalar('td_error', td_error, log_step)
            self._writer.add_scalar('q', qhat, log_step)
            self._writer.add_scalar('v_next', v_next, log_step)
            self._writer.add_scalar('alpha', alpha, log_step)

            if isinstance(self.eps, NoiseSchedule):
                e = self.eps.value
            else:
                e = self.eps

            self._writer.add_scalar('epsilon', e, log_step)
            self.writer.add_histogram('grad_w', grad_w, log_step)


class SemiGradientQLearning(SemiGradientSarsa):
    """
        A natural extension of approximate method Sarsa to Sarsa-max
        aka Q-Learning.
        Following the discussion of the tabular methods of Chapter 6 (6.5)
    """

    def step(self, experience: Experience, **kwargs):
        # ap is already taken from eps-greedy call
        s, a, r, sp, ap, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap, experience.done
        )

        log_step = None
        if 'log_step' in kwargs:
            log_step = kwargs['log_step']

        # For Expected Sarsa, next step state-action value is
        # the max_a(q(a, S')), aka Q-Learning.
        qhat = self.state_action_value(s, a)
        qhat_next = max(self.action_values(sp)) * (1 - done)

        tgt = r + self.discount * qhat_next
        td_error = tgt - qhat

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


        if (self._writer is not None) and (log_step is not None):
            self._writer.add_scalar('target', tgt, log_step)
            self._writer.add_scalar('td_error', td_error, log_step)
            self._writer.add_scalar('q', qhat, log_step)
            self._writer.add_scalar('q_next', qhat_next, log_step)
            self._writer.add_scalar('alpha', alpha, log_step)

            if isinstance(self.eps, NoiseSchedule):
                e = self.eps.value
            else:
                e = self.eps

            self._writer.add_scalar('epsilon', e, log_step)
            self.writer.add_histogram('grad_w', grad_w, log_step)


#######################################
# ----------- n-step ---------------- #
#######################################

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

        self._writer: Optional[SummaryWriter] = None

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    @writer.setter
    def writer(self, w: SummaryWriter):
        if w is not None:
            assert isinstance(w, SummaryWriter)
        self._writer = w

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


    def step(self, e: Experience, **kwargs):
        self.trajectory.append(e)
        tau = self.t - self.n + 1

        # If the episode ends before n-steps have been rolled out
        if e.done and (tau < 0):
            tau = 0

        if tau >= 0:
            self.update(tau, **kwargs)

        self.t += 1

    def update(self, tau, **kwargs):
        log_step = None
        if 'log_step' in kwargs:
            log_step = kwargs['log_step']

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

        qhat_next = None
        if not experience_tau_end.done:
            # Episode not terminated
            # (tau + n) - th td step portion of the target
            qhat_next = self.state_action_value(
                experience_tau_end.sp,
                experience_tau_end.ap
            )
            target += (self.discount ** self.n) * qhat_next


        # --- Policy Improvement --- #
        # This is still a TD method, so we still have a TD error
        qhat = self.state_action_value(experience_tau.s, experience_tau.a)
        td_error = target - qhat

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

        if (self._writer is not None) and (log_step is not None):
            self._writer.add_scalar('target', target, log_step)
            self._writer.add_scalar('td_error', td_error, log_step)
            self._writer.add_scalar('q', qhat, log_step)

            if qhat_next is not None:
                self._writer.add_scalar('q_next', qhat_next, log_step)

            self._writer.add_scalar('alpha', alpha, log_step)

            if isinstance(self.eps, NoiseSchedule):
                e = self.eps.value
            else:
                e = self.eps

            self._writer.add_scalar('epsilon', e, log_step)
            self.writer.add_histogram('grad_w', grad_w, log_step)


class nStepSemiGradientExpectedSarsa(nStepSemiGradientSarsa):

    def update(self, tau, **kwargs):
        log_step = None
        if 'log_step' in kwargs:
            log_step = kwargs['log_step']

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

        vhat_next = None
        if not experience_tau_end.done:
            # Episode not terminated
            # (tau + n) - th td step portion of the target
            # This is expected Sarsa, so we get E_pi(*|s)[Q(*, s)]
            vhat_next = self.state_value(experience_tau_end.sp)
            target += (self.discount ** self.n) * vhat_next

        # --- Policy Improvement --- #
        # This is still a TD method, so we still have a TD error
        qhat = self.state_action_value(experience_tau.s, experience_tau.a)
        td_error = target - qhat

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

        if (self._writer is not None) and (log_step is not None):
            self._writer.add_scalar('target', target, log_step)
            self._writer.add_scalar('td_error', td_error, log_step)
            self._writer.add_scalar('q', qhat, log_step)

            if vhat_next is not None:
                self._writer.add_scalar('v_next', vhat_next, log_step)

            self._writer.add_scalar('alpha', alpha, log_step)

            if isinstance(self.eps, NoiseSchedule):
                e = self.eps.value
            else:
                e = self.eps

            self._writer.add_scalar('epsilon', e, log_step)
            self.writer.add_histogram('grad_w', grad_w, log_step)


class nStepSemiGradientQLearning(nStepSemiGradientSarsa):

    def update(self, tau, **kwargs):
        log_step = None
        if 'log_step' in kwargs:
            log_step = kwargs['log_step']

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

        qhat_next = None
        if not experience_tau_end.done:
            # Episode not terminated
            # (tau + n) - th td step portion of the target
            # This is Sarsa max, so we get max_a(Q(*, s))
            qhat_next = max(self.action_values(experience_tau_end.sp))
            target += (self.discount ** self.n) * qhat_next

        # --- Policy Improvement --- #
        # This is still a TD method, so we still have a TD error
        qhat = self.state_action_value(experience_tau.s, experience_tau.a)
        td_error = target - qhat

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

        if (self._writer is not None) and (log_step is not None):
            self._writer.add_scalar('target', target, log_step)
            self._writer.add_scalar('td_error', td_error, log_step)
            self._writer.add_scalar('q', qhat, log_step)

            if qhat_next is not None:
                self._writer.add_scalar('q_next', qhat_next, log_step)

            self._writer.add_scalar('alpha', alpha, log_step)

            if isinstance(self.eps, NoiseSchedule):
                e = self.eps.value
            else:
                e = self.eps

            self._writer.add_scalar('epsilon', e, log_step)
            self.writer.add_histogram('grad_w', grad_w, log_step)


# ************* Differential Semi-Gradient ************* #

#######################################
# ----------- 1-step ---------------- #
#######################################

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
        self._writer: Optional[SummaryWriter] = None


    @property
    def writer(self) -> SummaryWriter:
        return self._writer


    @writer.setter
    def writer(self, w: SummaryWriter):
        if w is not None:
            assert isinstance(w, SummaryWriter)
        self._writer = w


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

        log_step = None
        if 'log_step' in kwargs:
            log_step = kwargs['log_step']

        # ap is already taken from eps-greedy call
        s, a, r, sp, ap, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap, experience.done
        )

        # ---- Policy Evaluation ---- #
        # Expecting continuing task
        assert done == 0

        qhat_next = self.state_action_value(sp, ap)
        qhat = self.state_action_value(s, a)
        target = r - self.reward_estimate + qhat_next
        delta = target - qhat

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

        if (self._writer is not None) and (log_step is not None):
            self._writer.add_scalar('target', target, log_step)
            self._writer.add_scalar('td_error', delta, log_step)
            self._writer.add_scalar('q', qhat, log_step)
            self._writer.add_scalar('q_next', qhat_next, log_step)
            self._writer.add_scalar('alpha', alpha, log_step)
            self._writer.add_scalar(
                'reward_estimate', self.reward_estimate, log_step)

            if isinstance(self.eps, NoiseSchedule):
                e = self.eps.value
            else:
                e = self.eps

            self._writer.add_scalar('epsilon', e, log_step)
            self.writer.add_histogram('grad_w', grad_w, log_step)


class DifferentialSemiGradientQLearning(DifferentialSemiGradientSarsa):

    def step(self, experience: Experience, **kwargs):
        log_step = None
        if 'log_step' in kwargs:
            log_step = kwargs['log_step']

        # ap is already taken from eps-greedy call
        s, a, r, sp, ap, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap, experience.done
        )

        # ---- Policy Evaluation ---- #
        # Expecting continuing task
        assert done == 0

        qhat_next = max(self.action_values(sp))
        qhat = self.state_action_value(s, a)
        target = r - self.reward_estimate + qhat_next
        delta = target - qhat

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

        if (self._writer is not None) and (log_step is not None):
            self._writer.add_scalar('target', target, log_step)
            self._writer.add_scalar('td_error', delta, log_step)
            self._writer.add_scalar('q', qhat, log_step)
            self._writer.add_scalar('q_next', qhat_next, log_step)
            self._writer.add_scalar('alpha', alpha, log_step)
            self._writer.add_scalar(
                'reward_estimate', self.reward_estimate, log_step)

            if isinstance(self.eps, NoiseSchedule):
                e = self.eps.value
            else:
                e = self.eps

            self._writer.add_scalar('epsilon', e, log_step)
            self.writer.add_histogram('grad_w', grad_w, log_step)


class DifferentialSemiGradientExpectedSarsa(DifferentialSemiGradientSarsa):

    def step(self, experience: Experience, **kwargs):
        log_step = None
        if 'log_step' in kwargs:
            log_step = kwargs['log_step']

        # ap is already taken from eps-greedy call
        s, a, r, sp, ap, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap, experience.done
        )

        # ---- Policy Evaluation ---- #
        # Expecting continuing task
        assert done == 0

        vhat_next = self.state_value(sp)
        qhat = self.state_action_value(s, a)
        target = r - self.reward_estimate + vhat_next
        delta = target - qhat

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
                "Invalid type for estimated reward update_coefficient"
            )

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

        if (self._writer is not None) and (log_step is not None):
            self._writer.add_scalar('target', target, log_step)
            self._writer.add_scalar('td_error', delta, log_step)
            self._writer.add_scalar('q', qhat, log_step)
            self._writer.add_scalar('v_next', vhat_next, log_step)
            self._writer.add_scalar('alpha', alpha, log_step)
            self._writer.add_scalar(
                'reward_estimate', self.reward_estimate, log_step)

            if isinstance(self.eps, NoiseSchedule):
                e = self.eps.value
            else:
                e = self.eps

            self._writer.add_scalar('epsilon', e, log_step)
            self.writer.add_histogram('grad_w', grad_w, log_step)


#######################################
# ----------- n-step ---------------- #
#######################################

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
            nsteps: int,
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
        self.nsteps = nsteps
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
        tau = self.t - self.nsteps + 1

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
            for i in range(tau, tau + self.nsteps)
        ]

        delta = (sum(rdiff) +
                 self.state_action_value(
                     self.trajectory[tau + self.nsteps - 1].sp,
                     self.trajectory[tau + self.nsteps - 1].ap
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


class DifferentialSemiGradient_nStepExpectedSarsa(
    DifferentialSemiGradient_nStepSarsa):
    """
        This algo is not in the book. It's simply the adaptation of the
        n-step Sarsa to n-Step Expected Sarsa
    """

    def update(self, tau, **kwargs):

        """
            In the book, for step "t", the experience is
            formated as (R[t+1], S[t+1], A[t], S[t]). So, for (10.14),
            given that our trajectory[t] = (R[t+1], S[t+1], A[t], S[t]),
            we sum the rewards over trajectory over tau --> tau + n - 1
        """

        rdiff = [
            self.trajectory[i].r - self.reward_estimate
            for i in range(tau, tau + self.nsteps)
        ]

        vhat_next = self.state_value(self.trajectory[tau + self.nsteps - 1].sp)
        target = sum(rdiff) + vhat_next

        estimate = self.state_action_value(
            self.trajectory[tau].s, self.trajectory[tau].a
        )

        delta = target - estimate

        if isinstance(self.estimated_reward_update_coefficient, LinearSchedule):
            beta = self.estimated_reward_update_coefficient.value
            self.estimated_reward_update_coefficient.step()
        elif isinstance(self.estimated_reward_update_coefficient, float):
            beta = self.estimated_reward_update_coefficient
        else:
            raise Exception(
                "Invalid type for estimated reward update_coefficient")

        # Compensate for the slowiness (i.e. nonstationarity) of the reward update
        # Ref. Section 2.7 in book
        self.reward_estimate_unbiased_trick += (
                beta * (1. - self.reward_estimate_unbiased_trick))
        self.reward_estimate += (
                (beta / self.reward_estimate_unbiased_trick) * delta
        )

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


class DifferentialSemiGradient_nStepQLearning(
    DifferentialSemiGradient_nStepSarsa):
    """
        This algo is not in the book. It's simply the adaptation of the
        n-step Sarsa to n-Step Expected Sarsa
    """

    def update(self, tau, **kwargs):

        """
            In the book, for step "t", the experience is
            formated as (R[t+1], S[t+1], A[t], S[t]). So, for (10.14),
            given that our trajectory[t] = (R[t+1], S[t+1], A[t], S[t]),
            we sum the rewards over trajectory over tau --> tau + n - 1
        """

        rdiff = [
            self.trajectory[i].r - self.reward_estimate
            for i in range(tau, tau + self.nsteps)
        ]

        qhat_next = max(
            self.action_values(self.trajectory[tau + self.nsteps - 1].sp)
        )

        target = sum(rdiff) + qhat_next

        estimate = self.state_action_value(
            self.trajectory[tau].s, self.trajectory[tau].a
        )

        delta = target - estimate

        if isinstance(
                self.estimated_reward_update_coefficient, LinearSchedule):
            beta = self.estimated_reward_update_coefficient.value
            self.estimated_reward_update_coefficient.step()
        elif isinstance(self.estimated_reward_update_coefficient, float):
            beta = self.estimated_reward_update_coefficient
        else:
            raise Exception(
                "Invalid type for estimated reward update_coefficient")

        # Compensate for the slowiness (i.e. nonstationarity) of the reward update
        # Ref. Section 2.7 in book
        self.reward_estimate_unbiased_trick += beta * (
                    1. - self.reward_estimate_unbiased_trick)

        self.reward_estimate += (
                (beta / self.reward_estimate_unbiased_trick) * delta)

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