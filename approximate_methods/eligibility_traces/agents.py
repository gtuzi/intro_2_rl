from typing import Union, Callable, Any, Optional
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from approximate_methods.utils import (
    LinearQEpsGreedyAgent,
    NoiseSchedule,
    Experience)
from shared.utils import LinearSchedule

BIG_NUMBER = 1e8

class SarsaLambda(LinearQEpsGreedyAgent):
    def __init__(
            self,
            feature_size: int,
            action_space_dims: int,
            update_coefficient: Union[float, NoiseSchedule],
            feature_fn: Callable[[Any, int], np.ndarray], # state, action(int) --> np.ndarray
            lam: float,
            discount: float = 0.9,
            eps: Union[float, LinearSchedule] = 0.01,
            trace_mode: str = 'replace'
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
        self.lam = lam
        self.z: Optional[np.ndarray] = None
        self.update_coefficient = update_coefficient
        self._writer: Optional[SummaryWriter] = None
        self.trace_mode = trace_mode

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
        self.z = np.zeros_like(self.w)  # z_{-1}

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0

        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()

        if isinstance(self.update_coefficient, LinearSchedule):
            self.update_coefficient.reset()

        # Eligibility traces pertain to one episode
        self.z = np.zeros_like(self.w)  # z_{-1}

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

        qphat = self.state_action_value(sp, ap) * (1 - done)
        qhat = self.state_action_value(s, a)
        grad_w = self.feature_fn(s, a) # Grad_wi(sum(xi * wi)) = xi

        tgt = r + self.discount * qphat
        td_error = tgt - qhat

        if isinstance(self.update_coefficient, LinearSchedule):
            alpha = self.update_coefficient.value
            self.update_coefficient.step()
        elif isinstance(self.update_coefficient, float):
            alpha = self.update_coefficient
        else:
            raise Exception("Invalid type for update_coefficient")

        # ----- Updates ------- #
        # Accumulating traces
        if self.trace_mode == 'accumulate':
            self.z += grad_w
        elif self.trace_mode == 'replace':
            self.z[abs(grad_w) > 1e-3] = 1.
        elif self.trace_mode == 'replace_clear':
            idc = abs(grad_w) > 1e-3
            self.z[idc] = 1.
            self.z[np.logical_not(idc)] = 0.
        else:
            raise Exception('trace mode not recognized.', self.trace_mode)

        weight_update = alpha * td_error * self.z  # To log
        self.w = np.clip(self.w + weight_update, -BIG_NUMBER, BIG_NUMBER)

        # Decay the trace
        if 1 - done:
            self.z *= self.lam * self.discount

        # Decay exploration if decayable
        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

        # ---- Logs ---- #
        if (self._writer is not None) and (log_step is not None):
            root_name = f'on_policy/semi_gradient/sarsa_lambda/'

            self._writer.add_scalar(root_name + 'weights_norm', np.linalg.norm(self.w), log_step)
            self._writer.add_scalar(root_name + 'target', tgt, log_step)
            self._writer.add_scalar(root_name + 'td_error', td_error, log_step)
            self._writer.add_scalar(root_name + 'qhat', qhat, log_step)
            self._writer.add_scalar(root_name + 'qphat', qphat, log_step)
            self._writer.add_scalar(root_name + 'alpha', alpha, log_step)
            self._writer.add_scalar(root_name + 'grad_w_norm', np.linalg.norm(grad_w), log_step)

            if isinstance(self.eps, NoiseSchedule):
                e = self.eps.value
            else:
                e = self.eps

            self._writer.add_scalar(root_name + 'epsilon', e, log_step)
            self._writer.add_histogram(root_name + 'grad_w', grad_w, log_step)
            self._writer.add_histogram(root_name + 'update', weight_update, log_step)
            self._writer.add_histogram(root_name + 'weights', self.w, log_step)


class TrueOnlineSarsaLambda(SarsaLambda):
    def __init__(
            self,
            feature_size: int,
            action_space_dims: int,
            update_coefficient: Union[float, NoiseSchedule],
            feature_fn: Callable[[Any, int], np.ndarray],
            lam: float,
            discount: float = 0.9,
            eps: Union[float, LinearSchedule] = 0.01
    ):
        super().__init__(
            feature_size=feature_size,
            action_space_dims=action_space_dims,
            update_coefficient=update_coefficient,
            feature_fn=feature_fn,
            lam=lam,
            discount=discount,
            eps=eps
        )

        self.Qold = 0.


    def initialize(self):
        super().initialize()
        self.Qold = 0

    def reset(self):
        super().reset()
        self.Qold = 0

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

        qphat = self.state_action_value(sp, ap) * (1 - done)
        qhat = self.state_action_value(s, a)
        x = self.feature_fn(s, a)

        tgt = r + self.discount * qphat
        td_error = tgt - qhat

        if isinstance(self.update_coefficient, LinearSchedule):
            alpha = self.update_coefficient.value
            self.update_coefficient.step()
        elif isinstance(self.update_coefficient, float):
            alpha = self.update_coefficient
        else:
            raise Exception("Invalid type for update_coefficient")


        self.z = (self.discount * self.lam * self.z) * (1 - done) + (
                1 - alpha * self.discount * self.lam * np.dot(self.z.T, x)
        ) * x

        weight_update = alpha * (td_error + qhat - self.Qold) * self.z
        weight_update -= alpha * (qhat - self.Qold) * x
        self.w = np.clip(self.w + weight_update, -BIG_NUMBER, BIG_NUMBER)

        self.Qold = qphat

        # Decay exploration if decayable
        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

        # ---- Logs ---- #
        if (self._writer is not None) and (log_step is not None):
            root_name = f'on_policy/semi_gradient/true_online_sarsa_lambda/'
            self._writer.add_scalar(root_name + 'weights_norm', np.linalg.norm(self.w), log_step)
            self._writer.add_scalar(root_name + 'target', tgt, log_step)
            self._writer.add_scalar(root_name + 'td_error', td_error, log_step)
            self._writer.add_scalar(root_name + 'qhat', qhat, log_step)
            self._writer.add_scalar(root_name + 'qphat', qphat, log_step)
            self._writer.add_scalar(root_name + 'alpha', alpha, log_step)
            self._writer.add_scalar(root_name + 'x', np.linalg.norm(x), log_step)

            if isinstance(self.eps, NoiseSchedule):
                e = self.eps.value
            else:
                e = self.eps

            self._writer.add_scalar(root_name + 'epsilon', e, log_step)
            self._writer.add_histogram(root_name + 'x', x, log_step)
            self._writer.add_histogram(root_name + 'update', weight_update, log_step)
            self._writer.add_histogram(root_name + 'weights', self.w, log_step)
