from typing import Union, Callable, Any, Optional
import numpy as np
from sympy import gamma

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


        self.z = (self.discount * self.lam * self.z) + (
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


class OffPolicyExpectedSarsaLambda(LinearQEpsGreedyAgent):
    def __init__(
            self,
            feature_size: int,
            action_space_dims: int,
            update_coefficient: Union[float, NoiseSchedule],
            feature_fn: Callable[[Any, int], np.ndarray],
            lam: Union[float, Callable[[Any, int], float]], # generalized λ(s, a)
            discount: Union[float, Callable[[Any, ], float]] = 0.9, # generalized 𝛾(s)
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
        self.lam = lam
        self.z: Optional[np.ndarray] = None
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
        self._step(experience, **kwargs)
        self.t += 1

    def _step(self, experience: Experience, **kwargs):
        # ap is already taken from eps-greedy call
        s, a, r, sp, ap, done, p = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap, experience.done,
            experience.p
        )

        vphat = self.state_value(sp) * (1 - done)
        qhat = self.state_action_value(s=s, a=a)
        grad_w = self.feature_fn(s, a)

        tgtp = self.get_sa_probability(s, a)
        assert p > 0
        rho = tgtp / p

        # region constants
        if isinstance(self.discount, Callable):
            gamma_t = self.discount(s)
            gamma_tt = self.discount(sp)
        elif isinstance(self.discount, float):
            gamma_t = self.discount
            gamma_tt = self.discount
        else:
            raise Exception(f'discount not valid: {self.discount}')

        if isinstance(self.lam, Callable):
            lam_t = self.lam(s, a)
            lam_tt = self.lam(sp, ap)
        elif isinstance(self.lam, float):
            lam_t = self.lam
            lam_tt = self.lam
        else:
            raise Exception(f"lambda not valid: {self.lam}")

        if isinstance(self.update_coefficient, LinearSchedule):
            alpha = self.update_coefficient.value
            self.update_coefficient.step()
        elif isinstance(self.update_coefficient, float):
            alpha = self.update_coefficient
        else:
            raise Exception("Invalid type for update_coefficient")

        # endregion

        delta_a = r + gamma_tt * vphat - qhat  # Expected Sarsa formulation
        self.z = rho * gamma_t * lam_t * self.z + grad_w
        weight_update = alpha * delta_a * self.z
        self.w = np.clip(self.w + weight_update, -BIG_NUMBER, BIG_NUMBER)

        # Decay exploration if decayable
        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

        log_step = None
        if 'log_step' in kwargs:
            log_step = kwargs['log_step']

        if (self._writer is not None) and (log_step is not None):
            root_name = f'off_policy/semi_gradient/offpolicy_expected_sarsa_lambda/'
            self._writer.add_scalar(root_name + 'weights_norm', np.linalg.norm(self.w), log_step)
            self._writer.add_scalar(root_name + 'weight_update_norm', np.linalg.norm(weight_update), log_step)
            self._writer.add_scalar(root_name + 'delta_a', delta_a, log_step)
            self._writer.add_scalar(root_name + 'alpha', alpha, log_step)
            self._writer.add_scalar(root_name + 'gamma_t', gamma_t, log_step)
            self._writer.add_scalar(root_name + 'gamma_tt', gamma_tt, log_step)
            self._writer.add_scalar(root_name + 'lam_t', lam_t, log_step)
            self._writer.add_scalar(root_name + 'grad_w_norm', np.linalg.norm(grad_w), log_step)
            self._writer.add_histogram(root_name + 'grad_w', grad_w, log_step)
            self._writer.add_histogram(root_name + 'weight_update', weight_update, log_step)
            self._writer.add_histogram(root_name + 'weights', self.w, log_step)

            if isinstance(self.eps, NoiseSchedule):
                e = self.eps.value
            else:
                e = self.eps
            self._writer.add_scalar(root_name + 'epsilon', e, log_step)


class TBLambda(LinearQEpsGreedyAgent):
    def __init__(
            self,
            feature_size: int,
            action_space_dims: int,
            update_coefficient: Union[float, NoiseSchedule],
            feature_fn: Callable[[Any, int], np.ndarray],
            lam: Union[float, Callable[[Any, int], float]],# generalized λ(s, a)
            discount: Union[float, Callable[[Any, ], float]] = 0.9,# generalized 𝛾(s)
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
        self.lam = lam
        self.z: Optional[np.ndarray] = None
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
        self._step(experience, **kwargs)
        self.t += 1

    def _step(self, experience: Experience, **kwargs):
        # ap is already taken from eps-greedy call
        s, a, r, sp, ap, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap, experience.done
        )

        vphat = self.state_value(sp) * (1 - done)
        qhat = self.state_action_value(s=s, a=a)
        grad_w = self.feature_fn(s, a)
        p = self.get_sa_probability(s, a)

        # region constants
        if isinstance(self.discount, Callable):
            gamma_t = self.discount(s)
            gamma_tt = self.discount(sp)
        elif isinstance(self.discount, float):
            gamma_t = self.discount
            gamma_tt = self.discount
        else:
            raise Exception(f'discount not valid: {self.discount}')

        if isinstance(self.lam, Callable):
            lam_t = self.lam(s, a)
            lam_tt = self.lam(sp, ap)
        elif isinstance(self.lam, float):
            lam_t = self.lam
            lam_tt = self.lam
        else:
            raise Exception(f"lambda not valid: {self.lam}")

        if isinstance(self.update_coefficient, LinearSchedule):
            alpha = self.update_coefficient.value
            self.update_coefficient.step()
        elif isinstance(self.update_coefficient, float):
            alpha = self.update_coefficient
        else:
            raise Exception("Invalid type for update_coefficient")

        # endregion

        delta_a = r + gamma_tt * vphat - qhat  # Expected Sarsa formulation
        self.z = gamma_t * lam_t * p * self.z + grad_w
        weight_update = alpha * delta_a * self.z
        self.w = np.clip(self.w + weight_update, -BIG_NUMBER, BIG_NUMBER)

        # Decay exploration if decayable
        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

        log_step = None
        if 'log_step' in kwargs:
            log_step = kwargs['log_step']

        if (self._writer is not None) and (log_step is not None):
            root_name = f'off_policy/semi_gradient/tb_lambda/'
            self._writer.add_scalar(root_name + 'weights_norm', np.linalg.norm(self.w), log_step)
            self._writer.add_scalar(root_name + 'weight_update_norm', np.linalg.norm(weight_update), log_step)
            self._writer.add_scalar(root_name + 'delta_a', delta_a, log_step)
            self._writer.add_scalar(root_name + 'alpha', alpha, log_step)
            self._writer.add_scalar(root_name + 'gamma_t', gamma_t, log_step)
            self._writer.add_scalar(root_name + 'gamma_tt', gamma_tt, log_step)
            self._writer.add_scalar(root_name + 'lam_t', lam_t, log_step)
            self._writer.add_scalar(root_name + 'grad_w_norm', np.linalg.norm(grad_w), log_step)
            self._writer.add_histogram(root_name + 'grad_w', grad_w, log_step)
            self._writer.add_histogram(root_name + 'weight_update', weight_update, log_step)
            self._writer.add_histogram(root_name + 'weights', self.w, log_step)

            if isinstance(self.eps, NoiseSchedule):
                e = self.eps.value
            else:
                e = self.eps
            self._writer.add_scalar(root_name + 'epsilon', e, log_step)


class GQLambda(LinearQEpsGreedyAgent):
    def __init__(
            self,
            feature_size: int,
            action_space_dims: int,
            update_coefficient: Union[float, NoiseSchedule],
            second_update_coefficient: Union[float, NoiseSchedule],
            feature_fn: Callable[[Any, int], np.ndarray],
            lam: Union[float, Callable[[Any, int], float]],  # generalized λ(s, a)
            discount: Union[float, Callable[[Any, ], float]] = 0.9, # generalized 𝛾(s)
            eps: Union[float, LinearSchedule] = 0.01
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if isinstance(update_coefficient, float):
            assert 0. < update_coefficient < 1.
        else:
            assert isinstance(update_coefficient, LinearSchedule)

        if isinstance(second_update_coefficient, float):
            assert 0. < second_update_coefficient < 1.
        else:
            assert isinstance(second_update_coefficient, LinearSchedule)

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
        self.q: Optional[np.ndarray] = None
        self.update_coefficient = update_coefficient
        self.second_update_coefficient = second_update_coefficient
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

        if isinstance(self.second_update_coefficient, LinearSchedule):
            self.second_update_coefficient.initialize()

        self.init_weights()
        self.z = np.zeros_like(self.w)  # z_{-1}
        self.q = np.zeros_like(self.w)

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0

        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()

        if isinstance(self.update_coefficient, LinearSchedule):
            self.update_coefficient.reset()

        if isinstance(self.second_update_coefficient, LinearSchedule):
            self.second_update_coefficient.reset()

        # Eligibility traces pertain to one episode
        self.z = np.zeros_like(self.w)  # z_{-1}
        self.q = np.zeros_like(self.w)

    def step(self,  experience: Experience, **kwargs):
        self._step(experience, **kwargs)
        self.t += 1

    def _step(self, experience: Experience, **kwargs):
        # ap is already taken from eps-greedy call
        s, a, r, sp, ap, done, p = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap, experience.done,
            experience.p
        )

        xpbar = np.sum([
            self.get_sa_probability(sp, _a) * self.feature_fn(sp, _a)
            for _a in range(self.action_space_dims)
        ], axis=0, keepdims=False) * (1 - done)

        x = self.feature_fn(s, a) # also the gradient wrt w

        tgtp = self.get_sa_probability(s, a)
        assert p > 0
        rho = tgtp / p

        # region constants

        if isinstance(self.discount, Callable):
            gamma_t = self.discount(s)
            gamma_tt = self.discount(sp)
        elif isinstance(self.discount, float):
            gamma_t = self.discount
            gamma_tt = self.discount
        else:
            raise Exception(f'discount not valid: {self.discount}')

        if isinstance(self.lam, Callable):
            lam_t = self.lam(s, a)
            lam_tt = self.lam(sp, ap)
        elif isinstance(self.lam, float):
            lam_t = self.lam
            lam_tt = self.lam
        else:
            raise Exception(f"lambda not valid: {self.lam}")

        if isinstance(self.update_coefficient, LinearSchedule):
            alpha = self.update_coefficient.value
            self.update_coefficient.step()
        elif isinstance(self.update_coefficient, float):
            alpha = self.update_coefficient
        else:
            raise Exception("Invalid type for update_coefficient")

        if isinstance(self.second_update_coefficient, LinearSchedule):
            beta = self.second_update_coefficient.value
            self.second_update_coefficient.step()
        elif isinstance(self.second_update_coefficient, float):
            beta = self.second_update_coefficient
        else:
            raise Exception("Invalid type for second_update_coefficient")

        # endregion

        delta_a = r + gamma_tt * np.dot(self.w.T, xpbar) - np.dot(self.w.T, x)
        self.z = (lam_t * gamma_t * rho * self.z) + x
        self.q = self.q + beta * delta_a * self.z - beta * np.dot(self.q.T, x) * x

        weight_update = alpha * delta_a * self.z - alpha * gamma_tt * (1 - lam_tt) * np.dot(self.z.T, self.q) * xpbar
        self.w = np.clip(self.w + weight_update, -BIG_NUMBER, BIG_NUMBER)

        # Decay exploration if decayable
        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

        log_step = None
        if 'log_step' in kwargs:
            log_step = kwargs['log_step']

        if (self._writer is not None) and (log_step is not None):
            root_name = f'off_policy/semi_gradient/gq_lambda/'
            self._writer.add_scalar(root_name + 'weights_norm', np.linalg.norm(self.w), log_step)
            self._writer.add_scalar(root_name + 'weight_update_norm', np.linalg.norm(weight_update), log_step)
            self._writer.add_scalar(root_name + 'delta_a', delta_a, log_step)

            self._writer.add_scalar(root_name + 'alpha', alpha, log_step)
            self._writer.add_scalar(root_name + 'beta', beta, log_step)
            self._writer.add_scalar(root_name + 'gamma_t', gamma_t, log_step)
            self._writer.add_scalar(root_name + 'gamma_tt', gamma_tt, log_step)
            self._writer.add_scalar(root_name + 'lam_t', lam_t, log_step)
            self._writer.add_scalar(root_name + 'x_norm', np.linalg.norm(x), log_step)

            if isinstance(self.eps, NoiseSchedule):
                e = self.eps.value
            else:
                e = self.eps

            self._writer.add_scalar(root_name + 'epsilon', e, log_step)
            self._writer.add_histogram(root_name + 'x', x, log_step)
            self._writer.add_histogram(root_name + 'weight_update', weight_update, log_step)
            self._writer.add_histogram(root_name + 'weights', self.w, log_step)
            self._writer.add_histogram(root_name + 'q', self.q, log_step)


class HQLambda(LinearQEpsGreedyAgent):
    """
        An action value adaptation of HTD(lambda). Where I replaced
        the state values for action values and related TD-error,
        which uses the expectation form (i.e. next step expected state value
        V(s')), similar as what was shown in the book for GTD --> GQ
    """
    def __init__(
            self,
            feature_size: int,
            action_space_dims: int,
            update_coefficient: Union[float, NoiseSchedule],
            second_update_coefficient: Union[float, NoiseSchedule],
            feature_fn: Callable[[Any, int], np.ndarray],
            lam: Union[float, Callable[[Any, int], float]],
            discount: Union[float, Callable[[Any, ], float]] = 0.9, # generalized λ(s, a)
            eps: Union[float, LinearSchedule] = 0.01 # generalized 𝛾(s)
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
        self.zb: Optional[np.ndarray] = None
        self.q: Optional[np.ndarray] = None
        self.update_coefficient = update_coefficient
        self.second_update_coefficient = second_update_coefficient
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

        if isinstance(self.second_update_coefficient, LinearSchedule):
            self.second_update_coefficient.initialize()

        self.init_weights()
        self.z = np.zeros_like(self.w)  # z_{-1}
        self.zb = np.zeros_like(self.w)  # z_{-1}
        self.q = np.zeros_like(self.w)

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0

        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()

        if isinstance(self.update_coefficient, LinearSchedule):
            self.update_coefficient.reset()

        if isinstance(self.second_update_coefficient, LinearSchedule):
            self.second_update_coefficient.reset()

        # Eligibility traces pertain to one episode
        self.z = np.zeros_like(self.w)  # z_{-1}
        self.zb = np.zeros_like(self.w)  # z_{-1}
        self.q = np.zeros_like(self.w)

    def step(self, experience: Experience, **kwargs):
        self._step(experience, **kwargs)
        self.t += 1

    def _step(self, experience: Experience, **kwargs):
        # ap is already taken from eps-greedy call
        s, a, r, sp, ap, done, p = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap, experience.done,
            experience.p
        )

        x = self.feature_fn(s, a)  # also the gradient wrt w (∇q(s, a, w))
        xp = self.feature_fn(sp, ap)

        xpbar = np.sum([
            self.get_sa_probability(sp, _a) * self.feature_fn(sp, _a)
            for _a in range(self.action_space_dims)
        ], axis=0, keepdims=False) * (1 - done)


        tgtp = self.get_sa_probability(s, a)
        assert p > 0
        rho = tgtp / p

        # region constants
        if isinstance(self.discount, Callable):
            gamma_t = self.discount(s)
            gamma_tt = self.discount(sp)
        elif isinstance(self.discount, float):
            gamma_t = self.discount
            gamma_tt = self.discount
        else:
            raise Exception(f'discount not valid: {self.discount}')

        if isinstance(self.lam, Callable):
            lam_t = self.lam(s, a)
        elif isinstance(self.lam, float):
            lam_t = self.lam
        else:
            raise Exception(f"lambda not valid: {self.lam}")

        if isinstance(self.update_coefficient, LinearSchedule):
            alpha = self.update_coefficient.value
            self.update_coefficient.step()
        elif isinstance(self.update_coefficient, float):
            alpha = self.update_coefficient
        else:
            raise Exception("Invalid type for update_coefficient")

        if isinstance(self.second_update_coefficient, LinearSchedule):
            beta = self.second_update_coefficient.value
            self.second_update_coefficient.step()
        elif isinstance(self.second_update_coefficient, float):
            beta = self.second_update_coefficient
        else:
            raise Exception("Invalid type for second_update_coefficient")
        # endregion

        # The expectation form of TD-error. Same as in Expected-Sarsa
        # Not spcified in the book particularly, but re-using the same
        # form as in GQ(lambda)
        delta_a = r + gamma_tt * np.dot(self.w.T, xpbar) - np.dot(self.w.T, x)

        weight_update = alpha * delta_a * self.z + alpha * np.dot((self.z - self.zb).T, self.q) * (x - gamma_tt * xp)
        self.w = np.clip(self.w + weight_update, -BIG_NUMBER, BIG_NUMBER)

        q_update = beta * delta_a * self.z - beta * np.dot(self.zb.T, self.q) * (x - gamma_tt * xp)
        self.q = self.q + q_update

        self.z = rho * gamma_t * lam_t * self.z + x
        self.zb = gamma_t * lam_t * self.zb + x

        # Decay exploration if decayable
        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

        # ---- Logs ---- #
        log_step = None
        if 'log_step' in kwargs:
            log_step = kwargs['log_step']

        if (self._writer is not None) and (log_step is not None):
            root_name = f'off_policy/semi_gradient/hq_lambda/'
            self._writer.add_scalar(root_name + 'weights_norm', np.linalg.norm(self.w), log_step)
            self._writer.add_scalar(root_name + 'weight_update_norm', np.linalg.norm(weight_update), log_step)
            self._writer.add_scalar(root_name + 'q_update_norm', np.linalg.norm(q_update), log_step)
            self._writer.add_scalar(root_name + 'delta_a', delta_a, log_step)

            self._writer.add_scalar(root_name + 'alpha', alpha, log_step)
            self._writer.add_scalar(root_name + 'beta', beta, log_step)
            self._writer.add_scalar(root_name + 'gamma_t', gamma_t, log_step)
            self._writer.add_scalar(root_name + 'gamma_tt', gamma_tt, log_step)
            self._writer.add_scalar(root_name + 'lam_t', lam_t, log_step)
            self._writer.add_scalar(root_name + 'x_norm', np.linalg.norm(x), log_step)
            self._writer.add_scalar(root_name + 'xp_norm', np.linalg.norm(xp), log_step)

            if isinstance(self.eps, NoiseSchedule):
                e = self.eps.value
            else:
                e = self.eps

            self._writer.add_scalar(root_name + 'epsilon', e, log_step)
            self._writer.add_histogram(root_name + 'x', x, log_step)
            self._writer.add_histogram(root_name + 'xp', xp, log_step)
            self._writer.add_histogram(root_name + 'weight_update', weight_update, log_step)
            self._writer.add_histogram(root_name + 'weights', self.w, log_step)
            self._writer.add_histogram(root_name + 'q', self.q, log_step)
            self._writer.add_histogram(root_name + 'q_update', q_update, log_step)

