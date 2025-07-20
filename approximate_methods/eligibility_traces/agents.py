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

        #### These parameters decay over episodes * horizon steps ####
        # if isinstance(self.eps, NoiseSchedule):
        #     self.eps.reset()

        # if isinstance(self.update_coefficient, LinearSchedule):
        #     self.update_coefficient.reset()

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
    """
        Implementation of the Sarsa in expectation from section 12.9
        in the book.
    """
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

        # These parameters decay over episode * horizon
        # if isinstance(self.eps, NoiseSchedule):
        #     self.eps.reset()
        #
        # if isinstance(self.update_coefficient, LinearSchedule):
        #     self.update_coefficient.reset()

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

        vphat = sum([
            self.get_sa_probability(sp, _a) * self.state_action_value(sp, _a)
            for _a in range(self.action_space_dims)
        ]) * (1 - done)  # (12.21)

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

        delta_a = r + gamma_tt * vphat - qhat  # (12.28) - E[Sarsa]
        # z[t] update before w[t+1] update
        # z[t] <-- f(*, z[t-1])
        self.z = rho * gamma_t * lam_t * self.z + grad_w  # (12.29)
        # w[t+1] = f(*, z[t], w[t])
        weight_update = alpha * delta_a * self.z
        self.w = np.clip(self.w + weight_update, -BIG_NUMBER, BIG_NUMBER) # (12.7)


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

        # These parameters decay over episodes * horizon steps
        # if isinstance(self.eps, NoiseSchedule):
        #     self.eps.reset()
        #
        # if isinstance(self.update_coefficient, LinearSchedule):
        #     self.update_coefficient.reset()

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

        vphat = sum([
            self.get_sa_probability(sp, _a) * self.state_action_value(sp, _a)
            for _a in range(self.action_space_dims)
        ]) * (1 - done)  # (12.21)

        qhat = self.state_action_value(s=s, a=a)
        grad_w = self.feature_fn(s, a)

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

        delta_a = r + gamma_tt * vphat - qhat  # (12.28)

        # z[t] <-- f(*, z[t-1])
        self.z = gamma_t * lam_t * self.get_sa_probability(s, a) * self.z + grad_w  # Section: 12.10

        # w[t+1] <-- f(*, w[t], z[t])
        weight_update = alpha * delta_a * self.z
        self.w = np.clip(self.w + weight_update, -BIG_NUMBER, BIG_NUMBER) # (12.7)

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
        self.v: Optional[np.ndarray] = None
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
        self.v = np.zeros_like(self.w)

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0

        # These parameters decay over episodes * horizon steps

        # if isinstance(self.eps, NoiseSchedule):
        #     self.eps.reset()
        #
        # if isinstance(self.update_coefficient, LinearSchedule):
        #     self.update_coefficient.reset()
        #
        # if isinstance(self.second_update_coefficient, LinearSchedule):
        #     self.second_update_coefficient.reset()

        # Eligibility traces pertain to one episode
        self.z = np.zeros_like(self.w)  # z_{-1}
        self.v = np.zeros_like(self.w)  # v_{0}

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

        # xbar = np.sum([
        #     self.get_sa_probability(s, _a) * self.feature_fn(s, _a)
        #     for _a in range(self.action_space_dims)
        # ], axis=0, keepdims=False)

        x = self.feature_fn(s, a) # also the gradient wrt w
        v = self.state_value(s)
        vp = self.state_value(sp) * (1 - done)
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

        # Refer to: Section 12.11
        delta_a = r + gamma_tt * np.dot(self.w.T, xpbar) - np.dot(self.w.T, x) # Section 12.11
        delta_s = r + gamma_tt * vp - v # (12.23)

        # z[t] <-- f(*, z[t-1])
        self.z = (lam_t * gamma_t * rho * self.z) + x  # (12.29)

        # w[t + 1] <-- f(*, w[t], z[t], v[t])
        weight_update = alpha * delta_a * self.z - alpha * gamma_tt * (1 - lam_tt) * np.dot(self.z.T, self.v) * xpbar  # Section 12.11
        self.w = np.clip(self.w + weight_update, -BIG_NUMBER, BIG_NUMBER) # Section 12.11

        # v[t+1] <-- f(*, v[t], z[t])
        self.v = self.v + beta * delta_s * self.z - beta * np.dot(self.v.T, x) * x # (12.30)

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
            self._writer.add_histogram(root_name + 'v', self.v, log_step)


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
        self.v: Optional[np.ndarray] = None
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
        self.v = np.zeros_like(self.w)

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0

        # These parameters decay over episodes * horizon steps
        # if isinstance(self.eps, NoiseSchedule):
        #     self.eps.reset()
        #
        # if isinstance(self.update_coefficient, LinearSchedule):
        #     self.update_coefficient.reset()
        #
        # if isinstance(self.second_update_coefficient, LinearSchedule):
        #     self.second_update_coefficient.reset()

        # Eligibility traces pertain to one episode
        self.z = np.zeros_like(self.w)  # z_{-1}
        self.zb = np.zeros_like(self.w)  # zb_{-1}
        self.v = np.zeros_like(self.w)   # v_{0}

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
        xp = self.feature_fn(sp, ap) * (1 - done)
        v = self.state_value(s)
        vp = self.state_value(sp) * (1 - done)

        xpbar = np.sum([
            self.get_sa_probability(sp, _a) * self.feature_fn(sp, _a)
            for _a in range(self.action_space_dims)
        ], axis=0, keepdims=False) * (1 - done)

        xbar = np.sum([
            self.get_sa_probability(s, _a) * self.feature_fn(s, _a)
            for _a in range(self.action_space_dims)
        ], axis=0, keepdims=False)

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

        # Not spcified in the book particularly, but following the same
        # pattern as in GQ(lambda)
        delta_s = r + gamma_tt * vp - v
        delta_a = r + gamma_tt * np.dot(self.w.T, xpbar) - np.dot(self.w.T, x)

        # z[t] <-- f(*, z[t - 1])
        self.z = rho * gamma_t * lam_t * self.z + x
        self.zb = gamma_t * lam_t * self.zb + x

        weight_update = alpha * delta_a * self.z + alpha * np.dot((self.z - self.zb).T, self.v) * (xbar - gamma_tt * xpbar)
        self.w = np.clip(self.w + weight_update, -BIG_NUMBER, BIG_NUMBER)

        v_update = beta * delta_s * self.z - beta * np.dot(self.zb.T, self.v) * (x - gamma_tt * xp)
        self.v = self.v + v_update

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
            self._writer.add_scalar(root_name + 'v_update_norm', np.linalg.norm(v_update), log_step)
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
            self._writer.add_histogram(root_name + 'v', self.v, log_step)
            self._writer.add_histogram(root_name + 'v_update', v_update, log_step)

