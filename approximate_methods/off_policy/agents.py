from typing import Union, Callable, Any, Optional
import numpy as np

from approximate_methods.utils import (
    LinearQEpsGreedyAgent,
    NoiseSchedule,
    Experience)
from shared.utils import LinearSchedule

from torch.utils.tensorboard import SummaryWriter


#######################################
# ----------- n-step ---------------- #
#######################################


class SemiGradient_nStepsSarsaOffPolicy(LinearQEpsGreedyAgent):

    def __init__(
            self,
            feature_size: int,
            action_space_dims: int,
            update_coefficient: Union[float, NoiseSchedule],
            feature_fn: Callable[[Any, int], np.ndarray], # state, action(int) --> np.ndarray
            nstep_sarsa: int,
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
        self.nstep_sarsa = nstep_sarsa
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
        self.trajectory = []
        self.t = 0

        if isinstance(self.eps, NoiseSchedule):
            # Reset noise to starting exploration
            self.eps.initialize()

        if isinstance(self.update_coefficient, LinearSchedule):
            self.update_coefficient.initialize()

        self.init_weights()

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0
        self.trajectory = []

        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()

        if isinstance(self.update_coefficient, LinearSchedule):
            self.update_coefficient.reset()

    def step(self, e: Experience, **kwargs):

        self.trajectory.append(e)

        tau = self.t - self.nstep_sarsa + 1

        # If the episode ends before n-steps have been rolled out
        if e.done and (tau < 0):
            tau = 0

        if tau >= 0:
            self.update(tau, **kwargs)

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

        self.t += 1

    def update(self, tau: int, **kwargs):
        log_step = None
        if 'log_step' in kwargs:
            log_step = kwargs['log_step']

        # starting from min(n-steps, T/done) back
        tau_end = min(tau+self.nstep_sarsa, len(self.trajectory))

        # -------------------------------------------------------- #
        # G_{t:t+n} = sum_{k=t}^{t+n-1}{\gamma ^{k - t} R_{k + 1}} + Q(S_{t+n}, A_{t+n})
        # G1: sum_{k=t}^{t+n-1}{\gamma ^{k - t} R_{k + 1}}
        # G2: \gamma^{n}q(S_{t+n}, A_{t+n})
        # G_{t:t+n} = G1 + G2
        # -------------------------------------------------------- #

        # G1 = sum_{k = t}^{t+n}{R_{k}}
        target = sum(
            [
                # Ref: 7.3 algorithm box, Eq: 7.4
                # R_{t+1} <- (A_t, S_t)
                (self.discount ** i) * e.r
                for i, e in enumerate(self.trajectory[tau:tau_end])
            ]
        )

        #
        # Episodic: tau + n < T
        # G2: \gamma^{n}q(S_{t+n}, A_{t+n})
        experience_tau_end = self.trajectory[tau_end - 1]

        qhat_next = None
        if not experience_tau_end.done:
            # Episode not terminated
            qhat_next = (
                    (self.discount ** self.nstep_sarsa) *
                    self.state_action_value(
                        experience_tau_end.sp, experience_tau_end.ap
                    )
            )

            target += qhat_next  # Eq: 7.4

        rho = [
            self.get_sa_probability(e.s, e.a) / e.p
            for e in self.trajectory[tau+1:tau_end]
        ]

        if not experience_tau_end.done:
            # Add the \rho for Q(S_{t+n}, A_{t+n})
            rho.append(self.get_sa_probability(
                experience_tau_end.sp,
                experience_tau_end.ap
            ) / experience_tau_end.pp)

        # rho = []
        # for e in self.trajectory[tau:tau_end]:
        #     p = self.get_sa_probability(e.sp, e.ap)
        #     rho.append(np.clip(p / e.pp, 1e-3, 1.))

        rho_prod = None
        if len(rho) > 0:
            # rho_prod = np.prod(rho)  # (7.10)
            rho_prod = 1.  # Force On-Policy updates
        else:
            assert experience_tau_end.done
            rho_prod = 1. # On-policy


        # --- Learn: Eq: (11.6) --- #
        qhat =  self.state_action_value(
            self.trajectory[tau].s,
            self.trajectory[tau].a
        )

        td_error = target - qhat

        grad_w = self.feature_fn(
            self.trajectory[tau].s,
            self.trajectory[tau].a
        )

        if isinstance(self.update_coefficient, LinearSchedule):
            alpha = self.update_coefficient.value
            self.update_coefficient.step()
        elif isinstance(self.update_coefficient, float):
            alpha = self.update_coefficient
        else:
            raise Exception("Invalid type for update_coefficient")

        assert rho_prod is not None, "rho_prod is None"

        update = alpha * rho_prod * td_error * grad_w
        self.w += update

        # Logs
        if (self._writer is not None) and (log_step is not None):
            root_name = f'off_policy/semi_gradient/{self.nstep_sarsa}/sarsa/'
            self._writer.add_scalar(root_name + 'weights_norm', np.linalg.norm(self.w), log_step)
            self._writer.add_scalar(root_name + 'target', target, log_step)
            self._writer.add_scalar(root_name + 'td_error', td_error, log_step)
            self._writer.add_scalar(root_name + 'qhat', qhat, log_step)

            if qhat_next is not None:
                self._writer.add_scalar(root_name + 'qhat_next', qhat_next, log_step)
            self._writer.add_scalar(root_name + 'alpha', alpha, log_step)
            self._writer.add_scalar(root_name + 'grad_w_norm', np.linalg.norm(grad_w), log_step)
            self._writer.add_scalar(root_name + 'rho_prod', rho_prod, log_step)


            self._writer.add_histogram(root_name + 'weights', self.w, log_step)
            self._writer.add_histogram(root_name + 'grad_w', grad_w, log_step)
            self._writer.add_histogram(root_name + 'update', update, log_step)

            if len(rho) > 0:
                self._writer.add_histogram(
                    root_name + 'rho', np.array(rho).reshape(-1), log_step)

    # def update_per_decision(self, tau: int):
    #     # starting from min(n-steps, T/done) back
    #     tau_end = min(tau + self.nstep_sarsa, len(self.trajectory))
    #
    #     rhos = [
    #         self.get_sa_probability(e.s, e.a) / e.p
    #         for e in self.trajectory[tau:tau_end]
    #     ]
    #
    #     rewards = [
    #         # Notationally, in the book, for a[t], reward is r[t+1].
    #         # So while the book starts the accumulation of rewards at
    #         # tau+1, this means that tau+1 indexes the
    #         # (s[tau], a[tau], r[tau+1], s[tau+1]) experience
    #         (self.discount ** i) * e.r
    #         for i, e in enumerate(self.trajectory[tau:tau_end])
    #     ]
    #
    #     # --- Per-decision target (~G[tau]) ---- #
    #     # G[tau] = sum(~R[tau:tau_end] + Q(s'[tau_end], a'[tau_end]))
    #     target = sum([rh * re for rh, re in zip(rhos, rewards)])
    #
    #     experience_tau_end = self.trajectory[tau_end-1]
    #
    #     # If this is the last experience in the episode
    #     if not experience_tau_end.done:
    #         # Episode not terminated
    #         rho_end = self.get_sa_probability(experience_tau_end.sp, experience_tau_end.ap) / experience_tau_end.pp
    #         target += rho_end * (self.discount ** self.nstep_sarsa) * self.state_action_value(experience_tau_end.sp, experience_tau_end.ap)
    #
    #     # --- TD Method --- #
    #
    #     td_error = (target - self.state_action_value(
    #         self.trajectory[tau].s, self.trajectory[tau].a))
    #
    #     # ---- Learn ---- #
    #     grad_w = self.feature_fn(self.trajectory[tau].s, self.trajectory[tau].a)
    #
    #     if isinstance(self.update_coefficient, LinearSchedule):
    #         alpha = self.update_coefficient.value
    #         self.update_coefficient.step()
    #     elif isinstance(self.update_coefficient, float):
    #         alpha = self.update_coefficient
    #     else:
    #         raise Exception("Invalid type for update_coefficient")
    #
    #     grad_w = np.clip(grad_w, -0.1, 0.1)
    #
    #     self.w += alpha * td_error * grad_w
    #
    #     if (self.t % 10) == 0:
    #         SemiGradient_nStepsSarsaOffPolicy.writer_T += 1
    #         t = SemiGradient_nStepsSarsaOffPolicy.writer_T
    #
    #         self.writer.add_histogram('weights', self.w, t)
    #         self.writer.add_histogram('grad_w', grad_w, t)
    #         self.writer.add_scalar('grad_w_norm', np.linalg.norm(self.w), t)
    #         self.writer.add_scalar('weights_norm', np.linalg.norm(grad_w), t)
    #         self.writer.add_scalar('td_error', td_error, t)
    #         self.writer.add_scalar('target', target, t)
    #         self.writer.add_scalar('rho', np.prod(rhos), t)
    #         self.writer.add_scalar('alpha', alpha, t)
