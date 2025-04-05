from typing import Union, Callable, Any
import numpy as np
from sympy import ceiling
import time

from approximate_methods.utils import (
    LinearQEpsGreedyAgent,
    NoiseSchedule,
    Experience)
from shared.utils import LinearSchedule

from torch.utils.tensorboard import SummaryWriter


class SemiGradient_nStepsSarsaOffPolicy(LinearQEpsGreedyAgent):
    writer_T = 0

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



    def __del__(self):
        try:
            self.writer.close()
        except:
            pass

    def initialize(self):
        self.trajectory = []
        self.t = 0

        # Create a SummaryWriter instance
        _c = current_time = int(time.time())
        self.writer = SummaryWriter(f'runs/nstep_offpolicy/{_c}')
        SemiGradient_nStepsSarsaOffPolicy.writer_T = 0

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


    def step(self, e: Experience):

        self.trajectory.append(e)

        tau = self.t - self.nstep_sarsa + 1

        # If the episode ends before n-steps have been rolled out
        if e.done and (tau < 0):
            tau = 0

        if tau >= 0:
            self.update(tau)

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

        self.t += 1

    def update(self, tau: int):

        # starting from min(n-steps, T/done) back
        tau_end = min(tau+self.nstep_sarsa, len(self.trajectory))

        target = sum(
            [
                # Ref: 7.3 algorithm box
                # (On-policy n-step Sarsa for estimating Q=q* or q_pi)
                # Notationally, for a[t], the book uses reward as r[t+1].
                # So while the sample target (G) starts the accumulation
                # of rewards at tau+1, this means that tau+1 indexes the reward
                # in the (s[tau], a[tau], r[tau+1], s[tau+1]) experience.
                # So here - G = sum_i(tau + 1):min(tau + n, T)(R[i])
                # Eq: 7.4
                (self.discount ** i) * e.r
                for i, e in enumerate(self.trajectory[tau:tau_end])
            ]
        )

        # Episodic: tau + n < T
        experience_tau_end = self.trajectory[tau_end - 1]
        if not experience_tau_end.done:
            # Episode not terminated
            target += (
                    (self.discount ** self.nstep_sarsa) *
                    self.state_action_value(
                        experience_tau_end.sp, experience_tau_end.ap)
            ) # Eq: 7.4


        # Refer to the definition of G at (7.4) and rho at (7.10)
        # G_t:t+h = R[t+1] + ...
        # rho_t:t+h = prod(pi(a[t] | s[t])/b(a[t] | s[t]), ...)
        #
        # Rho, however, is computed for one (a,s) ahead of the current rho
        # In a way, rho is looking at the (a|s) following the r[tau+1],
        # i.e a[tau+1] and s[tau + 1]
        # But since we don't have any (a, s) following r[tau + n_steps] this
        # is truncated one step earlier.

        rho = [
            self.get_sa_probability(e.sp, e.ap) / e.pp
            for e in self.trajectory[tau:min(tau_end, len(self.trajectory) - 1)]
        ]

        # rho = []
        # for e in self.trajectory[tau:tau_end]:
        #     p = self.get_sa_probability(e.sp, e.ap)
        #     rho.append(np.clip(p / e.pp, 1e-3, 1.))


        if len(rho) > 0:
            rho_prod = np.prod(rho)  # (7.10)
        else:
            rho_prod = 1.

        # --- Learn: Eq: (11.6) --- #
        experience_tau = self.trajectory[tau]
        td_error = target - self.state_action_value(
            experience_tau.s, experience_tau.a)

        grad_w = self.feature_fn(experience_tau.s, experience_tau.a)

        if isinstance(self.update_coefficient, LinearSchedule):
            alpha = self.update_coefficient.value
            self.update_coefficient.step()
        elif isinstance(self.update_coefficient, float):
            alpha = self.update_coefficient
        else:
            raise Exception("Invalid type for update_coefficient")

        self.w += alpha * rho_prod * td_error * grad_w

        # Logs
        if (self.t % 10) == 0:
            SemiGradient_nStepsSarsaOffPolicy.writer_T += 1
            t = SemiGradient_nStepsSarsaOffPolicy.writer_T

            self.writer.add_histogram('weights', self.w, t)
            self.writer.add_histogram('grad_w', grad_w, t)
            self.writer.add_histogram('rho', rho, t)
            self.writer.add_scalar('grad_w_norm', np.linalg.norm(self.w), t)
            self.writer.add_scalar('weights_norm', np.linalg.norm(grad_w), t)
            self.writer.add_scalar('td_error', td_error, t)
            self.writer.add_scalar('target', target, t)
            self.writer.add_scalar('rho_prod', rho_prod, t)
            self.writer.add_scalar('alpha', alpha, t)


    def update_per_decision(self, tau: int):
        # starting from min(n-steps, T/done) back
        tau_end = min(tau + self.nstep_sarsa, len(self.trajectory))

        rhos = [
            self.get_sa_probability(e.s, e.a) / e.p
            for e in self.trajectory[tau:tau_end]
        ]

        rewards = [
            # Notationally, in the book, for a[t], reward is r[t+1].
            # So while the book starts the accumulation of rewards at
            # tau+1, this means that tau+1 indexes the
            # (s[tau], a[tau], r[tau+1], s[tau+1]) experience
            (self.discount ** i) * e.r
            for i, e in enumerate(self.trajectory[tau:tau_end])
        ]

        # --- Per-decision target (~G[tau]) ---- #
        # G[tau] = sum(~R[tau:tau_end] + Q(s'[tau_end], a'[tau_end]))
        target = sum([rh * re for rh, re in zip(rhos, rewards)])

        experience_tau_end = self.trajectory[tau_end-1]

        # If this is the last experience in the episode
        if not experience_tau_end.done:
            # Episode not terminated
            rho_end = self.get_sa_probability(experience_tau_end.sp, experience_tau_end.ap) / experience_tau_end.pp
            target += rho_end * (self.discount ** self.nstep_sarsa) * self.state_action_value(experience_tau_end.sp, experience_tau_end.ap)

        # --- TD Method --- #

        td_error = (target - self.state_action_value(
            self.trajectory[tau].s, self.trajectory[tau].a))

        # ---- Learn ---- #
        grad_w = self.feature_fn(self.trajectory[tau].s, self.trajectory[tau].a)

        if isinstance(self.update_coefficient, LinearSchedule):
            alpha = self.update_coefficient.value
            self.update_coefficient.step()
        elif isinstance(self.update_coefficient, float):
            alpha = self.update_coefficient
        else:
            raise Exception("Invalid type for update_coefficient")

        grad_w = np.clip(grad_w, -0.1, 0.1)

        self.w += alpha * td_error * grad_w

        if (self.t % 10) == 0:
            SemiGradient_nStepsSarsaOffPolicy.writer_T += 1
            t = SemiGradient_nStepsSarsaOffPolicy.writer_T

            self.writer.add_histogram('weights', self.w, t)
            self.writer.add_histogram('grad_w', grad_w, t)
            self.writer.add_scalar('grad_w_norm', np.linalg.norm(self.w), t)
            self.writer.add_scalar('weights_norm', np.linalg.norm(grad_w), t)
            self.writer.add_scalar('td_error', td_error, t)
            self.writer.add_scalar('target', target, t)
            self.writer.add_scalar('rho', np.prod(rhos), t)
            self.writer.add_scalar('alpha', alpha, t)

