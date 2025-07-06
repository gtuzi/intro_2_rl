import copy
from copy import deepcopy
from os import wait3

import numpy as np


def feature_extractor(s, n_states):
    """ Just one-hot the state """
    assert 0 <= s < n_states
    x = np.zeros(n_states, dtype=np.float32)
    x[s] = 1.
    return x


class TDLambda:
    """
        Section: 12.2
    """
    def __init__(
            self,
            alpha,
            lam,
            gamma: float = 0.99,
            n_states: int = 6):

        self.n_states = n_states
        self.w = None
        self.z = None
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha

        self.reset_weights()
        self.t = 0

    def reset(self):
        self.t = 0
        self.z = np.zeros_like(self.w)

    def reset_weights(self):
        self.w = np.ones(self.n_states, dtype=np.float32) * 0.5 # Per example 7.1
        self.z = np.zeros_like(self.w)

    def v_fn(self, s):
        x = feature_extractor(s, self.n_states)
        return np.dot(x, self.w)

    def step(self, s, r, sp, done):

        x = feature_extractor(s, self.n_states)
        dv = x

        # Eligibility trace
        self.z = self.lam * self.gamma * self.z + dv

        # TD error
        tde = (r + self.gamma * self.v_fn(sp) * (1 - done)) - self.v_fn(s)

        # Update weights
        self.w = self.w + self.alpha * tde * self.z


class TTDLambda:
    """
        Section: 12.3
    """
    def __init__(
            self,
            alpha,
            lam,
            n_steps,
            gamma: float = 0.99,
            n_states: int = 6):

        self.n_states = n_states
        self.w = None
        self.wold = None
        self.z = None
        self.n_steps = n_steps
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha
        self.buffer = []

        self.reset_weights()
        self.t = 0

    def reset(self):
        self.t = 0
        self.z = np.zeros_like(self.w)
        self.buffer = []

    def reset_weights(self):
        self.t = 0
        self.w = np.ones(self.n_states, dtype=np.float32) * 0.5 # Per example 7.1
        self.wold = copy.deepcopy(self.w)
        self.z = np.zeros_like(self.w)
        self.buffer = []

    def v_fn(self, s):
        x = feature_extractor(s, self.n_states)
        return np.dot(x, self.w)

    def step(self, s, r, sp, done):
        self.t += 1
        x = feature_extractor(s, self.n_states)
        xp = feature_extractor(sp, self.n_states)

        # Note the different weights being used here
        vold = np.dot(x, self.wold)
        v = np.dot(x, self.w)
        vp = np.dot(xp, self.w)

        td_error = r + self.gamma * vp * (1 - done) - vold
        self.buffer.append(dict(td_error=td_error, v=v, vold=vold, vp=vp, grad_w=x))

        if self.t >= self.n_steps:
            self.learn()
            _ = self.buffer.pop(0)  # Old falls outside n-th step

        while done and (len(self.buffer) > 0):
            self.learn()
            _ = self.buffer.pop(0)

    def learn(self):
        v, vold, grad_w =  (
            self.buffer[0]['v'],
            self.buffer[0]['vold'],
            self.buffer[0]['grad_w']
        )

        G = vold + sum([
            ((self.gamma * self.lam) ** i) * be['td_error']
            for i, be in enumerate(self.buffer)
        ])

        # Update weights
        self.wold = copy.deepcopy(self.w)
        self.w += self.alpha * (G - v) * grad_w


class OfflineLambdaReturn:
    def __init__(
            self,
            gamma: float,
            alpha: float,
            lam: float,
            n_states: int
    ):
        assert 0 <= lam <= 1, f'Expected: 0 <= lam <= 1, got lam={lam}'
        assert 0 <= alpha <= 1, f'Expected: 0 <= alpha <= 1, got lam={alpha}'
        assert 0 <= gamma <= 1, f'Expected: 0 <= gamma <= 1, got lam={gamma}'
        assert n_states > 0

        self.n_states = n_states
        self.gamma = gamma
        self.alpha = alpha
        self.lam = lam
        self.w = None
        self.reset_weights()
        self.t = 0
        self.buffer = []

    def reset_weights(self):
        self.w = np.ones(self.n_states, dtype=np.float32) * 0.5 # Per example 7.1

    def reset(self):
        self.t = 0
        self.buffer.clear()

    def v_fn(self, s):
        x = feature_extractor(s, self.n_states)
        return np.dot(x, self.w)

    def step(self, s, r, sp, done):
        self.buffer.append((s, r, sp, done))
        if done:
            self.learn()
            self.buffer.clear()

    def learn(self):
        T = len(self.buffer)

        def Gt_fn(t):
            assert t >= 0

            if t < T:
                return sum(
                    [(self.gamma ** i) * r for i, (s, r, sp, done) in
                     enumerate(self.buffer[t:])]
                )
            else:
                return 0.

        def Gt_n_fn(t, n):
            Gt_r = sum([
                (self.gamma ** i) * r
                for i, (s, r, sp, done) in enumerate(self.buffer[t:t + n])]
            )

            return Gt_r + (self.gamma ** n) * self.v_fn(self.buffer[t + n][0])


        for t in range(T):

            # ---- (12.3) ----
            Gtlam = (1. - self.lam) * sum(
                [
                    ((self.lam) ** (n - 1)) * Gt_n_fn(t, n)
                    for n in range(1, T - t)
                ]
            ) + (self.lam ** (T - t - 1)) * Gt_fn(t)
            # --------------

            s = self.buffer[t][0]
            v = self.v_fn(s)
            grad_w = feature_extractor(s, n_states=self.n_states)
            self.w += self.alpha * (Gtlam - v) * grad_w


class OnlineLambdaReturn:
    def __init__(
            self,
            gamma: float,
            alpha: float,
            lam: float,
            n_states: int
    ):
        assert 0 <= lam <= 1, f'Expected: 0 <= lam <= 1, got lam={lam}'
        assert 0 <= alpha <= 1, f'Expected: 0 <= alpha <= 1, got lam={alpha}'
        assert 0 <= gamma <= 1, f'Expected: 0 <= gamma <= 1, got lam={gamma}'
        assert n_states > 0

        self.n_states = n_states
        self.gamma = gamma
        self.alpha = alpha
        self.lam = lam
        self.w = None
        self.reset_weights()
        self.t = 0
        self.buffer = []

    def reset_weights(self):
        self.w = np.ones(self.n_states, dtype=np.float32) * 0.5 # Per example 7.1
        self.w0 = deepcopy(self.w)

    def reset(self):
        self.t = 0
        self.buffer.clear()

    def v_fn(self, s):
        x = feature_extractor(s, self.n_states)
        return np.dot(x, self.w)

    def step(self, s, r, sp, done):
        self.buffer.append((s, r, sp, done))

        self.learn()

        if done:
            self.buffer.clear()
            # The weights at the end of this episode become the
            # first for the next episode.
            self.w0 = deepcopy(self.w)

    def learn(self):
        h = len(self.buffer)

        # The first weight vector w^{h0} in each sequence is that
        # inherited from the previous episode
        self.w = deepcopy(self.w0)

        def Gt_fn(t):
            assert t >= 0

            if t < h:
                return sum(
                    [(self.gamma ** i) * r for i, (s, r, sp, done) in
                     enumerate(self.buffer[t:])]
                )
            else:
                return 0.

        def Gt_n_fn(t, n):
            Gt_r = sum([
                (self.gamma ** i) * r
                for i, (s, r, sp, done) in enumerate(self.buffer[t:t + n])]
            )

            return Gt_r + (self.gamma ** n) * self.v_fn(self.buffer[t + n][0])


        for t in range(h):

            # ---- (12.3) ----
            Gtlam = (1. - self.lam) * sum(
                [
                    ((self.lam) ** (n - 1)) * Gt_n_fn(t, n)
                    for n in range(1, h - t)
                ]
            ) + (self.lam ** (h - t - 1)) * Gt_fn(t)
            # --------------

            s = self.buffer[t][0]
            v = self.v_fn(s)
            grad_w = feature_extractor(s, n_states=self.n_states)
            self.w += self.alpha * (Gtlam - v) * grad_w


class OnlineTDLambda:
    def __init__(
            self,
            alpha,
            lam,
            gamma: float = 0.99,
            n_states: int = 6):
        self.n_states = n_states
        self.v_old = None
        self.w = None
        self.z = None
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha

        self.reset_weights()
        self.t = 0

    def reset(self):
        self.t = 0
        self.z = np.zeros_like(self.w)
        self.v_old = 0

    def reset_weights(self):
        self.w = np.ones(self.n_states, dtype=np.float32) * 0.5 # Per example 7.1
        self.z = np.zeros_like(self.w)
        self.v_old = 0

    def v_fn(self, s):
        x = feature_extractor(s, self.n_states)
        return np.dot(x, self.w)

    def step(self, s, r, sp, done):

        v = self.v_fn(s)
        vp = self.v_fn(sp)

        delta = (r + self.gamma * vp) - v

        x = feature_extractor(s, self.n_states)

        # Eligibility trace
        self.z = self.lam * self.gamma * self.z + (1 - self.alpha * self.lam * self.gamma * np.matmul(self.z.T, x)) * x

        # Update weights
        self.w = self.w + self.alpha * (delta + v - self.v_old) * self.z - self.alpha * (v - self.v_old) * x

        self.v_old = vp
