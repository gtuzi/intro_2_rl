from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy


class BairdsCounterExampleEnvironment:
    def __init__(self, max_steps=None):
        self.num_states = 7
        self.num_actions = 2
        self.current_state = None
        self.reward = 0  # Reward is always 0
        self.t = 0
        self.terminal = False
        self.max_steps = max_steps

        self.action_to_name = ['solid_line', 'dashed_line']

        # Solid line
        self.a0_transitions = [
            [0., 0., 0, 0., 0., 0., 1.],  # S0
            [0., 0., 0, 0., 0., 0., 1.],  # S1
            [0., 0., 0, 0., 0., 0., 1.],  # S2
            [0., 0., 0, 0., 0., 0., 1.],  # S3
            [0., 0., 0, 0., 0., 0., 1.],  # S4
            [0., 0., 0, 0., 0., 0., 1.],  # S5
            [0., 0., 0, 0., 0., 0., 1.],  # S6
        ]

        # Dashed line
        self.a1_transitions = [
            [1., 0., 0, 0., 0., 0., 0.],  # S0
            [0., 1., 0, 0., 0., 0., 0.],  # S1
            [0., 0., 1, 0., 0., 0., 0.],  # S2
            [0., 0., 0, 1., 0., 0., 0.],  # S3
            [0., 0., 0, 0., 1., 0., 0.],  # S4
            [0., 0., 0, 0., 0., 1., 0.],  # S5
            [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 0.],  # S6
        ]

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_state = np.random.randint(0, self.num_states)
        self.t = 0
        self.terminal = False
        return int(self.current_state)

    def step(self, action: int):

        if not self.terminal:
            self.t += 1

            if action == 0:
                trans_probs = self.a0_transitions[self.current_state]
            elif action == 1:
                trans_probs = self.a1_transitions[self.current_state]
            else:
                raise RuntimeError('Invalid action')

            next_state = int(np.random.choice(self.num_states, p=trans_probs))

            # if next_state == 6:
            #     self.terminal = True

            if self.t == self.max_steps:
                self.terminal = True

            return next_state, self.reward, self.terminal

        return self.current_state, self.reward, self.terminal

    def get_transition(self, s, a):
        if a == 0:
            return self.a0_transitions[s]
        elif a == 1:
            return self.a1_transitions[s]
        else:
            raise RuntimeError('Invalid action')


def feature_extractor(s):
    # 7 states, 1 bias
    x = np.zeros(8, dtype=np.float32)
    if 0 <= s <= 5:
        x[s] = 2.
        x[7] = 1.
    elif s == 6:
        x[6] = 1.
        x[7] = 2
    else:
        raise RuntimeError('Invalid state')
    return x


class Pi_Sarsa:

    def __init__(
            self,
            gamma: float = 0.99,
            alpha: float = 0.01,
            assume_on_policy: bool = False,
    ):
        self.reset_weights()
        self.gamma = gamma
        self.alpha = alpha
        self.assume_on_policy = assume_on_policy

    def reset_weights(self):
        w1 = np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=np.float32)
        self.w = w1

    def reset(self):
        self.reset_weights()

    def act(self, s):
        # pi(solid_line|*) = 1
        return 0

    def v_fn(self, s):
        x = feature_extractor(s)
        return np.dot(x, self.w)

    def step(self, s, a, r, sp, ap):

        # Update weights for v_hat(, w)

        if self.assume_on_policy:
            rho = 1.
        else:
            rho = 1/(1/7) if a == 0 else 0.

        v = self.v_fn(s)
        vp = self.v_fn(sp)
        tgt = r + self.gamma * vp
        td_err = tgt - v
        grad_w = feature_extractor(s)
        self.w += self.alpha * rho * td_err * grad_w


class Pi_QLearning:
    def __init__(
            self,
            gamma: float = 0.99,
            alpha: float = 0.01,
    ):
        self.reset_weights()
        self.gamma = gamma
        self.alpha = alpha

    def reset_weights(self):
        w1 = np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=np.float32)
        w2 = np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=np.float32)
        self.w = np.array([w1, w2])

    def reset(self):
        self.reset_weights()

    def act(self, s):
        # pi(solid_line|*) = 1
        return 0

    def q_fn(self, s, a):
        x = feature_extractor(s)
        return np.dot(x, self.w[a, ...].squeeze())

    def step(self, s, a, r, sp, ap):
        q = self.q_fn(s, a)
        qp = max([self.q_fn(sp, a) for a in [0, 1]])
        tgt = r + self.gamma * qp
        td_err = tgt - q
        grad_w = feature_extractor(s)

        self.w[a, ...] += self.alpha * td_err * grad_w


class Pi_DP:
    def __init__(
            self,
            env,
            gamma: float = 0.99,
            alpha: float = 0.01,
            assume_on_policy: bool = False,
    ):
        self.env = env
        self.reset_weights()
        self.gamma = gamma
        self.alpha = alpha
        self.assume_on_policy = assume_on_policy

    def reset_weights(self):
        w1 = np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=np.float32)
        self.w = w1

    def reset(self):
        self.reset_weights()

    def act(self, s):
        # pi(solid_line|*) = 1
        return 0

    def v_fn(self, s):
        x = feature_extractor(s)
        return np.dot(x, self.w)


    def off_policy_sweep(self):

        # Update weights for v_hat(, w)
        n_states = 7

        for s in range(n_states):
            # sum_a[p(s' | s, a) * p(a | s)] = p(s' | s)
            # But since p(a == 1| *) = 0
            # => p(s' | s) = p(s' | s, a == 0)
            pa = [1, 0]
            vs = []
            for _a in [0, 1]:
                # Sweep over s'
                vs += [
                    gamma * p * pa[_a] * self.v_fn(_sp)
                    for p, _sp in zip(env.get_transition(s, 0), range(n_states))
                ]


            # E[gamma * v(s') | s] = gamma * E[v(s') | s] = Sum_a[p(s', a | s) * v(s')]
            # DP tgt: Sum_s'{R + gamma * E[v(s') | s]} = gamma * Sum_s'{E[v(s') | s]}
            # E[R] = 0

            tgt = 0 + np.sum(vs)

            td_errs = tgt - self.v_fn(s)
            grad_w = feature_extractor(s)

            self.w += (self.alpha / n_states) * np.sum(td_errs) * grad_w

    def on_policy_sweep(self):

        # Update weights for v_hat(, w)
        n_states = 7

        for s in range(n_states):
            # sum_a[p(s' | s, a) * p(a | s)] = p(s' | s)
            # But since p(a == 1| *) = 0
            # => p(s' | s) = p(s' | s, a == 0)

            vs = []
            pa = [1/7, 6/7]  # On-policy
            for a in [0, 1]:
                vs += [
                    gamma * p * pa[a] * self.v_fn(_s)
                    for p, _s in zip(env.get_transition(s, a), range(n_states))
                ]
            # E[R] = 0

            tgt = 0 + np.sum(vs)

            td_errs = tgt - self.v_fn(s)
            grad_w = feature_extractor(s)

            self.w += (self.alpha / n_states) * np.sum(td_errs) * grad_w

    def sweep(self):
        if self.assume_on_policy:
            self.on_policy_sweep()
        else:
            self.off_policy_sweep()


class BehavioralPolicy:

    def __init__(self, gamma: float = 0.99, alpha: float = 0.01):
        self.w = np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=np.float32)
        self.gamma = gamma
        self.alpha = alpha

    def reset(self):
        self.w = np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=np.float32)

    def act(self, s):
        return np.random.choice([0, 1], size=1, p = [1/7, 6/7])

    def step(self, s, a, r, sp, ap):
        x = feature_extractor(s)
        xp = feature_extractor(sp)

        v = np.dot(x, self.w)
        vp = np.dot(xp, self.w)
        tgt = r + self.gamma * vp
        td_err = tgt - v
        grad_w = x
        self.w += self.alpha * td_err * grad_w


def plot_hist(data, states_list=None, root: str = ''):
    """
    Plot a normalized histogram over a known set of integer states.

    Parameters
    ----------
    data : sequence of int
        Observed states (e.g. [0,1,0,0,1,1,1]).
    states_list : sequence of int, optional
        The complete list of states to show (in order).
        If None, will use sorted unique values from data.
    root : str, optional
        Prefix for the plot title.
    """
    data = np.array(data, dtype=int)

    # 1) Determine which states to plot
    if states_list is None:
        states = np.unique(data)
    else:
        states = np.array(states_list, dtype=int)

    # 2) Count & normalize
    counts = np.array([np.sum(data == s) for s in states], dtype=float)
    freqs = counts / counts.sum()

    # 3) Plot
    plt.figure()
    plt.bar(states, freqs, width=0.8, align='center')
    plt.xticks(states)
    plt.xlabel('States')
    plt.ylabel('Probability')
    plt.title(f"{root}")
    plt.tight_layout()
    plt.show()


def plot_vectors_with_grouped_labels(vectors, labels, tol=1e-1, root: str = ''):
    """
    Plots each element of a list of vectors over time,
    and groups labels for end‐values within a tolerance tol.

    Parameters:
    - vectors: list of iterables of equal length (shape T x D)
    - labels: list of length D with names for each dimension
    - tol: float, tolerance for grouping end‐values
    """
    arr = np.array(vectors)
    T, D = arr.shape
    time = np.arange(T)

    # Compute final values and sort their indices
    end_vals = arr[-1, :]
    idx_sorted = np.argsort(end_vals)

    # Cluster indices whose final values differ by <= tol
    clusters = []
    current = [idx_sorted[0]]
    for i in idx_sorted[1:]:
        if abs(end_vals[i] - end_vals[current[-1]]) <= tol:
            current.append(i)
        else:
            clusters.append(current)
            current = [i]
    clusters.append(current)

    # Plot all series
    plt.figure()
    for d in range(D):
        plt.plot(time, arr[:, d])

    # Annotate each cluster at the right edge
    for cluster in clusters:
        if len(cluster) == 1:
            i = cluster[0]
            label = labels[i]
            y = end_vals[i]
        else:
            i_min, i_max = min(cluster), max(cluster)
            label = f"{labels[i_min]}–{labels[i_max]}"
            y = end_vals[cluster].mean()
        plt.text(T - 1 + 0.5, y, label, va='center', fontsize=9)

    plt.xlim(0, T + 1)
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.title(root)
    plt.grid(True)
    plt.show()


def plot_plain(data, y_label: str):
    plt.plot(data)
    plt.xlabel('Time step')
    plt.ylabel(y_label)
    plt.show()


def sample_mdp(env):
    actions = [0, 1]

    for a in actions:
        starting_states = []
        next_states = []

        for _ in range(1000):
            s = env.reset()
            starting_states.append(s)

            done = False

            while (not done) and env.t < 100:
                r, sp, done = env.step(a)
                next_states.append(sp)
                if done:
                    break

        plot_hist(starting_states, root=f'Action {a} - Starting States')
        plot_hist(next_states, root=f'Action {a} - Next States')


def off_policy_sarsa(env, alpha = 0.01, gamma = 0.99, num_episodes = 100, force_on_policy = False):

    pi = Pi_Sarsa(gamma, alpha, assume_on_policy=force_on_policy)
    b = BehavioralPolicy(gamma, alpha)

    weights_over_time = [deepcopy(pi.w)]
    actions_over_time = []
    states_over_time = []
    v_over_time = []

    for e in range(num_episodes):

        s = env.reset()
        a = b.act(s)

        for t in range(env.max_steps):

            states_over_time.append(s)
            sp, r, done = env.step(a)
            ap = b.act(sp)

            pi.step(s, a, r, sp, ap)
            b.step(s, a, r, sp, ap)

            # Collect
            v_over_time.append(pi.v_fn(s))
            weights_over_time.append(deepcopy(pi.w))
            actions_over_time.append(a)

            if done:
                break
            else:
                s = sp
                a = ap

    plot_vectors_with_grouped_labels(
        weights_over_time,
        labels=[f'w{w + 1}' for w in range(8)],
        root=f'Weights'
    )

    # plot_plain(v_over_time, y_label=f'V(s)')
    # plot_hist(actions_over_time, states_list=[0, 1], root='Actions')
    # plot_hist(states_over_time,  root='States Visited')


def off_policy_qlearning(env, alpha = 0.01, gamma = 0.99, num_episodes = 100):

    pi = Pi_QLearning(gamma, alpha)

    b = BehavioralPolicy(gamma, alpha)

    weights0_over_time = [deepcopy(pi.w[0])]
    weights1_over_time = [deepcopy(pi.w[1])]
    actions_over_time = []
    states_over_time = []
    v_over_time = []

    for e in range(num_episodes):

        s = env.reset()
        a = b.act(s)

        for t in range(env.max_steps):

            states_over_time.append(s)
            sp, r, done = env.step(a)
            ap = b.act(sp)

            pi.step(s, a, r, sp, ap)
            b.step(s, a, r, sp, ap)

            # Collect
            v_over_time.append(pi.q_fn(s, a))
            weights0_over_time.append(deepcopy(pi.w[0]))
            weights1_over_time.append(deepcopy(pi.w[1]))
            actions_over_time.append(a)

            if done:
                break
            else:
                s = sp
                a = ap

    plot_vectors_with_grouped_labels(
        weights0_over_time,
        labels=[f'w{w + 1}' for w in range(8)],
        root=f'Weights (Solid Action)'
    )

    plot_vectors_with_grouped_labels(
        weights1_over_time,
        labels=[f'w{w + 1}' for w in range(8)],
        root=f'Weights (Dashed Action)'
    )


def off_policy_dp(env, alpha = 0.01, gamma = 0.99, num_sweeps = 100, force_on_policy = False):

    pi = Pi_DP(env=env, gamma=gamma, alpha=alpha, assume_on_policy=force_on_policy)

    weights_over_time = [deepcopy(pi.w)]

    for e in range(num_sweeps):
        for t in range(env.max_steps):
            pi.sweep()
            # Collect
            weights_over_time.append(deepcopy(pi.w))

    plot_vectors_with_grouped_labels(
        weights_over_time,
        labels=[f'w{w + 1}' for w in range(8)],
        root=f'Weights'
    )


def on_policy(env, alpha = 0.01, gamma = 0.99, num_episodes = 100):
    b = BehavioralPolicy(gamma, alpha)
    next_states_over_time = []
    actions_over_time = []

    for e in range(num_episodes):

        s = env.reset()

        for t in range(env.max_steps):
            a = b.act(s)

            sp, r, done = env.step(a)

            b.step(s, a, r, sp)
            actions_over_time.append(a)
            next_states_over_time.append(sp)

            if done:
                break
            else:
                s = sp

    plot_hist(next_states_over_time)


if __name__ == "__main__":
    T = 2
    num_episodes = 500
    gamma = 0.99
    alpha = 0.01

    env = BairdsCounterExampleEnvironment(T)

    off_policy_qlearning(env, alpha=alpha, gamma=gamma, num_episodes=num_episodes)

    off_policy_sarsa(env, alpha=alpha, gamma=gamma, num_episodes=num_episodes, force_on_policy=True)

    off_policy_dp(env, num_sweeps=num_episodes, gamma=gamma, alpha=alpha, force_on_policy=True)

    exit(0)