"""
    Run the experiment shown on Fig 12.6
"""

from itertools import product
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from random_walk_mrp import MRPX

#region plots
def plot_plain(data, y_label: str):
    plt.plot(data)
    plt.xlabel('Time step')
    plt.ylabel(y_label)
    plt.show()

def plot_multi_curves(
        curve_values,
        labels,
        x=None,
        title='TD(λ)',
        xlabel='α',
        ylabel='RMS Error @ End of Episode'
):
    """
        Plot multiple curves on the same axes and label each near their minimum y-point,
        slightly offset to avoid overlap.

    Parameters
    ----------
    curve_values : List[List[float]]
        A list of sequences, each containing the y-values of a curve.
    labels : List[str]
        A list of strings, one for each curve, e.g. ['0', '0.4', '0.8', '0.9'] for λ values.
    x : array-like of shape (M,), optional
        The common x-axis values for all curves. If None, will use linspace(0,1,M).
    title : str, default 'TD(λ)'
        Title of the plot.
    xlabel : str, default 'α'
        Label for the x-axis.
    ylabel : str, default 'Error'
        Label for the y-axis.
    """
    n_curves = len(curve_values)
    if x is None:
        M = len(curve_values[0])
        x = np.linspace(0, 1, M)

    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_curves))

    for idx, (y, lab) in enumerate(zip(curve_values, labels)):
        plt.plot(x, y, color=colors[idx], lw=2)

        # Annotate slightly above the minimum y-value to avoid overlap
        min_idx = np.argmin(y)
        x_annotate = x[min_idx]
        y_annotate = y[min_idx]
        plt.text(x_annotate, y_annotate - 0.015, f"λ={lab}",
                 color=colors[idx], fontsize=15, va='top', ha='left')

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xlim(x[0], x[-1])
    plt.ylim(0.25, 0.55)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

#endregion plots


def feature_extractor(s, n_states):
    """ Just one-hot the state """
    assert 0 <= s < n_states
    x = np.zeros(n_states, dtype=np.float32)
    x[s] = 1.
    return x


class TD_lambda:
    def __init__(
            self,
            alpha,
            lam,
            gamma: float = 0.99,
            n_states: int = 6):

        self.n_states = n_states
        self.v = None
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

def td_lamba_figure(
        alphas,
        lambdas,
        n_experiments: int,
        n_episodes = 10
):
    n_states = 19
    # Just numerically estimate the values for each state
    # if starting from that state
    true_values = MRPX.estimate_state_values(
        n_states=n_states,
        num_experiments=500
    )

    results = {
        (l, a): list() for l, a in product(lambdas, alphas)
    }

    results_array = np.zeros((len(lambdas), len(alphas)))

    for ia, a in enumerate(alphas):

        for il, l in enumerate(lambdas):

            for _ in tqdm(range(n_experiments), desc=f'alpha={a:.2f}, lambda={l: .2f}'):

                env = MRPX(n_states)  # n_states random walk

                estimator = TD_lambda(
                    alpha=a,
                    lam=l,
                    gamma=0.99,
                    n_states=n_states + 1  # +1 for terminal in MRPX
                )

                terminal_rewards = []
                errors = []

                for episode in range(n_episodes):
                    sinit = np.random.randint(0, n_states)
                    s = env.reset(initial_state=sinit)

                    done = False

                    while not done:
                        r, sp, done = env.step()
                        estimator.step(s=s, r=r, sp=sp, done=done)
                        s = sp

                    terminal_rewards.append(r)

                    # Figure 12.6: RMS error
                    # at the end
                    # of the episode
                    # over the first
                    # 10 episodes
                    state_errors = [
                        (true_values[s] - estimator.v_fn(s)) ** 2
                        for s in range(n_states)
                    ]

                    errors.append(np.sqrt(np.mean(state_errors)))

                results[(l, a)].append(np.mean(errors))

            # Average over experiments
            results_array[il, ia] = np.mean(results[(l, a)])

    return results_array


if __name__ == '__main__':
    alphas = np.linspace(0, 1, num=50)
    lambdas = [.4, .8, .9, .95, .975, .99, 1.]

    results = td_lamba_figure(
        alphas=alphas,
        lambdas=lambdas,
        n_experiments = 100,
        n_episodes=10
    )

    plot_multi_curves(results, labels=lambdas, x=alphas)

    exit(0)

