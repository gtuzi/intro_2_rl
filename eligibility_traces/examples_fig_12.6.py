"""
    Run the experiment shown on Fig 12.6
"""
from joblib import Parallel, delayed
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


def run_one_experiment(model_type: str, alpha, lam, n_states, n_episodes, true_values, gamma=0.99):
    """
    Run exactly ONE “experiment”:
    - instantiate a fresh MRPX environment
    - instantiate an OfflineLambdaReturn with (alpha, lam)
    - run for n_episodes, collecting RMS error at episode end
    - return the *average* RMS‐error over those n_episodes
    """
    env = MRPX(n_states)

    if model_type == 'offline_lambda':
        estimator = OfflineLambdaReturn(
            alpha=alpha,
            lam=lam,
            gamma=gamma,
            n_states=n_states + 1  # +1 if MRPX reserves an extra terminal index
        )

    elif model_type == 'td_lambda':
        estimator = TD_lambda(
            alpha=alpha,
            lam=lam,
            gamma=gamma,
            n_states=n_states + 1  # +1 for terminal in MRPX
        )
    else:
        raise NotImplemented

    rms_errors = []
    for _ in range(n_episodes):
        # pick a random start state in [0 .. n_states‐1]
        sinit = np.random.randint(0, n_states)
        s = env.reset(initial_state=sinit)
        done = False

        while not done:
            r, sp, done = env.step()
            estimator.step(s=s, r=r, sp=sp, done=done)
            s = sp

        # at episode end, compute RMS‐error across all non‐terminal states
        se = []
        for state in range(n_states):
            se.append((true_values[state] - estimator.v_fn(state)) ** 2)
        rms_errors.append(np.sqrt(np.mean(se)))

    # return the mean RMS error over all n_episodes
    return float(np.mean(rms_errors))


def return_figure_parallel(
        model_type: str,
        alphas,
        lambdas,
        n_experiments: int,
        n_episodes: int = 10
):
    """
    Exactly the same overall structure as your original, except:
      – We precompute `true_values` once.
      – We loop over (λ,α) combinations in serial (cheap).
      – Inside each (λ,α) pair, we launch all n_experiments *in parallel*.

    Returns
    -------
    results_array : np.ndarray of shape (len(lambdas), len(alphas))
        entry [il, ia] = average RMS‐error over n_experiments, for λ=lambdas[il], α=alphas[ia].
    """

    n_states = 19
    # 1) Precompute “ground truth” state‐values once (ensemble of 50 ests)
    true_values = MRPX.estimate_state_values(
        n_states=n_states,
        num_experiments=500
    )

    # Prepare a 2D array to store final means
    results_array = np.zeros((len(lambdas), len(alphas)), dtype=np.float64)

    # 2) Loop (λ, α) in serial (this is cheap: just n_states‐grid combinations)
    for ia, alpha in enumerate(alphas):
        for il, lam in enumerate(lambdas):
            # 3) Launch n_experiments calls of run_one_experiment(...) *in parallel*
            #    Each call returns a single‐experiment‐average‐RMS.
            #    We use n_jobs=-1 to utilize all CPU cores by default.
            single_runs = Parallel(n_jobs=-1)(
                delayed(run_one_experiment)(
                    model_type=model_type,
                    alpha=alpha,
                    lam=lam,
                    n_states=n_states,
                    n_episodes=n_episodes,
                    true_values=true_values,
                    gamma=0.99
                )
                for _ in range(n_experiments)
            )

            # 4) Average those n_experiments results to fill results_array
            results_array[il, ia] = np.mean(single_runs)

            print(f"{model_type}: Done α={alpha:.3f}, λ={lam:.3f} → {results_array[il, ia]:.4f}")

    return results_array



if __name__ == '__main__':
    alphas = np.linspace(0, 1, num=50)
    lambdas = [0., .4, .8, .9, .95, .975, .99, 1.]

    results = return_figure_parallel(
        model_type='offline_lambda',
        alphas=alphas,
        lambdas=lambdas,
        n_experiments=100,
        n_episodes=10
    )

    plot_multi_curves(
        results,
        title='Offline λ-Return',
        labels=lambdas,
        x=alphas)

    results = return_figure_parallel(
        model_type='td_lambda',
        alphas=alphas,
        lambdas=lambdas,
        n_experiments=100,
        n_episodes=10
    )

    plot_multi_curves(
        results,
        title='TD(λ)',
        labels=lambdas,
        x=alphas)

    exit(0)

