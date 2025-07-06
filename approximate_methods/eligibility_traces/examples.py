"""
    Generate experiments and their related figures
"""

from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from typing_extensions import Optional

from algorithms import (
    TDLambda,
    TTDLambda,
    OfflineLambdaReturn,
    OnlineLambdaReturn,
    OnlineTDLambda
)

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


def run_one_experiment(
        model_type: str,
        alpha: float,
        lam: float,
        n_states: int,
        n_episodes: int,
        true_values,
        gamma: float=0.99,
        n_steps: Optional[int] = None,
):
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
        estimator = TDLambda(
            alpha=alpha,
            lam=lam,
            gamma=gamma,
            n_states=n_states + 1  # +1 for terminal in MRPX
        )
    elif model_type == 'ttd_lambda':
        estimator = TTDLambda(
            alpha=alpha,
            lam=lam,
            gamma=gamma,
            n_steps=n_steps,
            n_states=n_states + 1  # +1 for terminal in MRPX
        )
    elif model_type == 'online_lambda':
        estimator = OnlineLambdaReturn(
            alpha=alpha,
            lam=lam,
            gamma=gamma,
            n_states=n_states + 1 # +1 if MRPX reserves an extra terminal index
        )
    elif model_type == 'online_td_lambda':
        estimator = OnlineTDLambda(
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


def experiments_parallel(
        model_type: str,
        alphas,
        lambdas,
        n_experiments: int,
        n_episodes: int = 10,
        n_steps: Optional[int] = None
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
                    n_steps=n_steps,
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

    # TODO: "online_lambda" takes way too long

    for n_steps in [1, 5, 10, 20, 40]:
        results = experiments_parallel(
            model_type='ttd_lambda',
            alphas=alphas,
            lambdas=lambdas,
            n_experiments=100,
            n_episodes=10,
            n_steps=n_steps
        )

        plot_multi_curves(
            results,
            title=f'TTD(λ): n = {n_steps}',
            labels=lambdas,
            x=alphas)

    results = experiments_parallel(
        model_type='online_td_lambda',
        alphas=alphas,
        lambdas=lambdas,
        n_experiments=100,
        n_episodes=10
    )

    plot_multi_curves(
        results,
        title='Online/True TD(λ)',
        labels=lambdas,
        x=alphas)

    results = experiments_parallel(
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

    # ------- Figure 12.6 -------- #
    results = experiments_parallel(
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

