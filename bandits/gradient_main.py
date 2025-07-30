import os.path
from typing import Callable, List
from joblib import Parallel, delayed

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from bandits.tools.coefficients import ConstantCoefficient
from nonassocative_value_functions import QMonteCarlo
from nonassociative_policies import (
    EpsGreedyPolicy,
    ActionValuePolicy,
    UCB1Policy, NaiivePreferencePolicy, Policy
)
from environments.testbed import NonAssocativeTestBed
from environments.continuous_reward_testbed import \
    ContinuousValueRewardTestBed

from utils import mk_clear_dir


def get_cont_reward_test_bed(
        reward_means,
        reward_randomness_scale,
        stationary=False
):
    return ContinuousValueRewardTestBed(
        reward_means=reward_means,
        reward_randomness_scales=reward_randomness_scale,
        stationary=stationary)


def run_non_associative_test_bed(
        test_bed: NonAssocativeTestBed,
        policy: Policy,
        n_steps: int):

    rewards = []
    regrets = []
    best_arms = []
    best_means = []
    cummulative_rewards = []
    cummulative_best_means = []

    for step in range(n_steps):
        a = policy(step=step)
        r = test_bed(action=a)
        policy.step(step=step, action=a, reward=r)
        rewards.append(r)
        best_arms.append(test_bed.best_arm)
        best_means.append(test_bed.best_mean)

        if step == 0:
            regrets.append(test_bed.best_mean - test_bed.arm_mean(a))
            cummulative_rewards.append(r)
            cummulative_best_means.append(test_bed.best_mean)
        else:
            regrets.append((test_bed.best_mean - test_bed.arm_mean(a)) + regrets[-1])
            cummulative_rewards.append(r + cummulative_rewards[-1])
            cummulative_best_means.append(
                test_bed.best_mean + cummulative_best_means[-1])

    return dict(
        rewards=rewards,
        regrets=regrets,
        best_means=best_means,
        best_arms=best_arms,
        cummulative_rewards=cummulative_rewards,
        cummulative_best_means=cummulative_best_means
    )


def parallel_simulate_over_1dparam(
        test_bed_constructor: Callable[[], NonAssocativeTestBed],
        policy_constructor: Callable[[float], Policy],
        params: List[float],
        n_trials: int,
        n_steps: int,
        desc=''
):
    rewards = {}
    regrets = {}
    best_means = {}
    best_arms = {}
    cummulative_rewards = {}
    cummulative_best_means = {}

    for param in tqdm(params, desc=desc):
        results = Parallel(n_jobs=-1)(
            delayed(run_non_associative_test_bed)(
                test_bed=test_bed_constructor(),
                policy=policy_constructor(param),
                n_steps=n_steps
            )
            for t in range(n_trials)
        )

        rewards[param] = [r['rewards'] for r in results]
        regrets[param] = [r['regrets'] for r in results]
        best_means[param] = [r['best_means'] for r in results]
        best_arms[param] = [r['best_arms'] for r in results]
        cummulative_rewards[param] = [r['cummulative_rewards'] for r in results]
        cummulative_best_means[param] = [r['cummulative_best_means'] for r in results]

    return dict(
        rewards=rewards,
        regrets=regrets,
        best_means=best_means,
        best_arms=best_arms,
        cummulative_rewards=cummulative_rewards,
        cummulative_best_means=cummulative_best_means
    )


def experiment_7(n_steps, n_trials):
    """
    Naiive preference policy over different constant temperatures

    Action is sampled as a ~ softmax(H)

    From eq (2.12) in Sutton book, 2nd edition (2018):

    H(A, t+1) = H(A, t) + alpha * Advantage * (1 - pi(A))
    H(o, t+1) = H(o, t) - alpha * Advantage * p(o)
    where:
    H: preference model
    R_bar: baseline, a moving average of reward received
    Advantage: R(t) - R_bar

    """
    exp = 7
    n_bandits = 10  # Each bandit is triggered by one action
    H0 = 0.0
    Rbar0 = 0.0
    alpha = 0.1
    temperatures = [0.1, 0.5, 1.0, 2.0, 4.0]
    reward_randomness_scale = 1.00
    plot_root_name = f'experiment_{exp}'

    test_bed_constructor = lambda: get_cont_reward_test_bed(
        reward_means=np.random.normal(
            0.,
            1.,
            size=n_bandits
        ).tolist(),
        reward_randomness_scale=[reward_randomness_scale] * n_bandits,
        stationary=True
    )

    policy_constructor = lambda _temp: NaiivePreferencePolicy(
        n_actions=n_bandits,
        preference_initial_value=H0,
        reward_initial_value=Rbar0,
        learning_rate=alpha,
        temperature=ConstantCoefficient(_temp)
    )

    results = parallel_simulate_over_1dparam(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor,
        params=temperatures,
        n_trials=n_trials,
        n_steps=n_steps,
        desc=f'Experiment {exp}')

    reward_averages = dict()
    regret_averages = dict()

    rewards = results['rewards']
    regrets = results['regrets']

    for temp in temperatures:
        reward_averages[temp] = []
        regret_averages[temp] = []

        for step in range(n_steps):
            # Average across trials at each step
            res = [rewards[temp][trial][step] for trial in range(n_trials)]
            reward_averages[temp].append(np.mean(res))

            res = [regrets[temp][trial][step] for trial in range(n_trials)]
            regret_averages[temp].append(np.mean(res) / (step + 1))

    d = os.path.join(os.getcwd(), 'images')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()
    for temp in temperatures:
        plt.plot(reward_averages[temp])
    plt.legend([f'T: {temp:.1e}' for temp in temperatures])
    plt.ylabel('Average Reward')
    plt.xlabel('Simulation Step')
    plt.title(f'Experiment {exp}: Eps-Greedy\nAvg. Rewards')
    plt.grid()
    try:
        plt.savefig(os.path.join(d, f'rewards_{plot_root_name}.png'))
    except:
        print(f'Could not save rewards_{plot_root_name} plot')
    finally:
        plt.show()

    _ = plt.figure()
    for temp in temperatures:
        plt.plot(regret_averages[temp])
    plt.legend([f'T: {temp:.1e}' for temp in temperatures])
    plt.ylabel('Regret/step')
    plt.xlabel('Simulation Step')
    plt.title(f'Experiment {exp}: Eps-Greedy\nAverage Regret')
    plt.grid()
    try:
        plt.savefig(os.path.join(d, f'regrets_{plot_root_name}.png'))
    except:
        print(f'Could not save regrets_{plot_root_name} plot')
    finally:
        plt.show()


if __name__ == '__main__':
    experiment_7(1000, 2000)
    exit(0)