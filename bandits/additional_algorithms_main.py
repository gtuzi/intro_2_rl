import os.path
from typing import Callable, List
from joblib import Parallel, delayed

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from bandits.environments.binary_reward_testbed import BinaryValueRewardTestBed
from bandits.nonassocative_value_functions import QMonteCarlo, \
    QCoefficientMovingAverage
from bandits.tools.coefficients import ConstantCoefficient
from nonassociative_policies import (
    Policy,
    SoftmaxExplorationPolicy,
    BernoulliGreedy, BernoulliThompsonSampling
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


def get_binary_reward_test_bed(
        success_rates,
        reward_randomness_scales: List = (),
        stationary=False
):
    return BinaryValueRewardTestBed(
        success_rates=success_rates,
        reward_randomness_scales=reward_randomness_scales,
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


def experiment_8(n_steps, n_trials):
    """
        Softmax-Exploration example. Iterate over different levels of temperature
        and compare between simple averaging and exponentially-weighted recency
        averaging (exponential moving average) of action-value / average value.
    """
    exp = 8
    n_bandits = 10  # Each bandit is triggered by one action
    Q0 = 0.0
    alpha = 0.1
    reward_randomness_scale = 1.0
    temperatures = [1e-2, 1., 2.]
    plot_root_text = f'experiment_{exp}'

    test_bed_constructor = lambda: get_cont_reward_test_bed(
        reward_means=np.random.normal(
            0,
            1.,
            size=n_bandits
        ).tolist(),
        reward_randomness_scale=[reward_randomness_scale] * n_bandits,
        stationary=False)

    # ---- Simple Average for sample mean ---- #

    q_constructor = lambda: QMonteCarlo(
        n_actions=n_bandits,
        initial_action_value=Q0
    )

    policy_constructor = lambda _temp: SoftmaxExplorationPolicy(
        q=q_constructor(),
        temperature=_temp
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


    # ------- Exponential averaging of the mean ------ #
    q_constructor = lambda: QCoefficientMovingAverage(
        n_actions=n_bandits,
        initial_action_value=Q0,
        coefficient=ConstantCoefficient(alpha),
    )

    policy_constructor = lambda _temp: SoftmaxExplorationPolicy(
        q=q_constructor(),
        temperature=_temp
    )

    results = parallel_simulate_over_1dparam(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor,
        params=temperatures,
        n_trials=n_trials,
        n_steps=n_steps,
        desc=f'Experiment {exp}')

    reward_averages_exp_avg = dict()
    regret_averages_exp_avg = dict()

    rewards_exp_avg = results['rewards']
    regrets_exp_avg = results['regrets']

    for temp in temperatures:
        reward_averages_exp_avg[temp] = []
        regret_averages_exp_avg[temp] = []

        for step in range(n_steps):
            # Average across trials at each step
            res = [rewards_exp_avg[temp][trial][step] for trial in range(n_trials)]
            reward_averages_exp_avg[temp].append(np.mean(res))

            res = [regrets_exp_avg[temp][trial][step] for trial in range(n_trials)]
            regret_averages_exp_avg[temp].append(np.mean(res) / (step + 1))


    d = os.path.join(os.getcwd(), 'images')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()

    legend = []
    plt.plot(reward_averages[temperatures[0]])
    legend.append(f'Simple-avg, $\\tau$: {temperatures[0]:.1e}')
    plt.plot(reward_averages[temperatures[1]])
    legend.append(f'Simple-avg, $\\tau$: {temperatures[1]:.1e}')
    plt.plot(reward_averages[temperatures[2]])
    legend.append(f'Simple-avg, $\\tau$: {temperatures[2]:.1e}')

    plt.plot(reward_averages_exp_avg[temperatures[0]])
    legend.append(f'Exp-avg, $\\tau$: {temperatures[0]:.1e}')
    plt.plot(reward_averages_exp_avg[temperatures[1]])
    legend.append(f'Exp-avg, $\\tau$: {temperatures[1]:.1e}')
    plt.plot(reward_averages_exp_avg[temperatures[2]])
    legend.append(f'Exp-avg, $\\tau$: {temperatures[2]:.1e}')

    plt.legend(legend)
    plt.ylabel('Average Reward')
    plt.xlabel('Simulation Step')
    plt.title(f'Experiment {exp}: SoftmaxExploration, Non-Stationary Env')
    try:
        plt.savefig(os.path.join(d, f'rewards_{plot_root_text}.png'))
    except:
        print(f'Could not save rewards_{plot_root_text} plots')
    finally:
        plt.show()


    _ = plt.figure()

    legend = []
    plt.plot(regret_averages[temperatures[0]])
    legend.append(f'Simple-avg, $\\tau$: {temperatures[0]:.1e}')
    plt.plot(regret_averages[temperatures[1]])
    legend.append(f'Simple-avg, $\\tau$: {temperatures[1]:.1e}')
    plt.plot(regret_averages[temperatures[2]])
    legend.append(f'Simple-avg, $\\tau$: {temperatures[2]:.1e}')

    plt.plot(regret_averages_exp_avg[temperatures[0]])
    legend.append(f'Exp-avg, $\\tau$: {temperatures[0]:.1e}')
    plt.plot(regret_averages_exp_avg[temperatures[1]])
    legend.append(f'Exp-avg, $\\tau$: {temperatures[1]:.1e}')
    plt.plot(regret_averages_exp_avg[temperatures[2]])
    legend.append(f'Exp-avg, $\\tau$: {temperatures[2]:.1e}')

    plt.legend(legend)
    plt.ylabel('Average Regret')
    plt.xlabel('Simulation Step')
    plt.title(
        f'Experiment {exp}: SoftmaxExploration, Non-Stationary Env.')
    try:
        plt.savefig(os.path.join(d, f'regrets_{plot_root_text}.png'))
    except:
        print(f'Could not save regrets_{plot_root_text} plots')
    finally:
        plt.show()


def experiment_9(n_steps, n_trials):
    from itertools import product

    """
        Beta-Bernoulli greedy algorithm (Algorithm 1 in
        https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf).
    """
    exp = 9
    n_bandits = 10  # Each bandit is triggered by one action
    alphas = [0.5, 1., 2]
    betas = [0.5, 1., 2]
    ab = list(product(alphas, betas))
    abi = list(range(len(ab)))
    plot_root_text = f'experiment_{exp}'

    min_success = 0.1
    max_successs = 0.9

    true_success_rates = lambda: [
        float(np.random.uniform(min_success, max_successs, size=1))
        for _ in range(n_bandits)
    ]

    test_bed_constructor = lambda: get_binary_reward_test_bed(
        success_rates=true_success_rates(),
        stationary=True)

    policy_constructor = lambda _i: BernoulliGreedy(
        initial_alpha = ab[_i][0],
        initial_beta = ab[_i][1],
        n_actions=n_bandits
    )

    results = parallel_simulate_over_1dparam(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor,
        params=abi,
        n_trials=n_trials,
        n_steps=n_steps,
        desc=f'Experiment {exp}')

    reward_averages = dict()
    regret_averages = dict()

    rewards = results['rewards']
    regrets = results['regrets']

    for i in abi:
        reward_averages[i] = []
        regret_averages[i] = []

        for step in range(n_steps):
            # Average across trials at each step
            res = [rewards[i][trial][step] for trial in range(n_trials)]
            reward_averages[i].append(np.mean(res))

            res = [regrets[i][trial][step] for trial in range(n_trials)]
            regret_averages[i].append(np.mean(res) / (step + 1))

    d = os.path.join(os.getcwd(), 'images')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()

    legend = []
    for i in abi:
        plt.plot(reward_averages[i])
        legend.append(f'$\\alpha$: {ab[i][0]:.1f}, $\\beta$: {ab[i][1]: .1f}')

    plt.legend(legend)
    plt.ylabel('Average Reward')
    plt.xlabel('Simulation Step')
    plt.title(f'Experiment {exp}: Bernoulli-Greedy')
    try:
        plt.savefig(os.path.join(d, f'rewards_{plot_root_text}.png'))
    except:
        print(f'Could not save rewards_{plot_root_text} plots')
    finally:
        plt.show()


    _ = plt.figure()

    legend = []
    for i in abi:
        plt.plot(regret_averages[i])
        legend.append(f'$\\alpha$: {ab[i][0]:.1f}, $\\beta$: {ab[i][1]: .1f}')

    plt.legend(legend)
    plt.ylabel('Average Regret')
    plt.xlabel('Simulation Step')
    plt.title(f'Experiment {exp}: Bernoulli-Greedy')
    try:
        plt.savefig(os.path.join(d, f'regrets_{plot_root_text}.png'))
    except:
        print(f'Could not save regrets_{plot_root_text} plots')
    finally:
        plt.show()


def experiment_10(n_steps, n_trials):
    from itertools import product

    """
        Bernoulli with Thompson-Sampling (Algorithm 2 in
        https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf).
    """
    exp = 10
    n_bandits = 10  # Each bandit is triggered by one action
    alphas = [0.5, 1., 2]
    betas = [0.5, 1., 2]
    ab = list(product(alphas, betas))
    abi = list(range(len(ab)))
    plot_root_text = f'experiment_{exp}'

    min_success = 0.1
    max_successs = 0.9

    true_success_rates = lambda: [
        float(np.random.uniform(min_success, max_successs, size=1))
        for _ in range(n_bandits)
    ]

    test_bed_constructor = lambda: get_binary_reward_test_bed(
        success_rates=true_success_rates(),
        stationary=True)

    policy_constructor = lambda _i: BernoulliThompsonSampling(
        initial_alpha = ab[_i][0],
        initial_beta = ab[_i][1],
        n_actions=n_bandits
    )

    results = parallel_simulate_over_1dparam(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor,
        params=abi,
        n_trials=n_trials,
        n_steps=n_steps,
        desc=f'Experiment {exp}')

    reward_averages = dict()
    regret_averages = dict()

    rewards = results['rewards']
    regrets = results['regrets']

    for i in abi:
        reward_averages[i] = []
        regret_averages[i] = []

        for step in range(n_steps):
            # Average across trials at each step
            res = [rewards[i][trial][step] for trial in range(n_trials)]
            reward_averages[i].append(np.mean(res))

            res = [regrets[i][trial][step] for trial in range(n_trials)]
            regret_averages[i].append(np.mean(res) / (step + 1))

    d = os.path.join(os.getcwd(), 'images')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()

    legend = []
    for i in abi:
        plt.plot(reward_averages[i])
        legend.append(f'$\\alpha$: {ab[i][0]:.1f}, $\\beta$: {ab[i][1]: .1f}')

    plt.legend(legend)
    plt.ylabel('Average Reward')
    plt.xlabel('Simulation Step')
    plt.title(f'Experiment {exp}: Bernoulli-TS')
    try:
        plt.savefig(os.path.join(d, f'rewards_{plot_root_text}.png'))
    except:
        print(f'Could not save rewards_{plot_root_text} plots')
    finally:
        plt.show()


    _ = plt.figure()

    legend = []
    for i in abi:
        plt.plot(regret_averages[i])
        legend.append(f'$\\alpha$: {ab[i][0]:.1f}, $\\beta$: {ab[i][1]: .1f}')

    plt.legend(legend)
    plt.ylabel('Average Regret')
    plt.xlabel('Simulation Step')
    plt.title(
        f'Experiment {exp}: Bernoulli-TS')
    try:
        plt.savefig(os.path.join(d, f'regrets_{plot_root_text}.png'))
    except:
        print(f'Could not save regrets_{plot_root_text} plots')
    finally:
        plt.show()


def experiment_11(n_steps, n_trials):
    from itertools import product

    """
        Bernoulli with Thompson-Sampling (Algorithm 2 in
        https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf).
    """
    exp = 11
    n_bandits = 10  # Each bandit is triggered by one action
    alphas = [0.5, 1., 2]
    betas = [0.5, 1., 2]
    ab = list(product(alphas, betas))
    abi = list(range(len(ab)))
    plot_root_text = f'experiment_{exp}'

    rand_scale = 0.02
    min_success = 0.1
    max_successs = 0.9

    true_success_rates = lambda: [
        float(np.random.uniform(min_success, max_successs, size=1))
        for _ in range(n_bandits)
    ]

    reward_randomness_scales = [rand_scale for _ in range(n_bandits)]

    test_bed_constructor = lambda: get_binary_reward_test_bed(
        success_rates=true_success_rates(),
        reward_randomness_scales=reward_randomness_scales,
        stationary=False)

    policy_constructor = lambda _i: BernoulliThompsonSampling(
        initial_alpha = ab[_i][0],
        initial_beta = ab[_i][1],
        n_actions=n_bandits
    )

    results = parallel_simulate_over_1dparam(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor,
        params=abi,
        n_trials=n_trials,
        n_steps=n_steps,
        desc=f'Experiment {exp}')

    reward_averages = dict()
    regret_averages = dict()

    rewards = results['rewards']
    regrets = results['regrets']

    for i in abi:
        reward_averages[i] = []
        regret_averages[i] = []

        for step in range(n_steps):
            # Average across trials at each step
            res = [rewards[i][trial][step] for trial in range(n_trials)]
            reward_averages[i].append(np.mean(res))

            res = [regrets[i][trial][step] for trial in range(n_trials)]
            regret_averages[i].append(np.mean(res) / (step + 1))

    d = os.path.join(os.getcwd(), 'images')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()

    legend = []
    for i in abi:
        plt.plot(reward_averages[i])
        legend.append(f'$\\alpha$: {ab[i][0]:.1f}, $\\beta$: {ab[i][1]: .1f}')

    plt.legend(legend)
    plt.ylabel('Average Reward')
    plt.xlabel('Simulation Step')
    plt.title(f'Experiment {exp}: Bernoulli-TS')
    try:
        plt.savefig(os.path.join(d, f'rewards_{plot_root_text}.png'))
    except:
        print(f'Could not save rewards_{plot_root_text} plots')
    finally:
        plt.show()


    _ = plt.figure()

    legend = []
    for i in abi:
        plt.plot(regret_averages[i])
        legend.append(f'$\\alpha$: {ab[i][0]:.1f}, $\\beta$: {ab[i][1]: .1f}')

    plt.legend(legend)
    plt.ylabel('Average Regret')
    plt.xlabel('Simulation Step')
    plt.title(
        f'Experiment {exp}: Bernoulli-TS')
    try:
        plt.savefig(os.path.join(d, f'regrets_{plot_root_text}.png'))
    except:
        print(f'Could not save regrets_{plot_root_text} plots')
    finally:
        plt.show()


if __name__ == '__main__':
    experiment_11(1000, 2000)
    exit(0)