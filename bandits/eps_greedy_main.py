import os.path
from typing import Callable, List, Any, Dict, Union, Tuple
from joblib import Parallel, delayed

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from tools.coefficients import ConstantCoefficient
from nonassocative_value_functions import QMonteCarlo, QCoefficientMovingAverage
from nonassociative_policies import EpsGreedyPolicy, ActionValuePolicy
from environments.testbed import NonAssocativeTestBed
from environments.continuous_reward_testbed import \
    ContinuousValueRewardTestBed

from utils import mk_clear_dir


def get_cont_reward_test_bed(reward_means, reward_randomness_scale, stationary=False):
    return ContinuousValueRewardTestBed(
        reward_means=reward_means,
        reward_randomness_scales=reward_randomness_scale,
        stationary=stationary)


def run_non_associative_test_bed(
        test_bed: NonAssocativeTestBed,
        policy: ActionValuePolicy,
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


def parallel_simulate_eps_greedy(
        test_bed_constructor: Callable[[], NonAssocativeTestBed],
        policy_constructor: Callable[[float], EpsGreedyPolicy],
        epsilons: List[float],
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

    for eps in tqdm(epsilons, desc=desc):
        results = Parallel(n_jobs=-1)(
            delayed(run_non_associative_test_bed)(
                test_bed=test_bed_constructor(),
                policy=policy_constructor(eps),
                n_steps=n_steps
            )
            for t in range(n_trials)
        )

        rewards[eps] = [r['rewards'] for r in results]
        regrets[eps] = [r['regrets'] for r in results]
        best_means[eps] = [r['best_means'] for r in results]
        best_arms[eps] = [r['best_arms'] for r in results]
        cummulative_rewards[eps] = [r['cummulative_rewards'] for r in results]
        cummulative_best_means[eps] = [r['cummulative_best_means'] for r in results]

    return dict(
        rewards=rewards,
        regrets=regrets,
        best_means=best_means,
        best_arms=best_arms,
        cummulative_rewards=cummulative_rewards,
        cummulative_best_means=cummulative_best_means
    )



def experiment_1(n_steps, n_trials):
    """
        For a fixed action-value (E[R | a]) for each bandit,
        compare the performance of epsilon greedy
        policies accross different values of exploration.

        Q uses sample averaging (unbiased estimator)
    """

    n_bandits = 10  # Each bandit is triggered by one action
    Q0 = 0.0
    epsilons = [0.0, 0.01, 0.1, 0.3]
    reward_randomness_scale = 0.00
    plot_root_name = 'experiment_1'

    test_bed_constructor = lambda: get_cont_reward_test_bed(
        reward_means=np.random.normal(
            0.,
            1.,
            size=n_bandits
        ).tolist(),
        reward_randomness_scale=[reward_randomness_scale] * n_bandits,
        stationary=True
    )

    q_constructor = lambda: QMonteCarlo(
        n_actions=n_bandits,
        initial_action_value=Q0
    )

    policy_constructor = lambda _e: EpsGreedyPolicy(
        q=q_constructor(),
        eps=_e
    )

    results = parallel_simulate_eps_greedy(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor,
        epsilons=epsilons,
        n_trials=n_trials,
        n_steps=n_steps,
        desc='Experiment 1')

    reward_averages = dict()
    regret_averages = dict()

    rewards = results['rewards']
    regrets = results['regrets']

    for eps in epsilons:
        reward_averages[eps] = []
        regret_averages[eps] = []

        for step in range(n_steps):
            # Average across trials at each step
            res = [rewards[eps][trial][step] for trial in range(n_trials)]
            reward_averages[eps].append(np.mean(res))

            res = [regrets[eps][trial][step] for trial in range(n_trials)]
            regret_averages[eps].append(np.mean(res) / (step + 1))

    d = os.path.join(os.getcwd(), 'images')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()
    for eps in epsilons:
        plt.plot(reward_averages[eps])
    plt.legend([f'eps: {eps:.1e}' for eps in epsilons])
    plt.ylabel('Average Reward')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 1: Eps-Greedy\nAvg. Rewards')
    plt.grid()
    try:
        plt.savefig(os.path.join(d, f'rewards_{plot_root_name}.png'))
    except:
        print(f'Could not save rewards_{plot_root_name} plot')
    finally:
        plt.show()

    _ = plt.figure()
    for eps in epsilons:
        plt.plot(regret_averages[eps])
    plt.legend([f'eps: {eps:.1e}' for eps in epsilons])
    plt.ylabel('Regret/step')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 1: Eps-Greedy\nAverage Regret')
    plt.grid()
    try:
        plt.savefig(os.path.join(d, f'regrets_{plot_root_name}.png'))
    except:
        print(f'Could not save regrets_{plot_root_name} plot')
    finally:
        plt.show()


def experiment_2(n_steps, n_trials):
    """
        For a fixed action-value (E[R | a]) for each bandit,
        compare the performance of epsilon greedy
        policies accross different values of exploration.

        Q fixed step size (biased estimator)
    """

    exp = 2
    n_bandits = 10  # Each bandit is triggered by one action
    Q0 = 0.0
    epsilons = [0.0, 0.01, 0.1, 0.3]
    alpha = 0.1
    reward_randomness_scale = 0.00
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

    q_constructor = lambda: QCoefficientMovingAverage(
        n_actions=n_bandits,
        initial_action_value=Q0,
        coefficient=ConstantCoefficient(alpha)
    )

    policy_constructor = lambda _e: EpsGreedyPolicy(
        q=q_constructor(),
        eps=_e
    )

    results = parallel_simulate_eps_greedy(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor,
        epsilons=epsilons,
        n_trials=n_trials,
        n_steps=n_steps,
        desc=f'Experiment {exp}')

    reward_averages = dict()
    regret_averages = dict()

    rewards = results['rewards']
    regrets = results['regrets']

    for eps in epsilons:
        reward_averages[eps] = []
        regret_averages[eps] = []

        for step in range(n_steps):
            # Average across trials at each step
            res = [rewards[eps][trial][step] for trial in range(n_trials)]
            reward_averages[eps].append(np.mean(res))

            res = [regrets[eps][trial][step] for trial in range(n_trials)]
            regret_averages[eps].append(np.mean(res) / (step + 1))

    d = os.path.join(os.getcwd(), 'images')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()
    for eps in epsilons:
        plt.plot(reward_averages[eps])
    plt.legend([f'eps: {eps:.1e}' for eps in epsilons])
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
    for eps in epsilons:
        plt.plot(regret_averages[eps])
    plt.legend([f'eps: {eps:.1e}' for eps in epsilons])
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
    experiment_2(n_steps=1000, n_trials=2000)

    exit(0)