import random
import copy
from typing import List, Callable
from joblib import Parallel, delayed

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import Env

from agents import SarsaLambda, TrueOnlineSarsaLambda


from approximate_methods.utils import (
    DiscreteActionAgent,
    SoftPolicy,
    Experience,
    TileCodingFeature)

from shared.utils import LinearSchedule


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
        title='Sarsa(λ)',
        xlabel='α',
        ylabel='Steps per Episode',
        ymin: int = 150,
        ymax: int = 300
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
    plt.ylim(ymin, ymax)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

#endregion plots


def run_env_episodic(
        env: Env,
        behavioral_agent: DiscreteActionAgent,
        target_agent: DiscreteActionAgent = None,
        reward_shaper: Callable = lambda reward, state, done, t: reward,
        T: int = 30,
        num_episodes: int = 10,
        seeds=None
):
    steps_per_episode = []
    sum_of_rewards_per_episode = []

    # ----- Unlearn ----- #
    behavioral_agent.initialize()
    if target_agent is not None:
        target_agent.initialize()

    for ei, episode in enumerate(tqdm(
            range(num_episodes),
            desc=f'Episode'
    )):
        # For seeds:
        # 1 -   None
        # 2 -   Single value
        # 3 -   Same as number of episodes
        seed = None
        if seeds is not None:
            if hasattr(seeds, '__getitem__'):
                assert len(seeds) == num_episodes
                seed = seeds[ei]
            elif isinstance(seeds, (float, int)):
                seed = seeds
            else:
                raise Exception("Seed format not recognized", str(seeds))

        # Set the seed for this episode
        random.seed(seed)
        np.random.seed(seed)

        # Noise state reset (not exploration level).
        # Clear any trajectories.
        # Clear any eligibility traces
        behavioral_agent.reset()

        # gymnasium v26 requires users to set seed
        # when resetting the environment
        s, info = env.reset(seed=seed) # s[0]
        a, p = behavioral_agent.act(s) # a[0]
        R = 0
        for t in range(T):
            sp, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated or (t + 1 == T)
            reward = reward_shaper(reward=r, state = sp, done=done, t=t)
            ap, pp = behavioral_agent.act(sp)
            R += r

            # If off-policy, capture target probabilities
            rhop = None
            if (target_agent is not None) and (
            isinstance(target_agent, SoftPolicy)
            ):
                target_pp = target_agent.get_sa_probability(sp, ap)
                rhop = target_pp / pp

            e = Experience(
                s=s,
                a=a,
                p=p,
                r=reward,
                sp=sp,
                ap=ap,
                pp=pp,
                done=int(done),
                rhop=rhop,
                t=t)

            behavioral_agent.step(e)

            if target_agent is not None:
                target_agent.step(e)

            if done:
                steps_per_episode.append(t)
                sum_of_rewards_per_episode.append(R)
                break
            else:
                s = sp
                a = ap
                p = pp

    return steps_per_episode, sum_of_rewards_per_episode


def build_env() -> Env:
    global ENV_NAME
    global RENDER
    global MAX_EPISODE_STEPS

    if ENV_NAME.lower() == 'MountainCar'.lower():
        env = gym.make(
            'MountainCar-v0',
            render_mode="human" if RENDER else None)
        env._max_episode_steps = MAX_EPISODE_STEPS
    elif ENV_NAME.lower() == 'AirRaid'.lower():
        env = gym.make(
            "ALE/AirRaid-v5",
            obs_type="rgb",
            render_mode="human" if RENDER else None
        )
    elif ENV_NAME.lower() == 'LunarLander'.lower():
        env = gym.make(
            "LunarLander-v2",
            render_mode="human" if RENDER else None
        )
    else:
        raise NotImplementedError

    return copy.deepcopy(env)


def run_one_experiment(
        model: str,
        num_episodes,
        T,
        alpha,
        lam,
        eps,
        seeds,
        reward_shaper: Callable,
        eps_builder: Callable = lambda x: x,
):
    env = build_env()

    num_tilings = 8
    num_tiles = 8
    max_size = 4096

    x0_low, x1_low = env.observation_space.low
    x0_high, x1_high = env.observation_space.high

    '''
        From Section 10.1:
            We used 8 tilings, with each tile covering 1/8th of 
            the bounded distance in each dimension
    '''
    feature_fn = TileCodingFeature(
        max_size, num_tiles, num_tilings, x0_low, x1_low, x0_high, x1_high)

    if model == 'SarsaLambda':
        agent = SarsaLambda(
            feature_size=max_size,
            action_space_dims=int(env.action_space.n),
            update_coefficient=alpha / num_tilings,
            feature_fn=feature_fn,
            discount=0.99,
            lam=lam,
            eps=eps_builder(eps),
            trace_mode='accumulate')

    elif model == 'TrueOnlineSarsaLambda':
        agent = TrueOnlineSarsaLambda(
            feature_size=max_size,
            action_space_dims=int(env.action_space.n),
            update_coefficient=alpha / num_tilings,
            feature_fn=feature_fn,
            discount=0.99,
            lam=lam,
            eps=eps_builder(eps)
        )
    else:
        raise NotImplemented(f'{model} model not recognized')

    steps_per_episode, sum_of_rewards_per_episode = run_env_episodic(
        env=env,
        behavioral_agent=agent,
        reward_shaper=reward_shaper,
        T=T,
        num_episodes=num_episodes,
        seeds=seeds)

    return dict(
        steps_per_episode=steps_per_episode,
        sum_of_rewards_per_episode=sum_of_rewards_per_episode
    )


def experiments_parallel(
        model: str,
        num_episodes,
        T,
        reward_shaper: Callable,
        eps_builder: Callable = lambda x: x,
        alphas = np.linspace(0, 1, num=50),
        lambdas=(0.99, 0.98, 0.96, 0.92, 0.84, 0.68, 0.),
        num_experiments: int = 10,
        eps=0.1,
        seeds=(1, 2)
):
    # Prepare a 2D array to store final means
    steps_per_episode = np.zeros((len(lambdas), len(alphas)), dtype=np.float64)
    sum_of_rewards_per_episode = np.zeros((len(lambdas), len(alphas)), dtype=np.float64)

    # Offset the original seeds
    seeds_per_experiment = [
        [s + e * len(seeds) for s in seeds]
        for e in range(num_experiments)
    ]

    # Loop (λ, α) in serial (this is cheap: just n_states‐grid combinations)
    for ia, alpha in enumerate(alphas):
        for il, lam in enumerate(lambdas):
            #    Launch n_experiments calls of run_one_experiment(...) *in parallel*
            #    Each call returns a single‐experiment‐average‐RMS.
            #    We use n_jobs=-1 to utilize all CPU cores by default.
            res = Parallel(n_jobs=-1)(
                delayed(run_one_experiment)(
                    model = model,
                    num_episodes=num_episodes,
                    T=T,
                    alpha=alpha,
                    lam=lam,
                    eps=eps,
                    reward_shaper=reward_shaper,
                    eps_builder=eps_builder,
                    seeds = seeds_per_experiment[e]
                )
                for e in range(num_experiments)
            )

            steps = [r['steps_per_episode'] for r in res]
            sum_rewards = [r['sum_of_rewards_per_episode'] for r in res]

            # 4) Average those n_experiments results to fill results_array
            steps_per_episode[il, ia] = np.mean(steps)
            sum_of_rewards_per_episode[il, ia] = np.mean(sum_rewards)

            print(f"\n{model} - Done α={alpha:.3f}, λ={lam:.3f} → steps/episode: {steps_per_episode[il, ia]:.4f}, sum(r)/episode: {sum_of_rewards_per_episode[il, ia]:.4f}")

    if model == 'SarsaLambda':
        title = 'Sarsa(λ)'
    elif model == 'TrueOnlineSarsaLambda':
        title = 'True Online Sarsa(λ)'
    else:
        raise NotImplemented

    plot_multi_curves(
        steps_per_episode,
        title=title,
        labels=lambdas,
        x=alphas,
        ylabel='Avg. Steps/Episode',
    )

    plot_multi_curves(
        sum_of_rewards_per_episode,
        title=title,
        labels=lambdas,
        x=alphas,
        ylabel='Avg. Sum(Rewards)/Episode',
        ymin= -500,
        ymax= -150
    )

def sarsa_lambda_experiments(
        num_episodes,
        T,
        reward_shaper: Callable,
        eps_builder: Callable = lambda x: x,
        alphas = np.linspace(0, 1, num=50),
        lambdas=(0.99, 0.98, 0.96, 0.92, 0.84, 0.68, 0.),
        eps=0.1,
        seeds=(1, 2)
):

    env = build_env()

    steps_per_episode = np.zeros((len(lambdas), len(alphas)), dtype=np.float64)
    sum_of_rewards_per_episode = np.zeros((len(lambdas), len(alphas)), dtype=np.float64)

    num_tilings = 8
    num_tiles = 8
    max_size = 4096

    x0_low, x1_low = env.observation_space.low
    x0_high, x1_high = env.observation_space.high

    '''
        From Section 10.1:
            We used 8 tilings, with each tile covering 1/8th of 
            the bounded distance in each dimension
    '''
    feature_fn = TileCodingFeature(
        max_size, num_tiles, num_tilings, x0_low, x1_low, x0_high, x1_high)

    for il, lam in enumerate(lambdas):
        for ia, alpha in enumerate(alphas):
            agent = SarsaLambda(
                feature_size=max_size,
                action_space_dims=int(env.action_space.n),
                update_coefficient=alpha / num_tilings,
                feature_fn=feature_fn,
                discount=0.99,
                lam=lam,
                eps=eps_builder(eps),
                trace_mode='replace',
            )

            steps, sum_rewards = run_env_episodic(
                env=env,
                behavioral_agent=agent,
                reward_shaper=reward_shaper,
                T=T,
                num_episodes=num_episodes,
                seeds=seeds)

            # Average those n_experiments results to fill results_array
            steps_per_episode[il, ia] = np.mean(steps)
            sum_of_rewards_per_episode[il, ia] = np.mean(sum_rewards)

            print(f"\n{type(agent)}: Done α={alpha:.3f}, λ={lam:.3f} → steps/episode: {steps_per_episode[il, ia]:.4f}, sum(r)/episode: {sum_of_rewards_per_episode[il, ia]:.4f}")

    plot_multi_curves(
        steps_per_episode,
        title='Sarsa(λ)',
        labels=lambdas,
        ylabel='Avg. Steps / Episode',
        x=alphas)

    plot_multi_curves(
        sum_of_rewards_per_episode,
        title='Sarsa(λ)',
        labels=lambdas,
        ylabel='Avg. Sum(Rewards) / Episode',
        x=alphas)


def true_online_sarsa_lambda_experiments(
        num_episodes,
        T,
        reward_shaper: Callable,
        eps_builder: Callable = lambda x: x,
        alphas = np.linspace(0, 1, num=50),
        lambdas=(0.99, 0.98, 0.96, 0.92, 0.84, 0.68, 0.),
        eps=0.1,
        seeds=(1, 2)
):

    env = build_env()

    steps_per_episode = np.zeros((len(lambdas), len(alphas)), dtype=np.float64)
    sum_of_rewards_per_episode = np.zeros((len(lambdas), len(alphas)), dtype=np.float64)

    num_tilings = 8
    num_tiles = 8
    max_size = 4096

    x0_low, x1_low = env.observation_space.low
    x0_high, x1_high = env.observation_space.high

    '''
        From Section 10.1:
            We used 8 tilings, with each tile covering 1/8th of 
            the bounded distance in each dimension
    '''
    feature_fn = TileCodingFeature(
        max_size, num_tiles, num_tilings, x0_low, x1_low, x0_high, x1_high)

    for il, lam in enumerate(lambdas):
        for ia, alpha in enumerate(alphas):
            agent = TrueOnlineSarsaLambda(
                feature_size=max_size,
                action_space_dims=int(env.action_space.n),
                update_coefficient=alpha / num_tilings,
                feature_fn=feature_fn,
                discount=0.99,
                lam=lam,
                eps=eps_builder(eps))

            steps, sum_rewards = run_env_episodic(
                env=env,
                behavioral_agent=agent,
                reward_shaper=reward_shaper,
                T=T,
                num_episodes=num_episodes,
                seeds=seeds)

            # Average those n_experiments results to fill results_array
            steps_per_episode[il, ia] = np.mean(steps)
            sum_of_rewards_per_episode[il, ia] = np.mean(sum_rewards)

            print(f"\n{type(agent)}: Done α={alpha:.3f}, λ={lam:.3f} → steps/episode: {steps_per_episode[il, ia]:.4f}, sum(r)/episode: {sum_of_rewards_per_episode[il, ia]:.4f}")

    plot_multi_curves(
        steps_per_episode,
        title='Sarsa(λ)',
        labels=lambdas,
        ylabel='Avg. Steps / Episode',
        x=alphas)

    plot_multi_curves(
        sum_of_rewards_per_episode,
        title='Sarsa(λ)',
        labels=lambdas,
        ylabel='Avg. Sum(Rewards) / Episode',
        x=alphas)

if __name__ == '__main__':

    model = 'TrueOnlineSarsaLambda'
    # model = 'SarsaLambda'

    do_log = False
    alphas = np.linspace(0.2, 1.9, num=10)
    lambdas = [0., 0.68, .84, .92, .96, .98, .99]

    RENDER = False
    ENV_NAME = 'MountainCar'

    num_episodes = None
    T = None

    if ENV_NAME == 'MountainCar':
        num_episodes = 50
        T = 999
        MAX_EPISODE_STEPS = T

    def build_greedy_eps_sched(start):
        """
            Sarsa requires pi --> greedy as one of the conditions
            for convergence.
        """
        return LinearSchedule(start, end=0.0, steps=(num_episodes // 8) * T)

    def build_update_coefficient_sched(
            start,
            end,
            steps=(num_episodes // 8) * T
    ):
        return LinearSchedule(start, end=end, steps=steps)

    def base_reward(reward: float, state: np.ndarray, done: bool, t: int):
        return reward

    experiments_parallel(
        model=model,
        num_episodes=num_episodes,
        num_experiments=100,
        T=T,
        reward_shaper=base_reward,
        eps=0.0,
        eps_builder=build_greedy_eps_sched,
        alphas=alphas,
        lambdas=lambdas,
        seeds=[i for i in range(num_episodes)]
    )




