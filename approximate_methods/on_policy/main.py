import os
import random
from typing import List, Callable, Union, Optional

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import gymnasium as gym
from gymnasium import Env

from approximate_methods.on_policy.agents import (
    SemiGradientSarsa, nStepSemiGradientSarsa)

from approximate_methods.tiles3 import IHT, tiles
from approximate_methods.utils import (
    DiscreteActionAgent,
    LinearQEpsGreedyAgent,
    SoftPolicy,
    Experience)
from shared.utils import LinearEpsSchedule

global env_name


def plot(
        returns_over_seeds_over_agent: List,
        legend: List[str],
        title: str = 'Algo',
        save = True
):
    # Function to create raw dataframes from the sequences
    def create_raw_df(sequences, group_name):
        num_seeds = len(sequences)
        num_episodes = len(sequences[0])
        df = pd.DataFrame({
            'episodes': np.tile(np.arange(num_episodes), num_seeds),
            'returns': np.concatenate(sequences),
            'seed': np.repeat(np.arange(num_seeds), num_episodes),
            'group': group_name
        })
        return df

    # List of dataframes
    dfs = []
    for label, return_ in zip(legend, returns_over_seeds_over_agent):
        df = create_raw_df(return_, label)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Calculate mean, min, max, and std values for each group over the seeds
    df_grouped = df.groupby(['episodes', 'group']).agg(
        mean=('returns', 'mean'),
        std=('returns', 'std'),
        min=('returns', 'min'),
        max=('returns', 'max')
    ).reset_index()

    # Plot using Seaborn and Matplotlib
    plt.figure(figsize=(10, 6))

    # Plot the mean with seaborn
    sns.lineplot(x='episodes', y='mean', hue='group', data=df_grouped,
                 palette="deep")

    # Add the standard deviation as a shaded area, constrained by min and max values
    for group in df_grouped['group'].unique():
        group_data = df_grouped[df_grouped['group'] == group]

        # Clip std deviation within min and max bounds
        lower_bound = np.maximum(group_data['mean'] - group_data['std'],
                                 group_data['min'])
        upper_bound = np.minimum(group_data['mean'] + group_data['std'],
                                 group_data['max'])

        plt.fill_between(group_data['episodes'], lower_bound, upper_bound,
                         alpha=0.3)

    # Plot the min/max as dashed lines AFTER the standard deviation plot
    for group in df_grouped['group'].unique():
        group_data = df_grouped[df_grouped['group'] == group]
        plt.plot(group_data['episodes'], group_data['min'], linestyle='--',
                 color='gray', alpha=0.7)
        plt.plot(group_data['episodes'], group_data['max'], linestyle='--',
                 color='gray', alpha=0.7)

    # Add labels and title
    plt.xlabel('Episodes')
    plt.ylabel('Returns = Sum(Rewards)')
    plt.title(title)
    plt.legend(title='Agent')

    if save:
        save_dir = 'images/results'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        trail = '_'.join(legend)
        plt.savefig(f'{save_dir}/{title}_{trail}.png') if save else None
    else:
        plt.show()


def value_plot(
        V0_over_seeds_over_agent: List,
        G0_over_seeds_over_agent: List,
        legend: List[str],
        title: str = 'Algo',
        alpha = 0.5,
        save = True
):
    def create_raw_df(sequences, group_name):
        num_seeds = len(sequences)
        num_episodes = len(sequences[0])
        df = pd.DataFrame({
            'episodes': np.tile(np.arange(num_episodes), num_seeds),
            'returns': np.concatenate(sequences),
            'seed': np.repeat(np.arange(num_seeds), num_episodes),
            'group': group_name
        })
        return df

    V0_dfs = []
    for label, r in zip(legend, V0_over_seeds_over_agent):
        df = create_raw_df(r, 'V0: ' + label)
        V0_dfs.append(df)
    V0_df = pd.concat(V0_dfs, ignore_index=True)

    G0_dfs = []
    for label, r in zip(legend, G0_over_seeds_over_agent):
        df = create_raw_df(r, 'G0: ' + label)
        G0_dfs.append(df)
    G0_df = pd.concat(G0_dfs, ignore_index=True)

    # Plot using Seaborn
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        x='episodes',
        y='returns',
        hue='group',
        data=G0_df,
        errorbar='sd',
        palette=("blue",),
        alpha=0.5 * alpha)

    sns.lineplot(
        x='episodes',
        y='returns',
        hue='group',
        data=V0_df,
        errorbar='sd',
        palette=("red",),
        alpha=alpha)

    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(title)
    plt.legend(title='Agent')

    if save:
        save_dir = 'images/results'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        trail = '_'.join(legend)
        plt.savefig(f'{save_dir}/{title}_{trail}.png') if save else None
    else:
        plt.show()


def evaluate_agent(
        env: Env,
        agent: LinearQEpsGreedyAgent,
        greedy_eval: bool = True,
        seed = None,
        T: int = 30,
        num_episodes: int = 10,
        reward_shaper: Callable = lambda reward, state, done, t: reward,

):
    returns_over_episodes = []
    V0_over_episodes = []
    G0_over_episodes = []

    for _ in tqdm(range(num_episodes), desc=f'Eval'):

        rewards_over_time = []

        # Noise state reset (not exploration level)
        agent.reset()

        # gymnasium v26 requires users to set seed while resetting the
        # environment
        state, info = env.reset(seed=seed)

        state_0 = state
        G0 = 0.
        gamma = agent.discount

        for t in range(T):
            if greedy_eval:
                action, _ = agent.get_greedy_action(state)
            else:
                action, _ = agent.act(state)

            next_state, reward_raw, terminated, truncated, info = env.step(action)

            done = terminated or truncated or (t + 1 == T)

            reward = reward_shaper(reward=reward_raw, state = next_state, done=done, t=t)

            G0 += (gamma ** t) * reward_raw

            rewards_over_time.append(reward_raw)

            if done:
                break
            else:
                state = next_state

        if greedy_eval:
            V0 = agent.optimal_state_value(state_0)
        else:
            V0 = agent.state_value(state_0)

        returns_over_episodes.append(sum(rewards_over_time))
        V0_over_episodes.append(V0)
        G0_over_episodes.append(G0)

    return returns_over_episodes, V0_over_episodes, G0_over_episodes


def run_env(
        env: Env,
        behavioral_agent: DiscreteActionAgent,
        target_agent: DiscreteActionAgent = None,
        reward_shaper: Callable = lambda reward, state, done, t: reward,
        T: int = 30,
        num_episodes: int = 10,
        do_eval: bool = True,
        eval_num_episodes: int = 10,
        evaluate_frequency: int = 5,
        greedy_eval: bool = True,
        train_seeds=(1, 2, 3, 4)
):
    V0_over_seeds = []
    eval_V0_over_seeds = []
    G0_over_seeds = []
    eval_G0_over_seeds = []

    returns_over_seeds = []
    eval_returns_over_seeds = []

    for seed_i, seed in enumerate(train_seeds):
        random.seed(seed)
        np.random.seed(seed)

        returns_over_episodes = []
        eval_rreturns_over_episodes = []
        V0_over_episodes = []
        eval_V0_over_episodes = []
        G0_over_episodes = []
        eval_G0_over_episodes = []

        behavioral_agent.initialize()  # Unlearn
        if target_agent is not None:
            target_agent.initialize()  # Unlearn

        gamma = None
        if (target_agent is not None) and isinstance(
                target_agent, LinearQEpsGreedyAgent):
            gamma = target_agent.discount
        elif isinstance(behavioral_agent, LinearQEpsGreedyAgent):
            gamma = behavioral_agent.discount

        for episode in tqdm(
                range(num_episodes), desc=f'Episodes for seed_{seed}'):
            rewards_over_time = []

            # Noise state reset (not exploration level). Clear any trajectories
            behavioral_agent.reset()

            # gymnasium v26 requires users to set seed while resetting the
            # environment
            state, info = env.reset(seed=seed)

            state_0 = state
            G0 = 0.

            # a[0]
            action, p = behavioral_agent.act(state)

            for t in range(T):
                next_state, reward_raw, terminated, truncated, info = \
                    env.step(action)

                done = terminated or truncated or (t + 1 == T)

                reward = reward_shaper(reward=reward_raw, state = next_state, done=done, t=t)

                rewards_over_time.append(reward_raw)

                next_action, next_p = behavioral_agent.act(next_state)

                G0 += (gamma ** t) * reward_raw

                rhop = None
                if (target_agent is not None) and (
                isinstance(target_agent, SoftPolicy)
                ):
                    target_next_p = target_agent.get_sa_probability(
                        next_state, next_action)
                    rhop = target_next_p / next_p

                e = Experience(
                    s=state,
                    a=action,
                    p=p,
                    r=reward,
                    sp=next_state,
                    ap=next_action,
                    pp=next_p,
                    done=int(done),
                    rhop=rhop,
                    t=t
                )

                behavioral_agent.step(e)

                if target_agent is not None:
                    target_agent.step(e)

                if done:
                    break
                else:
                    state = next_state
                    action = next_action
                    p = next_p

            value = None
            if (target_agent is not None) and isinstance(
                    target_agent, LinearQEpsGreedyAgent):
                value = target_agent.state_value(state_0)
            elif isinstance(behavioral_agent, LinearQEpsGreedyAgent):
                value = behavioral_agent.state_value(state_0)

            # The environment terminates on positive reward
            # Opt 1: final reward of the episode
            # rewards_over_episodes.append(rewards_over_time[-1])

            # Opt 2: sum of rewards in episode (for cases where there's
            # always a reward)
            returns_over_episodes.append(sum(rewards_over_time))
            V0_over_episodes.append(value)
            G0_over_episodes.append(G0)

            if (
                    do_eval and
                    (
                        (
                            (episode % evaluate_frequency == 0) and
                            (eval_num_episodes > 0)
                        ) or
                        (episode + 1 == num_episodes) # evaluate last episode
                    )
            ):
                eval_agent = behavioral_agent

                if target_agent is not None:
                    eval_agent = target_agent

                eval_return, eval_v0, eval_g0, = evaluate_agent(
                    env,
                    agent=eval_agent,
                    greedy_eval=greedy_eval,
                    seed=seed,
                    T=T,
                    num_episodes=eval_num_episodes,
                    reward_shaper=reward_shaper
                )

                eval_rreturns_over_episodes += eval_return
                eval_V0_over_episodes += eval_v0
                eval_G0_over_episodes += eval_g0

        returns_over_seeds.append(returns_over_episodes)
        eval_returns_over_seeds.append(eval_rreturns_over_episodes)

        V0_over_seeds.append(V0_over_episodes)
        eval_V0_over_seeds.append(eval_V0_over_episodes)
        G0_over_seeds.append(G0_over_episodes)
        eval_G0_over_seeds.append(eval_G0_over_episodes)

    print(
        "\nAverage Total Rewards / Seed: Train = {0:.2f}, Eval = {1:.2f}".format(
            np.sum(np.array(returns_over_seeds)) / len(train_seeds),
            np.sum(np.array(eval_returns_over_seeds)) / len(train_seeds)
        ),
    )

    return (
        returns_over_seeds,
        eval_returns_over_seeds,
        V0_over_seeds,
        eval_V0_over_seeds,
        G0_over_seeds,
        eval_G0_over_seeds
    )


def build_env(render: bool = False) -> Env:
    global env_name

    if env_name.lower() == 'MountainCar'.lower():
        env = gym.make(
            'MountainCar-v0',
            render_mode="human" if render else None)

    else:
        raise NotImplementedError

    return env


class TileCodingFeature:
    def __init__(
            self,
            max_size: int,
            num_tiles: int,
            num_tilings: int,
            x0_low: float,
            x1_low: float,
            x0_high: float,
            x1_high: float):

        self.iht = IHT(max_size)
        self.x0_low = x0_low
        self.x1_low = x1_low
        self.x0_high = x0_high
        self.x1_high  = x1_high
        self.num_tiles = num_tiles
        self.num_tilings = num_tilings

    def __call__(
            self,
            x: Union[List, np.ndarray],
            a: int, **kwargs
    ) -> np.ndarray:

        x0, x1 = x[0], x[1]
        feats = tiles(
            ihtORsize=self.iht,
            numtilings=self.num_tilings,
            floats=[
                self.num_tiles * x0 / (self.x0_high - self.x0_low),
                self.num_tiles * x1 / (self.x1_high - self.x1_low)
            ],
            ints=[a]
        )

        feats = np.array(feats)

        res = np.zeros((self.iht.size, 1), dtype=np.float32)
        res[feats] = 1.
        return res


def semigradient_sarsa_experiments(
        num_episodes,
        T,
        reward_shaper: Callable,
        eps_builder: Callable = lambda x: x,
        update_coefficient: Optional[Union[float, LinearEpsSchedule]] = None,
        epses=(0.01, 0.1, 1.),
        seeds=(1, 2),
        do_performance_plot=True,
        base_name: str = ''
):
    train_returns_over_seeds_over_over_agent = []
    eval_returns_over_seeds_over_over_agent = []
    train_V0_returns_over_seeds_over_over_agent = []
    eval_V0_returns_over_seeds_over_over_agent = []
    train_G0_returns_over_seeds_over_over_agent = []
    eval_G0_returns_over_seeds_over_over_agent = []

    legend = []

    env = build_env(False)

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

    if update_coefficient is None:
        update_coefficient = 1 / (3 * num_tilings)

    for eps in epses:
        agent = SemiGradientSarsa(
            feature_size=max_size,
            action_space_dims=int(env.action_space.n),
            update_coefficient=update_coefficient,
            feature_fn=feature_fn,
            discount=0.99,
            eps=eps_builder(eps)
        )

        (
            train_returns_over_seeds,
            eval_returns_over_seeds,
            train_V0_over_seeds,
            eval_V0_over_seeds,
            train_G0_over_seeds,
            eval_G0_over_seeds
         ) = run_env(
            env=env,
            behavioral_agent=agent,
            reward_shaper=reward_shaper,
            num_episodes=num_episodes,
            T=T,
            eval_num_episodes=1,
            train_seeds=seeds)

        train_returns_over_seeds_over_over_agent.append(
            train_returns_over_seeds)
        eval_returns_over_seeds_over_over_agent.append(
            eval_returns_over_seeds)
        train_V0_returns_over_seeds_over_over_agent.append(
            train_V0_over_seeds)
        eval_V0_returns_over_seeds_over_over_agent.append(
            eval_V0_over_seeds)
        train_G0_returns_over_seeds_over_over_agent.append(
            train_G0_over_seeds)
        eval_G0_returns_over_seeds_over_over_agent.append(
            eval_G0_over_seeds)

        legend.append(f'eps: {eps}')

        if do_performance_plot:
            plot(train_returns_over_seeds_over_over_agent,
                 legend=legend,
                 title=base_name + f'SemiGradientSarsa_Train')

            plot(eval_returns_over_seeds_over_over_agent,
                 legend=legend,
                 title=base_name + f'SemiGradientSarsa_Eval')

        train_returns_over_seeds_over_over_agent.clear()
        eval_returns_over_seeds_over_over_agent.clear()
        train_V0_returns_over_seeds_over_over_agent.clear()
        eval_V0_returns_over_seeds_over_over_agent.clear()
        train_G0_returns_over_seeds_over_over_agent.clear()
        eval_G0_returns_over_seeds_over_over_agent.clear()
        legend.clear()


def nstep_semigradient_sarsa_experiments(
        num_episodes,
        T,
        n,
        reward_shaper: Callable,
        eps_builder: Callable = lambda x: x,
        update_coefficient: Optional[Union[float, LinearEpsSchedule]] = None,
        epses=(0.01, 0.1, 1.),
        seeds=(1, 2),
        do_performance_plot=True,
        base_name: str = ''):
    train_returns_over_seeds_over_over_agent = []
    eval_returns_over_seeds_over_over_agent = []
    train_V0_returns_over_seeds_over_over_agent = []
    eval_V0_returns_over_seeds_over_over_agent = []
    train_G0_returns_over_seeds_over_over_agent = []
    eval_G0_returns_over_seeds_over_over_agent = []

    legend = []

    env = build_env(False)

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

    if update_coefficient is None:
        update_coefficient = 1 / (3 * num_tilings)

    for eps in epses:
        agent = nStepSemiGradientSarsa(
            feature_size=max_size,
            action_space_dims=int(env.action_space.n),
            n=n,
            update_coefficient=update_coefficient,
            feature_fn=feature_fn,
            discount=0.99,
            eps=eps_builder(eps)
        )

        (
            train_returns_over_seeds,
            eval_returns_over_seeds,
            train_V0_over_seeds,
            eval_V0_over_seeds,
            train_G0_over_seeds,
            eval_G0_over_seeds
        ) = run_env(
            env=env,
            behavioral_agent=agent,
            reward_shaper=reward_shaper,
            num_episodes=num_episodes,
            T=T,
            eval_num_episodes=1,
            train_seeds=seeds)

        train_returns_over_seeds_over_over_agent.append(
            train_returns_over_seeds)
        eval_returns_over_seeds_over_over_agent.append(
            eval_returns_over_seeds)
        train_V0_returns_over_seeds_over_over_agent.append(
            train_V0_over_seeds)
        eval_V0_returns_over_seeds_over_over_agent.append(
            eval_V0_over_seeds)
        train_G0_returns_over_seeds_over_over_agent.append(
            train_G0_over_seeds)
        eval_G0_returns_over_seeds_over_over_agent.append(
            eval_G0_over_seeds)

        legend.append(f'eps: {eps}')

        if do_performance_plot:
            plot(train_returns_over_seeds_over_over_agent,
                 legend=legend,
                 title=base_name + f'nStepSemiGradientSarsa_Train')

            plot(eval_returns_over_seeds_over_over_agent,
                 legend=legend,
                 title=base_name + f'nStepSemiGradientSarsa_Eval')

        train_returns_over_seeds_over_over_agent.clear()
        eval_returns_over_seeds_over_over_agent.clear()
        train_V0_returns_over_seeds_over_over_agent.clear()
        eval_V0_returns_over_seeds_over_over_agent.clear()
        train_G0_returns_over_seeds_over_over_agent.clear()
        eval_G0_returns_over_seeds_over_over_agent.clear()
        legend.clear()

if __name__ == '__main__':
    do_sarsa = False
    do_nstep_sarsa = True

    epses = (0.01, 0.05, 0.1)

    seeds = tuple(range(0, 10))

    env_name = 'MountainCar'

    if env_name == 'MountainCar':
        num_episodes = 200
        # Environment truncates the length of the episode at 999.
        T = 999
    else:
        raise NotImplementedError('Environment not implemented')


    def build_greedy_eps_sched(start):
        """
            Sarsa requires pi --> greedy as one of the conditions for
            convergence.
        :param start:
        :return:
        """
        return LinearEpsSchedule(
            start, end=0.0, steps=(num_episodes // 8) * T)

    def build_update_coefficient_sched(start, end, steps = (num_episodes // 8) * T):
        return LinearEpsSchedule(start, end=end, steps=steps)

    n_steps = 4

    # Base Reward
    if 1:
        def base_reward(reward: float,  state:np.ndarray, done: bool, t: int):
            return reward

        if do_sarsa:
            semigradient_sarsa_experiments(
                num_episodes=num_episodes,
                T=T,
                reward_shaper=base_reward,
                eps_builder=build_greedy_eps_sched,
                update_coefficient=build_update_coefficient_sched(
                    start=1 / (2 * 8), end=1 / (10 * 8)),
                epses=epses,
                seeds=seeds,
                base_name='Base Reward_'
            )

        if do_nstep_sarsa:
            nstep_semigradient_sarsa_experiments(
                num_episodes=num_episodes,
                T=T,
                n=n_steps,
                reward_shaper=base_reward,
                eps_builder=build_greedy_eps_sched,
                update_coefficient=build_update_coefficient_sched(
                    start=1 / (2 * 8), end=1 / (10 * 8)),
                epses=epses,
                seeds=seeds,
                base_name='Base Reward_'
            )

    # Position reward shaping
    if 1:
        def reward_shaper_position(reward: float, state:np.ndarray, done: bool, t: int):
            k = 0.1
            return reward + k * (state[0] - 0.45)

        if do_sarsa:
            semigradient_sarsa_experiments(
                num_episodes=num_episodes,
                T=T,
                reward_shaper=reward_shaper_position,
                update_coefficient=build_update_coefficient_sched(
                    start=1 / (2 * 8), end=1 / (10 * 8)),
                epses=epses,
                seeds=seeds,
                base_name='Position Reward_'
            )

        if do_nstep_sarsa:
            nstep_semigradient_sarsa_experiments(
                num_episodes=num_episodes,
                T=T,
                n=n_steps,
                reward_shaper=reward_shaper_position,
                update_coefficient=build_update_coefficient_sched(
                    start=1 / (2 * 8), end=1 / (10 * 8)),
                epses=epses,
                seeds=seeds,
                base_name='Position Reward_'
            )

    # Position + velocity reward shaping
    if 1:
        def reward_shaper_position_velocity(reward: float, state:np.ndarray, done: bool, t: int):
            k = 0.1
            return reward + k * (state[0] - 0.45)  * (0.07/(abs(state[1]) + 0.001))

        if do_sarsa:
            semigradient_sarsa_experiments(
                num_episodes=num_episodes,
                T=T,
                reward_shaper=reward_shaper_position_velocity,
                update_coefficient=build_update_coefficient_sched(
                    start=1/(2 * 8), end=1/(10 * 8)),
                epses=epses,
                seeds=seeds,
                base_name='Position_and_Velocity Reward_')

        if do_nstep_sarsa:
            nstep_semigradient_sarsa_experiments(
                num_episodes=num_episodes,
                T=T,
                n=n_steps,
                reward_shaper=reward_shaper_position_velocity,
                update_coefficient=build_update_coefficient_sched(
                    start=1 / (2 * 8), end=1 / (10 * 8)),
                epses=epses,
                seeds=seeds,
                base_name='Position_and_Velocity Reward_'
            )

    # Velocity reward shaping
    if 1:
        def reward_shaper_velocity(
                reward: float, state: np.ndarray, done: bool, t: int):
            k = 0.1
            return reward + k * np.sign((state[0] - 0.45)) * (0.07 / (abs(state[1]) + 0.001))

        if do_sarsa:
            semigradient_sarsa_experiments(
                num_episodes=num_episodes,
                T=T,
                reward_shaper=reward_shaper_velocity,
                update_coefficient=build_update_coefficient_sched(
                    start= 1 / (2 * 8), end=1 / (10 * 8)),
                epses=epses,
                seeds=seeds,
                base_name='Velocity Reward_')

        if do_nstep_sarsa:
            nstep_semigradient_sarsa_experiments(
                num_episodes=num_episodes,
                T=T,
                n=n_steps,
                reward_shaper=reward_shaper_velocity,
                update_coefficient=build_update_coefficient_sched(
                    start=1 / (2 * 8), end=1 / (10 * 8)),
                epses=epses,
                seeds=seeds,
                base_name='Velocity Reward_'
            )

    exit(0)

