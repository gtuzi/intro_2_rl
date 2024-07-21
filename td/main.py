import random
from typing import List, Callable
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import randint
import gymnasium as gym
from gymnasium import Env

from utils import (
    LinearEpsSchedule,
    Experience,
    DiscreteActionAgent,
    DiscreteActionRandomAgent,
    QEpsGreedyAgent, SoftPolicy
)

from agents import (
    Sarsa,
    QLearning,
    ExpectedSarsa,
    nStepSarsa,
    nStepsSarsaOffPolicy,
    QSigmaOffPolicy
)

global env_name


def plot(
        rewards_over_seeds_over_agent: List,
        legend: List[str],
        title: str = 'Algo'
):
    # for r in rewards_over_seeds_over_agent:
    #     r = np.array(r)
    #     r = np.mean(r, axis=0)
    #     plt.plot(np.cumsum(r))
    #
    # plt.legend(legend)
    # plt.xlabel('Episodes')
    # plt.ylabel('Cummulative Returns')
    # plt.title(title)
    # plt.show()

    # -------  Raw ------ #
    def create_raw_df(sequences, group_name):
        num_seeds = len(sequences)
        num_episodes = len(sequences[0])
        df = pd.DataFrame({
            'episodes': np.tile(np.arange(num_episodes), num_seeds),
            'rewards': np.concatenate(sequences),
            'seed': np.repeat(np.arange(num_seeds), num_episodes),
            'group': group_name
        })
        return df

    dfs = []
    for label, r in zip(legend, rewards_over_seeds_over_agent):
        df = create_raw_df(r, label)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x='episodes', y='rewards', hue='group', data=df, errorbar='sd')
    plt.xlabel('Episodes')
    plt.ylabel('Raw Rewards')
    plt.title(title)
    plt.legend(title='Agent')
    plt.show()

    # ------ Cummulative Sum --------- #
    def create_cumsum_df(sequences, group_name):
        num_seeds = len(sequences)
        num_episodes = len(sequences[0])

        # Compute the cumulative sum for each sequence
        cumulative_sums = [list(pd.Series(seq).cumsum()) for seq in sequences]

        df = pd.DataFrame({
            'episodes': [t for t in range(num_episodes)] * num_seeds,
            'rewards': [value for seq in cumulative_sums for value in seq],
            'seed': [i for i in range(num_seeds) for _ in range(num_episodes)],
            'group': group_name
        })
        return df

    dfs = []
    for label, r in zip(legend, rewards_over_seeds_over_agent):
        df = create_cumsum_df(r, label)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='episodes', y='rewards', hue='group', data=df,
                 errorbar='sd')
    plt.xlabel('Episodes')
    plt.ylabel('CumSum Rewards')
    plt.title(title)
    plt.legend(title='Agent')
    plt.show()


def value_plot(
        V0_over_seeds_over_agent: List,
        G0_over_seeds_over_agent: List,
        legend: List[str],
        title: str = 'Algo'
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
        df = create_raw_df(r, 'V0:' + label)
        V0_dfs.append(df)
    V0_df = pd.concat(V0_dfs, ignore_index=True)

    G0_dfs = []
    for label, r in zip(legend, G0_over_seeds_over_agent):
        df = create_raw_df(r, 'G0' + label)
        G0_dfs.append(df)
    G0_df = pd.concat(G0_dfs, ignore_index=True)

    # Plot using Seaborn
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        x='episodes',
        y='returns',
        hue='group',
        data=V0_df,
        errorbar='sd',
        palette=("red",))

    sns.lineplot(
        x='episodes',
        y='returns',
        hue='group',
        data=G0_df,
        errorbar='sd',
        palette=("blue",))

    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(title)
    plt.legend(title='Agent')
    plt.show()


def build_env(render: bool = False) -> Env:
    global env_name

    if env_name == 'FrozenLake':
        env = gym.make(
            'FrozenLake-v1',
            render_mode="human" if render else None,
            desc=None,
            map_name="4x4",
            is_slippery=False)

    elif env_name == 'Taxi':
        env = gym.make(
            'Taxi-v3',
            render_mode="human" if render else None
        )
    elif env_name == 'CliffWalking':
        env = gym.make(
            "CliffWalking-v0",
            # render_mode="human",
            # max_episode_steps=100,
            render_mode = "human" if render else None
        )

    else:
        raise NotImplementedError

    return env


def evaluate_agent(
        env: Env,
        agent: QEpsGreedyAgent,
        greedy_eval: bool = True,
        seed = None,
        T: int = 30,
        num_episodes: int = 10,
        reward_shaper: Callable = lambda reward, done, t: reward,

):
    rewards_over_episodes = []
    S0_updates_over_episodes = []
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

            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated or (t + 1 == T)

            reward = reward_shaper(reward=reward, done=done, t=t)

            G0 += (gamma ** t) * reward

            rewards_over_time.append(reward)

            if done:
                break
            else:
                state = next_state

        if greedy_eval:
            V0 = agent.optimal_state_value(state_0)
            S0_update_count = agent.optimal_state_update_count(state_0)
        else:
            V0 = agent.state_value(state_0)
            S0_update_count = agent.state_update_count(state_0)


        rewards_over_episodes.append(sum(rewards_over_time))
        V0_over_episodes.append(V0)
        G0_over_episodes.append(G0)
        S0_updates_over_episodes.append(S0_update_count)

    return rewards_over_episodes, V0_over_episodes, G0_over_episodes


def run_env(
        env: Env,
        behavioral_agent: DiscreteActionAgent,
        target_agent: DiscreteActionAgent = None,
        reward_shaper: Callable = lambda reward, done, t: reward,
        sigma_fn: Callable[[int], float] = lambda t: 1.,
        T: int = 30,
        num_episodes: int = 10,
        do_eval: bool = True,
        eval_num_episodes: int = 10,
        evaluate_frequency: int = 50,
        greedy_eval: bool = True,
        train_seeds=(1, 2, 3, 4)
):
    V0_over_seeds = []
    eval_V0_over_seeds = []
    G0_over_seeds = []
    eval_G0_over_seeds = []

    rewards_over_seeds = []
    eval_rewards_over_seeds = []

    for seed_i, seed in enumerate(train_seeds):
        random.seed(seed)
        np.random.seed(seed)

        rewards_over_episodes = []
        eval_rewards_over_episodes = []
        V0_over_episodes = []
        eval_V0_over_episodes = []
        G0_over_episodes = []
        eval_G0_over_episodes = []

        behavioral_agent.initialize()  # Unlearn
        if target_agent is not None:
            target_agent.initialize()  # Unlearn

        gamma = None
        if (target_agent is not None) and isinstance(
                target_agent, QEpsGreedyAgent):
            gamma = target_agent.discount
        elif isinstance(behavioral_agent, QEpsGreedyAgent):
            gamma = behavioral_agent.discount

        for episode in tqdm(
                range(num_episodes), desc=f'Episodes for seed_{seed_i}'):
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
                next_state, reward, terminated, truncated, info = \
                    env.step(action)

                done = terminated or truncated or (t + 1 == T)

                reward = reward_shaper(reward=reward, done=done, t=t)

                rewards_over_time.append(reward)

                next_action, next_p = behavioral_agent.act(next_state)

                sigmap = sigma_fn(t + 1)

                G0 += (gamma ** t) * reward

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
                    sigmap=sigmap,
                    rhop=rhop,
                    t=t  # To debug the indexing of the trajectory
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
                    target_agent, QEpsGreedyAgent):
                value = target_agent.state_value(state_0)
            elif isinstance(behavioral_agent, QEpsGreedyAgent):
                value = behavioral_agent.state_value(state_0)

            # The environment terminates on positive reward
            # Opt 1: final reward of the episode
            # rewards_over_episodes.append(rewards_over_time[-1])

            # Opt 2: sum of rewards in episode (for cases where there's
            # always a reward)
            rewards_over_episodes.append(sum(rewards_over_time))
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

                eval_r, eval_v0, eval_g0, = evaluate_agent(
                    env,
                    agent=eval_agent,
                    greedy_eval=greedy_eval,
                    seed=seed,
                    T=T,
                    num_episodes=eval_num_episodes,
                    reward_shaper=reward_shaper
                )

                eval_rewards_over_episodes += eval_r
                eval_V0_over_episodes += eval_v0
                eval_G0_over_episodes += eval_g0

        rewards_over_seeds.append(rewards_over_episodes)
        eval_rewards_over_seeds.append(eval_rewards_over_episodes)

        V0_over_seeds.append(V0_over_episodes)
        eval_V0_over_seeds.append(eval_V0_over_episodes)
        G0_over_seeds.append(G0_over_episodes)
        eval_G0_over_seeds.append(eval_G0_over_episodes)

    print(
        "\nAverage Total Rewards / Seed: Train = {0:.2f}, Eval = {1:.2f}".format(
            np.sum(np.array(rewards_over_seeds)) / len(train_seeds),
            np.sum(np.array(eval_rewards_over_seeds)) / len(train_seeds)
        ),
    )

    return (
        rewards_over_seeds,
        eval_rewards_over_seeds,
        V0_over_seeds,
        eval_V0_over_seeds,
        G0_over_seeds,
        eval_G0_over_seeds
    )


def sarsa_experiments(
        num_episodes,
        T,
        q_init,
        reward_shaper,
        epses=(0.01, 0.1, 1.),
        alpha=0.2,
        do_random=True,
        train_seeds=(1, 2),
        do_value_plot=True,
        do_performance_plot=True
):
    train_returns_over_seeds_over_over_agent = []
    eval_returns_over_seeds_over_over_agent = []
    train_V0_returns_over_seeds_over_over_agent = []
    eval_V0_returns_over_seeds_over_over_agent = []
    train_G0_returns_over_seeds_over_over_agent = []
    eval_G0_returns_over_seeds_over_over_agent = []

    legend = []
    env = build_env()

    if do_random:
        random_agent = DiscreteActionRandomAgent(
            action_space_dims=int(env.action_space.n),
            obs_space_dims=int(env.observation_space.n),
            distribution=randint,
            distribution_args=dict(low=0, high=int(env.action_space.n))
        )

        train_returns_over_seeds, eval_returns_over_seeds = run_env(
            env,
            random_agent,
            T=T,
            num_episodes=num_episodes,
            reward_shaper=reward_shaper,
            train_seeds=train_seeds
        )

        train_returns_over_seeds_over_over_agent.append(
            train_returns_over_seeds)
        eval_returns_over_seeds_over_over_agent.append(eval_returns_over_seeds)
        legend.append('Random')

    def build_greedy_eps_sched(start):
        """
            Sarsa requires pi --> greedy as one of the conditions for
            convergence.
        :param start:
        :return:
        """
        return LinearEpsSchedule(
            start, end=0.0, steps=(num_episodes // 4) * T)

    def build_fixed_eps_sched(start):
        return LinearEpsSchedule(
            start, end=start, steps=(num_episodes // 4) * T)

    for eps in epses:
        agent = Sarsa(
            action_space_dims=int(env.action_space.n),
            obs_space_dims=int(env.observation_space.n),
            discount=0.999,
            eps=build_greedy_eps_sched(eps),
            qval_init=q_init,
            update_coefficient=alpha
        )

        (
            train_returns_over_seeds,
            eval_returns_over_seeds,
            train_V0_over_seeds,
            eval_V0_over_seeds,
            train_G0_over_seeds,
            eval_G0_over_seeds
        ) = run_env(
            env,
            agent,
            greedy_eval=True,
            evaluate_frequency=max(1, num_episodes // 2),
            T=T,
            num_episodes=num_episodes,
            reward_shaper=reward_shaper,
            train_seeds=train_seeds
        )

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

        legend.append(f'eps: {eps}, alpha: {alpha}')

    if do_value_plot:
        value_plot(
            V0_over_seeds_over_agent=train_V0_returns_over_seeds_over_over_agent,
            G0_over_seeds_over_agent=train_G0_returns_over_seeds_over_over_agent,
            legend=legend,
            title=f'Train - Sarsa Returns\nQinit: {q_init}'
        )
        value_plot(
            V0_over_seeds_over_agent=eval_V0_returns_over_seeds_over_over_agent,
            G0_over_seeds_over_agent=eval_G0_returns_over_seeds_over_over_agent,
            legend=legend,
            title=f'Eval - Sarsa Returns\nQinit: {q_init}'
        )

    if do_performance_plot:
        plot(train_returns_over_seeds_over_over_agent,
             legend=legend,
             title=f'Sarsa Train\nQinit: {q_init}')

        plot(eval_returns_over_seeds_over_over_agent,
             legend=legend,
             title=f'Sarsa Eval\nQinit: {q_init}')


def expected_sarsa_experiments(
        num_episodes,
        T,
        q_init,
        reward_shaper,
        epses=(0.01, 0.1, 1.),
        alpha=0.2,
        do_random=True,
        train_seeds=(1, 2),
        do_value_plot=True,
        do_performance_plot=True
):
    train_returns_over_seeds_over_over_agent = []
    eval_returns_over_seeds_over_over_agent = []
    train_V0_returns_over_seeds_over_over_agent = []
    eval_V0_returns_over_seeds_over_over_agent = []
    train_G0_returns_over_seeds_over_over_agent = []
    eval_G0_returns_over_seeds_over_over_agent = []

    legend = []
    env = build_env()

    if do_random:
        random_agent = DiscreteActionRandomAgent(
            action_space_dims=int(env.action_space.n),
            obs_space_dims=int(env.observation_space.n),
            distribution=randint,
            distribution_args=dict(low=0, high=int(env.action_space.n))
        )

        train_returns_over_seeds, eval_returns_over_seeds = run_env(
            env,
            random_agent,
            T=T,
            num_episodes=num_episodes,
            reward_shaper=reward_shaper,
            train_seeds=train_seeds
        )

        train_returns_over_seeds_over_over_agent.append(
            train_returns_over_seeds)
        eval_returns_over_seeds_over_over_agent.append(eval_returns_over_seeds)
        legend.append('Random')

    def build_greedy_eps_sched(start):
        return LinearEpsSchedule(
            start, end=0.0, steps=(num_episodes // 4) * T)

    def build_fixed_eps_sched(start):
        return LinearEpsSchedule(
            start, end=start, steps=(num_episodes // 4) * T)

    for eps in epses:
        agent = ExpectedSarsa(
            action_space_dims=int(env.action_space.n),
            obs_space_dims=int(env.observation_space.n),
            discount=0.999,
            eps=build_greedy_eps_sched(eps),
            qval_init=q_init,
            update_coefficient=alpha
        )

        (
            train_returns_over_seeds,
            eval_returns_over_seeds,
            train_V0_over_seeds,
            eval_V0_over_seeds,
            train_G0_over_seeds,
            eval_G0_over_seeds
        ) = run_env(
            env,
            agent,
            greedy_eval=True,
            T=T,
            num_episodes=num_episodes,
            reward_shaper=reward_shaper,
            train_seeds=train_seeds
        )

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

        legend.append(f'eps: {eps}, alpha: {alpha}')

    if do_value_plot:
        value_plot(
            V0_over_seeds_over_agent=train_V0_returns_over_seeds_over_over_agent,
            G0_over_seeds_over_agent=train_G0_returns_over_seeds_over_over_agent,
            legend=legend,
            title=f'Train - Expected-Sarsa Returns\nQinit: {q_init}'
        )

        value_plot(
            V0_over_seeds_over_agent=eval_V0_returns_over_seeds_over_over_agent,
            G0_over_seeds_over_agent=eval_G0_returns_over_seeds_over_over_agent,
            legend=legend,
            title=f'Eval - Expected-Sarsa Returns\nQinit: {q_init}'
        )

    if do_performance_plot:
        plot(train_returns_over_seeds_over_over_agent,
             legend=legend,
             title=f'Expected-Sarsa Train\nQinit: {q_init}')

        plot(eval_returns_over_seeds_over_over_agent,
             legend=legend,
             title=f'Expected-Sarsa Eval\nQinit: {q_init}')


def qlearning_experiments(
        num_episodes,
        T,
        q_init,
        reward_shaper,
        epses=(0.01, 0.1, 1.),
        alpha=0.2,
        do_random=True,
        train_seeds=(1, 2),
        do_value_plot=True,
        do_performance_plot=True
):
    train_returns_over_seeds_over_over_agent = []
    eval_returns_over_seeds_over_over_agent = []
    train_V0_returns_over_seeds_over_over_agent = []
    eval_V0_returns_over_seeds_over_over_agent = []
    train_G0_returns_over_seeds_over_over_agent = []
    eval_G0_returns_over_seeds_over_over_agent = []

    legend = []
    env = build_env()

    if do_random:
        random_agent = DiscreteActionRandomAgent(
            action_space_dims=int(env.action_space.n),
            obs_space_dims=int(env.observation_space.n),
            distribution=randint,
            distribution_args=dict(low=0, high=int(env.action_space.n))
        )

        train_returns_over_seeds, eval_returns_over_seeds = run_env(
            env,
            random_agent,
            T=T,
            num_episodes=num_episodes,
            reward_shaper=reward_shaper,
            train_seeds=train_seeds
        )

        train_returns_over_seeds_over_over_agent.append(
            train_returns_over_seeds)
        eval_returns_over_seeds_over_over_agent.append(eval_returns_over_seeds)
        legend.append('Random')

    def build_greedy_eps_sched(start):
        return LinearEpsSchedule(
            start, end=0.0, steps=(num_episodes // 4) * T)

    def build_fixed_eps_sched(start):
        return LinearEpsSchedule(
            start, end=start, steps=(num_episodes // 4) * T)

    for eps in epses:
        agent = QLearning(
            action_space_dims=int(env.action_space.n),
            obs_space_dims=int(env.observation_space.n),
            discount=0.999,
            eps=build_greedy_eps_sched(eps),
            qval_init=q_init,
            update_coefficient=alpha
        )

        (
            train_returns_over_seeds,
            eval_returns_over_seeds,
            train_V0_over_seeds,
            eval_V0_over_seeds,
            train_G0_over_seeds,
            eval_G0_over_seeds
        ) = run_env(
            env,
            agent,
            greedy_eval=True,
            T=T,
            num_episodes=num_episodes,
            reward_shaper=reward_shaper,
            train_seeds=train_seeds
        )

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

        legend.append(f'eps: {eps}, alpha: {alpha}')

    if do_value_plot:
        value_plot(
            V0_over_seeds_over_agent=train_V0_returns_over_seeds_over_over_agent,
            G0_over_seeds_over_agent=train_G0_returns_over_seeds_over_over_agent,
            legend=legend,
            title=f'Train Q-learning Returns\nQinit: {q_init}')

        value_plot(
            V0_over_seeds_over_agent=eval_V0_returns_over_seeds_over_over_agent,
            G0_over_seeds_over_agent=eval_G0_returns_over_seeds_over_over_agent,
            legend=legend,
            title=f'Eval Q-learning Returns\nQinit: {q_init}')

    if do_performance_plot:
        plot(train_returns_over_seeds_over_over_agent,
             legend=legend,
             title=f'Q-learning Train\nQinit: {q_init}')

        plot(eval_returns_over_seeds_over_over_agent,
             legend=legend,
             title=f'Q-learning Eval\nQinit: {q_init}')


def nstep_sarsa_experiments(
        num_episodes,
        T,
        q_init,
        reward_shaper,
        n=1,
        epses=(0.01, 0.1, 1.),
        alpha=0.2,
        do_random=True,
        train_seeds=(1, 2),
        do_value_plot=True,
        do_performance_plot=True
):
    train_returns_over_seeds_over_over_agent = []
    eval_returns_over_seeds_over_over_agent = []
    train_V0_returns_over_seeds_over_over_agent = []
    eval_V0_returns_over_seeds_over_over_agent = []
    train_G0_returns_over_seeds_over_over_agent = []
    eval_G0_returns_over_seeds_over_over_agent = []

    legend = []

    env = build_env()

    if do_random:
        random_agent = DiscreteActionRandomAgent(
            action_space_dims=int(env.action_space.n),
            obs_space_dims=int(env.observation_space.n),
            distribution=randint,
            distribution_args=dict(low=0, high=int(env.action_space.n))
        )

        train_returns_over_seeds, eval_returns_over_seeds = run_env(
            env,
            random_agent,
            T=T,
            num_episodes=num_episodes,
            reward_shaper=reward_shaper,
            train_seeds=train_seeds
        )

        train_returns_over_seeds_over_over_agent.append(
            train_returns_over_seeds)
        eval_returns_over_seeds_over_over_agent.append(eval_returns_over_seeds)
        legend.append('Random')

    def build_greedy_eps_sched(start):
        return LinearEpsSchedule(
            start, end=0.0, steps=(num_episodes // 4) * T)

    def build_fixed_eps_sched(start):
        return LinearEpsSchedule(
            start, end=start, steps=(num_episodes // 4) * T)

    for eps in epses:
        agent = nStepSarsa(
            action_space_dims=int(env.action_space.n),
            obs_space_dims=int(env.observation_space.n),
            n=n,
            discount=0.999,
            eps=build_greedy_eps_sched(eps),
            qval_init=q_init,
            update_coefficient=alpha
        )

        (
            train_returns_over_seeds,
            eval_returns_over_seeds,
            train_V0_over_seeds,
            eval_V0_over_seeds,
            train_G0_over_seeds,
            eval_G0_over_seeds
        ) = run_env(
            env,
            agent,
            greedy_eval=True,
            T=T,
            num_episodes=num_episodes,
            reward_shaper=reward_shaper,
            do_eval=True,
            train_seeds=train_seeds
        )

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

        legend.append(f'eps: {eps}, n: {n}, alpha: {alpha}')

    if do_value_plot:
        value_plot(
            V0_over_seeds_over_agent=train_V0_returns_over_seeds_over_over_agent,
            G0_over_seeds_over_agent=train_G0_returns_over_seeds_over_over_agent,
            legend=legend,
            title=f'Train - nStepSarsa Returns\nQinit: {q_init}')

        value_plot(
            V0_over_seeds_over_agent=eval_V0_returns_over_seeds_over_over_agent,
            G0_over_seeds_over_agent=eval_G0_returns_over_seeds_over_over_agent,
            legend=legend,
            title=f'Eval - nStepSarsa Returns\nQinit: {q_init}')

    if do_performance_plot:
        plot(train_returns_over_seeds_over_over_agent,
             legend=legend,
             title=f'nStepSarsa Train\nQinit: {q_init}')

        plot(eval_returns_over_seeds_over_over_agent,
             legend=legend,
             title=f'nStepSarsa Eval\nQinit: {q_init}')


def offpolicy_nstep_sarsa_experiments(
        num_episodes,
        T,
        q_init,
        reward_shaper,
        n=1,
        epses=(0.01, 0.1, 1.),
        alpha=0.2,
        do_random=True,
        train_seeds=(1, 2),
        do_value_plot=True,
        do_performance_plot=True
):
    train_returns_over_seeds_over_over_agent = []
    eval_returns_over_seeds_over_over_agent = []
    train_V0_returns_over_seeds_over_over_agent = []
    eval_V0_returns_over_seeds_over_over_agent = []
    train_G0_returns_over_seeds_over_over_agent = []
    eval_G0_returns_over_seeds_over_over_agent = []

    legend = []

    env = build_env()

    random_agent = DiscreteActionRandomAgent(
        action_space_dims=int(env.action_space.n),
        obs_space_dims=int(env.observation_space.n),
        distribution=randint,
        distribution_args=dict(low=0, high=int(env.action_space.n))
    )

    if do_random:
        train_returns_over_seeds, eval_returns_over_seeds = run_env(
            env,
            random_agent,
            T=T,
            num_episodes=num_episodes,
            reward_shaper=reward_shaper,
            train_seeds=train_seeds
        )

        train_returns_over_seeds_over_over_agent.append(
            train_returns_over_seeds)
        eval_returns_over_seeds_over_over_agent.append(eval_returns_over_seeds)
        legend.append('Random')

    def build_greedy_eps_sched(start):
        return LinearEpsSchedule(
            start, end=0.0, steps=(num_episodes // 4) * T)

    def build_fixed_eps_sched(start):
        return LinearEpsSchedule(
            start, end=start, steps=(num_episodes // 4) * T)

    behavior_agent = random_agent

    for eps in epses:
        agent = nStepsSarsaOffPolicy(
            action_space_dims=int(env.action_space.n),
            obs_space_dims=int(env.observation_space.n),
            n=n,
            discount=0.999,
            eps=build_greedy_eps_sched(eps),
            qval_init=q_init,
            update_coefficient=alpha
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
            target_agent=agent,
            behavioral_agent=behavior_agent,
            greedy_eval=True,
            T=T,
            num_episodes=num_episodes,
            reward_shaper=reward_shaper,
            do_eval=True,
            evaluate_frequency=max(1, num_episodes // 4),
            train_seeds=train_seeds
        )

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

        legend.append(f'eps: {eps}, n: {n}, alpha: {alpha}')

    if do_value_plot:
        value_plot(
            V0_over_seeds_over_agent=train_V0_returns_over_seeds_over_over_agent,
            G0_over_seeds_over_agent=train_G0_returns_over_seeds_over_over_agent,
            legend=legend,
            title=f'Train - Off-Policy nStepSarsa Returns\nQinit: {q_init}'
        )

        value_plot(
            V0_over_seeds_over_agent=eval_V0_returns_over_seeds_over_over_agent,
            G0_over_seeds_over_agent=eval_G0_returns_over_seeds_over_over_agent,
            legend=legend,
            title=f'Eval - Off-Policy nStepSarsa Returns\nQinit: {q_init}'
        )

    if do_performance_plot:
        plot(train_returns_over_seeds_over_over_agent,
             legend=legend,
             title=f'Off-Policy nStepSarsa Train\nQinit: {q_init}')

        plot(eval_returns_over_seeds_over_over_agent,
             legend=legend,
             title=f'Off-Policy nStepSarsa Eval\nQinit: {q_init}')


def offpolicy_nstep_qsigma_experiments(
        num_episodes,
        T,
        q_init,
        reward_shaper,
        n=1,
        sigma_fn=lambda t: 1.,
        epses=(0.01, 0.1, 1.),
        alpha=0.2,
        do_random=True,
        train_seeds=(1, 2),
        do_value_plot=True,
        do_performance_plot=True
):
    train_returns_over_seeds_over_over_agent = []
    eval_returns_over_seeds_over_over_agent = []
    train_V0_returns_over_seeds_over_over_agent = []
    eval_V0_returns_over_seeds_over_over_agent = []
    train_G0_returns_over_seeds_over_over_agent = []
    eval_G0_returns_over_seeds_over_over_agent = []

    legend = []

    env = build_env()

    random_agent = DiscreteActionRandomAgent(
        action_space_dims=int(env.action_space.n),
        obs_space_dims=int(env.observation_space.n),
        distribution=randint,
        distribution_args=dict(low=0, high=int(env.action_space.n))
    )

    if do_random:
        train_returns_over_seeds, eval_returns_over_seeds = run_env(
            env,
            random_agent,
            T=T,
            num_episodes=num_episodes,
            reward_shaper=reward_shaper,
            train_seeds=train_seeds
        )

        train_returns_over_seeds_over_over_agent.append(
            train_returns_over_seeds)
        eval_returns_over_seeds_over_over_agent.append(eval_returns_over_seeds)
        legend.append('Random')

    def build_greedy_eps_sched(start):
        return LinearEpsSchedule(
            start, end=0.0, steps=(num_episodes // 4) * T)

    def build_fixed_eps_sched(start):
        return LinearEpsSchedule(
            start, end=start, steps=(num_episodes // 4) * T)

    behavior_agent = random_agent

    for eps in epses:
        agent = QSigmaOffPolicy(
            action_space_dims=int(env.action_space.n),
            obs_space_dims=int(env.observation_space.n),
            n=n,
            discount=0.999,
            eps=build_greedy_eps_sched(eps),
            qval_init=q_init,
            update_coefficient=alpha
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
            target_agent=agent,
            behavioral_agent=behavior_agent,
            greedy_eval=True,
            T=T,
            num_episodes=num_episodes,
            reward_shaper=reward_shaper,
            sigma_fn=sigma_fn,
            do_eval=True,
            train_seeds=train_seeds
        )

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

        legend.append(f'eps: {eps}, n: {n}, alpha: {alpha}')

    if do_value_plot:
        value_plot(
            V0_over_seeds_over_agent=train_V0_returns_over_seeds_over_over_agent,
            G0_over_seeds_over_agent=train_G0_returns_over_seeds_over_over_agent,
            legend=legend,
            title=f'Train - Off-Policy nStep QSigma Returns\nQinit: {q_init}')

        value_plot(
            V0_over_seeds_over_agent=eval_V0_returns_over_seeds_over_over_agent,
            G0_over_seeds_over_agent=eval_G0_returns_over_seeds_over_over_agent,
            legend=legend,
            title=f'Eval - Off-Policy nStep QSigma Returns\nQinit: {q_init}')

    if do_performance_plot:
        plot(train_returns_over_seeds_over_over_agent,
             legend=legend,
             title=f'Off-Policy nStep QSigma Train\nQinit: {q_init}')

        plot(eval_returns_over_seeds_over_over_agent,
             legend=legend,
             title=f'Off-Policy nStep QSigma Eval\nQinit: {q_init}')


if __name__ == '__main__':
    num_episodes = 500

    T = 100
    # T = 50

    q_init = 0.0
    epses = (0.1,)
    do_random = False
    seeds = (1, 2)
    alpha = 0.2

    # env_name = 'Taxi'
    # env_name = 'FrozenLake'
    env_name = 'CliffWalking'


    # def reward_shaper(reward: float, done: bool, t: int):
    #     k = 0.001
    #     return reward - (k * (t + 1)) if done else -(k * (t + 1))

    def reward_shaper(reward: float, done: bool, t: int):
        return reward


    if 1:
        sarsa_experiments(
            num_episodes=num_episodes,
            T=T,
            q_init=q_init,
            reward_shaper=reward_shaper,
            epses=epses,
            alpha=alpha,
            do_random=do_random,
            train_seeds=seeds
        )

    if 0:
        expected_sarsa_experiments(
            num_episodes=num_episodes,
            T=T,
            q_init=q_init,
            reward_shaper=reward_shaper,
            epses=epses,
            alpha=alpha,
            do_random=do_random,
            train_seeds=seeds
        )

    if 0:
        qlearning_experiments(
            num_episodes=num_episodes,
            T=T,
            q_init=q_init,
            reward_shaper=reward_shaper,
            epses=epses,
            alpha=alpha,
            do_random=do_random,
            train_seeds=seeds)

    if 0:
        nstep_sarsa_experiments(
            num_episodes=num_episodes,
            T=T,
            q_init=q_init,
            reward_shaper=reward_shaper,
            n=2,
            epses=epses,
            alpha=alpha,
            do_random=do_random,
            train_seeds=seeds
        )

    if 0:
        # Note: behavioral agent is random (U), so expect
        # the training returns to be bad.
        offpolicy_nstep_sarsa_experiments(
            num_episodes=num_episodes,
            T=T,
            q_init=q_init,
            reward_shaper=reward_shaper,
            n=5,
            epses=epses,
            alpha=alpha,
            do_random=do_random,
            train_seeds=seeds,
            do_value_plot=True,
            do_performance_plot=True
        )

    if 1:
        # TODO: not tuned at all
        # Note: behavioral agent is random (U), so expect
        # the training returns to be bad.
        # sigma_fn = lambda t: 0.5
        sigma_fn = lambda t: 0 if (t % 2 == 0) else 1.
        offpolicy_nstep_qsigma_experiments(
            num_episodes=num_episodes,
            T=T,
            q_init=q_init,
            reward_shaper=reward_shaper,
            n=5,
            sigma_fn=sigma_fn,
            epses=epses,
            alpha=alpha,
            do_random=do_random,
            train_seeds=seeds
        )

    exit(0)
