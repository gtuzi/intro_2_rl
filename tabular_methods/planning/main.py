from typing import Dict, List
import random
from collections import Counter
import pygame
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tabular_methods.planning.agents import TabularDynaQAgent, TabularPrioritizedSweepingAgent
from envMaze import ExtendedMazeEnvironment

from tabular_methods.utils import Experience, LinearSchedule, QEpsGreedyAgent


###############################################


def plot(
        rewards_over_seeds_over_agent: List,
        legend: List[str],
        title: str = 'Algo'
):
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
    sns.lineplot(
        x='episodes', y='rewards', hue='group', data=df, errorbar='sd')
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
        df = create_raw_df(r, 'G0:' + label)
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
    plt.legend(title='Sum(R) & V[0]')
    plt.show()


def rewards_vs_steps_plot(
        cummulative_reward_vs_terminal_steps_over_agent: List,
        legend: List[str],
        title: str = 'Cumulative Rewards'
):
    def create_raw_df(sequences_over_seeds, group_name):
        steps_over_seeds = [
            [step for r, step in seq] for seq in sequences_over_seeds]
        cum_r_over_seeds = [
            [cr for cr, step in seq] for seq in sequences_over_seeds]

        num_seeds = len(steps_over_seeds)
        num_episodes = len(steps_over_seeds[0])

        df = pd.DataFrame({
            # 'steps': np.concatenate(steps_over_seeds),
            'steps': np.array(
                [s for steps in steps_over_seeds for s in steps]),
            # 'cum_r': np.concatenate(cum_r_over_seeds),
            'cum_r': np.array(
                [cr for cum_rs in cum_r_over_seeds for cr in cum_rs]),
            'group': group_name
        })
        return df

    cum_rewards_dfs = []
    max_x_steps = []
    for label, r in zip(
            legend,
            cummulative_reward_vs_terminal_steps_over_agent
    ):
        df = create_raw_df(r, label)
        cum_rewards_dfs.append(df)
        max_x_steps.append(max(df['steps'].values))

    cum_rewards_df = pd.concat(cum_rewards_dfs, ignore_index=True)

    # Plot using Seaborn
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        x='steps',
        y='cum_r',
        hue='group',
        data=cum_rewards_df,
        errorbar='sd'
    )

    plt.xlim(left=-5, right=min(max_x_steps))
    plt.xlabel('Steps')
    plt.ylabel('Cummulative Returns')
    plt.title(title)
    plt.legend(title='Agent')
    plt.show()

    # Zoom in
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        x='steps',
        y='cum_r',
        hue='group',
        data=cum_rewards_df,
        errorbar='sd'
    )

    plt.xlim(left=-1, right=100)
    plt.xlabel('Steps')
    plt.ylabel('Cummulative Returns')
    plt.title(title + ': zoomed in')
    plt.legend(title='Agent')
    plt.show()


def first_shortest_terminal_step_plot(
        first_shortest_terminal_global_step_vs_episode_step_over_agent: List,
        legend: List[str],
        title: str = 'First Shortest Episode'
):
    # A: row == agent, col == seeds
    A = first_shortest_terminal_global_step_vs_episode_step_over_agent

    # Generate a color for each row
    colors = plt.cm.rainbow(np.linspace(0, 1, len(A)))

    # Flatten the list and count occurrences
    all_points = [tuple(point) for row in A for point in row]
    point_counts = Counter(all_points)

    # Find the maximum count for normalization
    max_count = max(point_counts.values())

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each row with its own color
    for row_index, (row, leg, color) in enumerate(zip(A, legend, colors)):
        x, y = zip(*row)

        # Calculate alpha values based on point frequency
        alphas = [
            point_counts[(x[i], y[i])] / max_count for i in range(len(x))
        ]

        # Create a scatter plot for this row
        scatter = ax.scatter(
            x, y, c=[color],
            alpha=alphas,
            label=f'{leg}'
        )

    # Customize the plot
    ax.set_xlabel('Global-Step')
    ax.set_ylabel('First Shortest Episode-Step')
    ax.set_title('First Shortest Episode Step @ Global Step ')
    ax.legend()

    plt.title(title)

    # Show the plot
    plt.tight_layout()

    # Show the plot
    plt.show()

##############################################


def create_new_maze():
    env = ExtendedMazeEnvironment(
        save_maze=True,
        load_maze=False
    )

    _ = env.reset(randomize_start_goal=True)

    # display maze and wait for user SPACE for confirmation
    while not env.space_clicked:
        env.handle_event()
        env.drawGrid()
        env.showChar()
        pygame.display.update()

    env.save_maze()


def get_maze_environment(
        randomize_start_goal=False,
        skip_visual_confirmation: bool = True):
    """
        skip_visual_confirmation: no user confirmation

        Returns: environment, s0
    """

    # Setup the environment dynamically. Spacebar to confirm environment
    env = ExtendedMazeEnvironment(
        save_maze=False,
        load_maze=True
    )

    env.load_maze()
    s0 = env.reset(randomize_start_goal=randomize_start_goal)

    # display maze and wait for user SPACE for confirmation
    while (not env.space_clicked) and (not skip_visual_confirmation):
        env.handle_event()
        env.drawGrid()
        env.showChar()
        pygame.display.update()

    return env, s0


def run_maze_env(
        agent,
        maxEpisodes,
        max_steps_per_episode,
        randomize_start_goal,
        title: str = ''
):
    results = dict(
        raw_rewards=[],
        raw_terminal=[],
        first_shortest_terminal_step=[np.inf, np.inf]  # global step, episode-step
    )

    global_total_steps = 0

    for episode in range(maxEpisodes):
        results[f'episode_{episode}'] = dict(
            rewards=[],
            V0=None,
            G0=0,
            num_steps=0
        )

        agent.reset()

        # Redundant, but environment needs to be reset
        env, s = get_maze_environment(
            randomize_start_goal=randomize_start_goal,
            skip_visual_confirmation=True
        )

        G0 = 0.
        terminal = False

        state_0 = s
        a, p = agent.act(s)

        while (not terminal and
               (
                       (max_steps_per_episode <= 0) or
                       (env._num_ep_steps < max_steps_per_episode)
               ) and not env.close_clicked
        ):

            global_total_steps += 1

            env.handle_event()
            env.drawGrid(state_val_fn=agent.state_value)
            env.showChar()

            r, sp, terminal = env.step(a)

            ap, pp = None, None

            # Some update methods need the next action before they
            # update. Some other do not, so the next step should
            # be taken after the update
            if agent.requires_next_step_before_update:
                # Take the next step
                ap, pp = agent.act(sp)

            env.drawBlackBox(sp)
            pygame.display.update()

            agent.step(
                Experience(
                    s=s,
                    a=a,
                    p=p,
                    r=r,
                    sp=sp,
                    ap=ap,
                    pp=pp,
                    done=int(terminal)
                )
            )

            if not agent.requires_next_step_before_update:
                # Take the next step after agent update
                ap, pp = agent.act(sp)

            t = env.num_steps() - 1
            G0 += (agent.discount ** t) * r

            results[f'episode_{episode}']['rewards'].append(r)
            results['raw_rewards'].append(r)
            results['raw_terminal'].append(int(terminal))

            if not terminal:
                s = sp
                # Move these over for the next "current" part of the experience
                a = ap
                p = pp
            else:
                print(
                    f'{title} - Episode: {episode}, steps taken: {env.num_steps()}')
                results[f'episode_{episode}']['num_steps'] = env.num_steps()
                results[f'episode_{episode}']['G0'] = G0

                if results['first_shortest_terminal_step'][1] > env.num_steps():
                    # At what global step were we able to hit the shortest
                    # episode-step ?
                    results['first_shortest_terminal_step'][0] = (
                        global_total_steps)

                    results['first_shortest_terminal_step'][
                        1] = env.num_steps()

                if isinstance(agent, QEpsGreedyAgent):
                    results[f'episode_{episode}']['V0'] = (
                        agent.state_value(state_0)
                    )

    return results


def run_dynaq(
        model_steps,
        maxEpisodes=200,
        plus_k=None,
        max_steps_per_episode=0,
        randomize_start_goal=False,
        skip_visual_confirmation=False,
        td_update_type='qlearning',
        title: str = ''
):
    """
        n: num simulation steps
        save_load:  True on fresh maze setup
    """

    def build_greedy_eps_sched(start):
        """
            Sarsa requires pi --> greedy as one of the conditions for
            convergence.
        :param start:
        :return:
        """
        return LinearSchedule(
            start,
            end=0.0,
            steps=(maxEpisodes // 4) * max(max_steps_per_episode, 100)
        )

    env, s = get_maze_environment(
        skip_visual_confirmation=skip_visual_confirmation
    )

    n_actions = env.num_actions
    n_states = env.num_states

    agent = TabularDynaQAgent(
        action_space_dims=n_actions,
        obs_space_dims=n_states,
        update_coefficient=0.1,
        dynaq_plus_k=plus_k,
        model_steps=model_steps,
        eps=0.1,
        td_update_type=td_update_type
    )

    agent.initialize()

    results = run_maze_env(
        agent,
        maxEpisodes=maxEpisodes,
        max_steps_per_episode=max_steps_per_episode,
        randomize_start_goal=randomize_start_goal,
        title=f'{title} - DynaQ' + ('Plus' if plus_k is not None else '')
    )

    return results


def dynaq_experiments(
        maxEpisodes=200,
        max_steps_per_episode=0,
        randomize_start_goal=False,
        do_create_new_maze: bool = False,
        skip_visual_confirmation=False,
        model_steps=5,
        seeds=(1, 2, 3),
        td_update_type='qlearning',
        title: str = ''
) -> dict:
    if do_create_new_maze:
        create_new_maze()

    results_over_seeds = dict()

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        results = run_dynaq(
            model_steps,
            maxEpisodes=maxEpisodes,
            max_steps_per_episode=max_steps_per_episode,
            randomize_start_goal=randomize_start_goal,
            skip_visual_confirmation=skip_visual_confirmation,
            td_update_type=td_update_type,
            title=f'{title} - Seed: {seed}'
        )

        results_over_seeds[seed] = results

    return results_over_seeds


def dynaq_plus_experiments(
        maxEpisodes=200,
        max_steps_per_episode=0,
        randomize_start_goal=False,
        do_create_new_maze: bool = False,
        skip_visual_confirmation=False,
        model_steps=5,
        seeds=(1, 2, 3),
        td_update_type='qlearning',
        title: str = ''
) -> dict:
    plus_k = 0.001

    if do_create_new_maze:
        create_new_maze()

    results_over_seeds = dict()

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        results = run_dynaq(
            model_steps,
            maxEpisodes=maxEpisodes,
            max_steps_per_episode=max_steps_per_episode,
            randomize_start_goal=randomize_start_goal,
            plus_k=plus_k,
            skip_visual_confirmation=skip_visual_confirmation,
            td_update_type=td_update_type,
            title=f'{title} - Seed: {seed}'
        )

        results_over_seeds[seed] = results

    return results_over_seeds


def prioritized_sweeping_experiments(
        maxEpisodes=200,
        max_steps_per_episode=0,
        randomize_start_goal=False,
        skip_visual_confirmation=False,
        do_create_new_maze: bool = False,
        model_steps=5,
        seeds=(1, 2, 3),
        title: str = ''
) -> dict:
    if do_create_new_maze:
        create_new_maze()

    def build_greedy_eps_sched(start):
        """
            Sarsa requires pi --> greedy as one of the conditions for
            convergence.
        :param start:
        :return:
        """
        return LinearSchedule(
            start,
            end=0.0,
            steps=(maxEpisodes // 4) * max(max_steps_per_episode, 100)
        )

    env, s = get_maze_environment(
        skip_visual_confirmation=skip_visual_confirmation
    )

    n_actions = env.num_actions
    n_states = env.num_states

    results_over_seeds = dict()

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        agent = TabularPrioritizedSweepingAgent(
            action_space_dims=n_actions,
            obs_space_dims=n_states,
            update_coefficient=0.1,
            model_steps=model_steps,
            eps=0.1,
            priority_threshold=0.00
        )

        agent.initialize()

        results = run_maze_env(
            agent,
            maxEpisodes=maxEpisodes,
            max_steps_per_episode=max_steps_per_episode,
            randomize_start_goal=randomize_start_goal,
            title=f'{title} - Prioritized Sweeping - Seed: {seed}'
        )

        results_over_seeds[seed] = results

    return results_over_seeds


def process_experiment_results(
        results_over_agents: List[Dict],
        agent_names: List,
        do_value_plot: bool = True,
        do_rewards_vs_steps_plot: bool = True,
        do_first_shortest_terminal_step_plot: bool = True
):
    train_returns_over_seeds_over_over_agent = []
    train_V0_returns_over_seeds_over_over_agent = []
    train_G0_returns_over_seeds_over_over_agent = []
    cummulative_reward_vs_terminal_steps_over_agent = []
    first_shortest_terminal_global_step_vs_episode_step_over_agent = []

    legend = []

    for results, agent_name in zip(results_over_agents, agent_names):
        legend.append(agent_name)

        seeds = list(results.keys())
        V0_over_seeds = []
        G0_over_seeds = []
        rewards_over_seeds = []
        raw_rewards_over_seeds = []
        cummulative_reward_vs_terminal_steps_over_seeds = []
        first_shortest_terminal_global_step_vs_episode_step_over_seeds = []

        for seed in seeds:
            V0_over_episodes = []
            G0_over_episodes = []
            rewards_over_episodes = []
            steps_to_goal_over_episode = []
            cummulative_reward_vs_terminal_steps_over_episode = []

            cummulative_reward = 0
            num_episodes = 0

            for e_name in list(results[seed].keys()):
                if 'episode' not in e_name:
                    continue

                num_episodes += 1
                V0_over_episodes.append(results[seed][e_name]['V0'])
                G0_over_episodes.append(results[seed][e_name]['G0'])
                rewards_over_episodes.append(results[seed][e_name]['rewards'])

                num_steps = results[seed][e_name]['num_steps']
                reward_at_terminal = results[seed][e_name]['rewards'][
                    num_steps]
                cummulative_reward += sum(results[seed][e_name]['rewards'])
                steps_to_goal_over_episode.append(
                    results[seed][e_name]['num_steps'])
                cummulative_reward_vs_terminal_steps_over_episode.append(
                    (cummulative_reward, num_steps))

            V0_over_seeds.append(V0_over_episodes)
            G0_over_seeds.append(G0_over_episodes)
            rewards_over_seeds.append(rewards_over_episodes)
            raw_rewards_over_seeds.append(results[seed]['raw_rewards'])
            cummulative_reward_vs_terminal_steps_over_seeds.append(
                cummulative_reward_vs_terminal_steps_over_episode)
            first_shortest_terminal_global_step_vs_episode_step_over_seeds.append(
                results[seed]['first_shortest_terminal_step']
            )

        train_returns_over_seeds_over_over_agent.append(rewards_over_seeds)
        train_V0_returns_over_seeds_over_over_agent.append(V0_over_seeds)
        train_G0_returns_over_seeds_over_over_agent.append(G0_over_seeds)
        cummulative_reward_vs_terminal_steps_over_agent.append(
            cummulative_reward_vs_terminal_steps_over_seeds)

        first_shortest_terminal_global_step_vs_episode_step_over_agent.append(
            first_shortest_terminal_global_step_vs_episode_step_over_seeds
        )

    if do_first_shortest_terminal_step_plot:
        first_shortest_terminal_step_plot(
            first_shortest_terminal_global_step_vs_episode_step_over_agent,
            legend=legend
        )

    if do_rewards_vs_steps_plot:
        rewards_vs_steps_plot(
            cummulative_reward_vs_terminal_steps_over_agent,
            legend=legend
        )

    if do_value_plot:
        for i, agent_name in enumerate(agent_names):
            value_plot(
                V0_over_seeds_over_agent=train_V0_returns_over_seeds_over_over_agent[
                                         i:i + 1],
                G0_over_seeds_over_agent=train_G0_returns_over_seeds_over_over_agent[
                                         i:i + 1],
                legend=legend[i:i + 1],
                title=f'Values - {agent_name}'
            )


if __name__ == '__main__':
    maxEpisodes = 100
    seeds = list(range(20))
    max_steps_per_episode = 0
    model_steps = 5

    results_over_agents = []
    agent_names = []

    if 0:
        results = prioritized_sweeping_experiments(
            maxEpisodes=maxEpisodes,
            max_steps_per_episode=max_steps_per_episode,
            seeds=seeds,
            model_steps=model_steps,
            skip_visual_confirmation=True,
            do_create_new_maze=False
        )
        results_over_agents.append(results)
        agent_names.append('Prioritized Sweep')

    if 1:
        # Select from: {qlearning | sarsa | expected_sarsa}
        td_update_type = 'expected_sarsa'

        results = dynaq_experiments(
            maxEpisodes=maxEpisodes,
            max_steps_per_episode=max_steps_per_episode,
            seeds=seeds,
            model_steps=model_steps,
            skip_visual_confirmation=True,
            do_create_new_maze=False,
            td_update_type=td_update_type
        )
        results_over_agents.append(results)
        agent_names.append('DynaQ')

    if 0:
        # Select from: {qlearning | sarsa | expected_sarsa}
        td_update_type = 'qlearning'
        results = dynaq_plus_experiments(
            maxEpisodes=maxEpisodes,
            max_steps_per_episode=max_steps_per_episode,
            seeds=seeds,
            model_steps=model_steps,
            skip_visual_confirmation=True,
            do_create_new_maze=False,
            td_update_type=td_update_type
        )
        results_over_agents.append(results)
        agent_names.append('DynaQPlus')

    process_experiment_results(
        results_over_agents=results_over_agents,
        agent_names=agent_names,
        do_value_plot=True
    )

    exit(0)
