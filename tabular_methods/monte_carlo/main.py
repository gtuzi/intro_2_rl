import os
import random
from typing import List, Callable
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

import numpy as np
from scipy.stats import randint
import gymnasium as gym
from gymnasium import Env

from agents import MCOnPolicyFirstVisitGLIE, MCOffPolicy

from tabular_methods.utils import (
    LinearSchedule,
    Experience,
    DiscreteActionAgent,
    DiscreteActionRandomAgent
)


def plot(
        rewards_over_seeds_over_agent: List,
        legend: List[str],
        title: str = 'Algo',
        save: bool = True
):

    _ = plt.figure()

    for rewards_over_seeds_over_agent in rewards_over_seeds_over_agent:
        rewards_over_seeds_np = np.array(rewards_over_seeds_over_agent)
        rewards_over_seeds_np = np.mean(rewards_over_seeds_np, axis=0)
        plt.plot(np.cumsum(rewards_over_seeds_np))

    plt.legend(legend)
    plt.xlabel('Episodes')
    plt.ylabel('Mean(Sum(r))')
    plt.title(title)

    if save:
        save_dir = 'images/results'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fname = f'{save_dir}/{title}'
        fname = fname.replace(' ', '')
        fname = fname.replace(":", "_")
        fname = fname.replace("-", "_")
        fname = fname.replace(".", "_")
        fname = fname.replace("\n", "_")
        fname += '.png'
        plt.savefig(fname) if save else None
    else:
        plt.show()


def build_env(render: bool = False) -> Env:
    env = gym.make(
        'FrozenLake-v1',
        render_mode="human" if render else None,
        desc=None,
        map_name="4x4",
        is_slippery=False)

    return env


def evaluate_agent(
        env: Env,
        agent: DiscreteActionAgent,
        T: int = 30,
        num_episodes: int = 10,
        reward_shaper: Callable = lambda reward, done, t: reward
):
    rewards_over_episodes = []

    for _ in tqdm(range(num_episodes), desc=f'Eval'
                  ):
        rewards_over_time = []

        # Noise state reset (not exploration level)
        agent.reset()

        # gymnasium v26 requires users to set seed while resetting the
        # environment
        state, info = env.reset()

        for t in range(T):
            action, p = agent.get_greedy_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated or (t + 1 == T)

            reward = reward_shaper(reward=reward, done=done, t=t)

            rewards_over_time.append(reward)

            if done:
                break
            else:
                state = next_state

        rewards_over_episodes.append(sum(rewards_over_time))

    return rewards_over_episodes


def run_env(
        env: Env,
        behavioral_agent: DiscreteActionAgent,
        target_agent: DiscreteActionAgent = None,
        reward_shaper: Callable = lambda reward, done, t: reward,
        T: int = 30,
        num_episodes: int = 10,
        eval_num_episodes: int = 10,
        evaluate_frequency: int = 50,
        train_seeds=(1, 2, 3, 4)
):

    wrapped_env = gym.wrappers.RecordEpisodeStatistics(
        env, 50)  # Records episode-reward

    rewards_over_seeds = []
    eval_rewards_over_seeds = []

    for seed_i, seed in enumerate(train_seeds):
        random.seed(seed)
        np.random.seed(seed)

        rewards_over_episodes = []
        eval_rewards_over_episodes = []

        behavioral_agent.initialize()  # Unlearn

        if target_agent is not None:
            target_agent.initialize()

        for episode in tqdm(
                range(num_episodes),
                desc=f'Episodes for seed_{seed_i}'
        ):
            trajectory = []
            rewards_over_time = []

            # Noise state reset (not exploration level)
            behavioral_agent.reset()

            # gymnasium v26 requires users to set seed while resetting the
            # environment
            state, info = wrapped_env.reset(seed=seed)

            for t in range(T):
                action, p = behavioral_agent.act(state)
                next_state, reward, terminated, truncated, info = \
                    wrapped_env.step(action)

                done = terminated or truncated or (t + 1 == T)

                reward = reward_shaper(reward=reward, done=done, t=t)

                rewards_over_time.append(reward)

                trajectory.append(
                    Experience(
                        s=state,
                        a=action,
                        r=reward,
                        sp=next_state,
                        p=p,
                        done=int(done))
                )

                if done:
                    break
                else:
                    state = next_state

            # The environment terminates on positive reward
            # Opt 1: final reward of the episode
            # rewards_over_episodes.append(rewards_over_time[-1])

            # Opt 2: sum of rewards in episode (for cases where there's
            # always a reward)
            rewards_over_episodes.append(sum(rewards_over_time))

            # MC agents take the whole trajectory of the episode
            behavioral_agent.step(trajectory)

            # Target agent only learns
            if target_agent is not None:
                target_agent.step(deepcopy(trajectory))

            if (episode % evaluate_frequency == 0) and (eval_num_episodes > 0):
                eval_agent = behavioral_agent
                if target_agent is not None:
                    eval_agent = target_agent

                eval_rewards_over_episodes += evaluate_agent(
                    env,
                    agent=eval_agent,
                    T=T,
                    num_episodes=eval_num_episodes,
                    reward_shaper=reward_shaper
                )

        rewards_over_seeds.append(rewards_over_episodes)
        eval_rewards_over_seeds.append(eval_rewards_over_episodes)

    print(
        "\nAverage Total Rewards / Seed: Train = {0:.2f}, Eval = {1:.2f}".format(
            np.sum(np.array(rewards_over_seeds)) / len(train_seeds),
            np.sum(np.array(eval_rewards_over_seeds)) / len(train_seeds)
        ),
    )

    return rewards_over_seeds, eval_rewards_over_seeds



def off_policy_experiments(num_episodes, T):

    """
        Use U[num_actions] as behavior agent
        Vary the eps across Offpolicy agents
    """

    train_returns_over_seeds_over_over_agent = []
    eval_returns_over_seeds_over_over_agent = []
    legend = []

    def build_eps_sched(start, end=0.0):
        return LinearSchedule(start, end=end, steps=num_episodes)

    env = build_env()

    random_agent = DiscreteActionRandomAgent(
        action_space_dims=int(env.action_space.n),
        obs_space_dims=int(env.observation_space.n),
        distribution=randint,
        distribution_args=dict(low=0, high=int(env.action_space.n))
    )

    behavior_agent = random_agent

    for q_init in [-1, 0., 1]:
        for eps in [0.1, 0.2]:
            agent_off_policy = MCOffPolicy(
                action_space_dims=int(env.action_space.n),
                obs_space_dims=int(env.observation_space.n),
                discount=0.99,
                eps=build_eps_sched(eps, eps),
                qval_init=q_init
            )

            train_returns_over_seeds, eval_returns_over_seeds = run_env(
                env=env,
                behavioral_agent=behavior_agent,
                target_agent=agent_off_policy,
                T=T,
                num_episodes=num_episodes
            )

            train_returns_over_seeds_over_over_agent.append(
                train_returns_over_seeds)
            eval_returns_over_seeds_over_over_agent.append(
                eval_returns_over_seeds)
            legend.append(r'$\epsilon = {}$, q_init = {}'.format(eps, q_init))

    plot(
        eval_returns_over_seeds_over_over_agent,
        legend=legend,
        title=f'MCOffPolicy-Eval-Target'
    )


def on_policy_experiments_averaged_step_size(num_episodes, T, seeds):
    """
        Step size is averaged over (s,a) visits
    """

    train_returns_over_seeds_over_over_agent = []
    eval_returns_over_seeds_over_over_agent = []
    legend = []
    env = build_env()

    rand_agent = DiscreteActionRandomAgent(
        action_space_dims=int(env.action_space.n),
        obs_space_dims=int(env.observation_space.n),
        distribution=randint,
        distribution_args=dict(low=0, high=int(env.action_space.n))
    )

    train_returns_over_seeds, eval_returns_over_seeds = run_env(
        env,
        rand_agent,
        T=T,
        num_episodes=num_episodes,
        train_seeds=seeds)

    train_returns_over_seeds_over_over_agent.append(train_returns_over_seeds)
    eval_returns_over_seeds_over_over_agent.append(eval_returns_over_seeds)
    legend.append('Uniform Random')

    def build_eps_sched(start):
        return LinearSchedule(start, end=0.0, steps=num_episodes)

    step_size = None

    for q_init in [-1, 0, 1]:
        for eps in [0.05, 0.1]:
            agent = MCOnPolicyFirstVisitGLIE(
                action_space_dims=int(env.action_space.n),
                obs_space_dims=int(env.observation_space.n),
                discount=0.99,
                eps=build_eps_sched(eps),
                qval_init=q_init,
                step_size=step_size)

            train_returns_over_seeds, eval_returns_over_seeds = run_env(
                env,
                agent,
                T=T,
                num_episodes=num_episodes,
                train_seeds=seeds)

            train_returns_over_seeds_over_over_agent.append(
                train_returns_over_seeds)

            eval_returns_over_seeds_over_over_agent.append(
                eval_returns_over_seeds)
            legend.append(r'$\epsilon = {}$, q_init = {}'.format(eps, q_init))

    plot(
        train_returns_over_seeds_over_over_agent,
        legend=legend,
        title=f'MCOnPolicyFirstVisit - Avg. Step Size\nTrain'
    )

    plot(
        eval_returns_over_seeds_over_over_agent,
        legend=legend,
        title=f'MCOnPolicyFirstVisit - Avg. Step Size\nEval'
    )


def on_policy_experiments_fixed_step_size(num_episodes, T, seeds):
    train_returns_over_seeds_over_over_agent = []
    eval_returns_over_seeds_over_over_agent = []
    legend = []
    env = build_env()

    rand_agent = DiscreteActionRandomAgent(
        action_space_dims=int(env.action_space.n),
        obs_space_dims=int(env.observation_space.n),
        distribution=randint,
        distribution_args=dict(low=0, high=int(env.action_space.n))
    )

    train_returns_over_seeds, eval_returns_over_seeds = run_env(
        env,
        rand_agent,
        T=T,
        num_episodes=num_episodes,
        train_seeds=seeds)

    train_returns_over_seeds_over_over_agent.append(train_returns_over_seeds)
    eval_returns_over_seeds_over_over_agent.append(eval_returns_over_seeds)
    legend.append('Uniform Random')

    def build_eps_sched(start):
        return LinearSchedule(start, end=0.0, steps=num_episodes)

    q_init = 0

    for step_size in [0.05, 0.1, 0.2, 0.3]:
        for eps in [0.1]:
            agent = MCOnPolicyFirstVisitGLIE(
                action_space_dims=int(env.action_space.n),
                obs_space_dims=int(env.observation_space.n),
                discount=0.99,
                eps=build_eps_sched(eps),
                qval_init=q_init,
                step_size=step_size)

            train_returns_over_seeds, eval_returns_over_seeds = run_env(
                env,
                agent,
                T=T,
                num_episodes=num_episodes,
                train_seeds=seeds)

            train_returns_over_seeds_over_over_agent.append(
                train_returns_over_seeds)

            eval_returns_over_seeds_over_over_agent.append(
                eval_returns_over_seeds)
            legend.append(r'$\epsilon = {}$, step_size = {}'.format(eps, step_size))

    plot(
        train_returns_over_seeds_over_over_agent,
        legend=legend,
        title=f'MCOnPolicyFirstVisit - Fixed Step Size\nTrain'
    )

    plot(
        eval_returns_over_seeds_over_over_agent,
        legend=legend,
        title=f'MCOnPolicyFirstVisit - Fixed Step Size\nEval'
    )


if __name__ == '__main__':
    num_episodes = 3000
    T = 30
    seeds = tuple(range(5))

    off_policy_experiments(
        num_episodes=num_episodes,
        T=T
    )

    num_episodes = 300
    on_policy_experiments_averaged_step_size(
        num_episodes=num_episodes,
        T=T,
        seeds=seeds)

    on_policy_experiments_fixed_step_size(
        num_episodes=num_episodes,
        T=T,
        seeds=seeds)

    exit(0)
