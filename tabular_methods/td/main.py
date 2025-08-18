from typing import Callable

import numpy as np
import gymnasium as gym
from gymnasium import Env
import seaborn as sns

from shared.utils import CosineDecaySchedule
from tabular_methods.utils import (
    LinearSchedule,
    Experience,
    DiscreteActionAgent,
    DiscreteActionRandomAgent,
    QEpsGreedyAgent, SoftPolicy
)

from tabular_methods.train import (
    parallel_train, sequential_train, preprocess_for_numerical_plots,
    preprocess_for_distribution_plots
)

from tabular_methods.plot import (
    plot_training_metrics,
    plot_evaluation_metrics,
    plot_value_accuracy,
    plot_action_distribution,
    buil_high_contrast_palette
)

from agents import (
    Sarsa,
    QLearning,
    ExpectedSarsa,
    nStepSarsa,
    nStepsSarsaOffPolicy,
    QSigmaOffPolicy
)

global ENV_NAME


def build_env(name: str, **kwargs) -> Env:
    render = kwargs.pop('render', False)
    if name == 'FrozenLake':
        env = gym.make(
            'FrozenLake-v1',
            render_mode="human" if render else None,
            desc=None,
            map_name="4x4",
            is_slippery=True,
            **kwargs
        )
    elif name == 'Taxi':
        env = gym.make(
            'Taxi-v3',
            render_mode="human" if render else None,
            **kwargs
        )
    elif name == 'CliffWalking':
        env = gym.make(
            "CliffWalking-v0",
            render_mode = "human" if render else None,
            **kwargs
        )
    else:
        raise NotImplementedError

    return env


def reward_shaper(reward: float, done: bool, t: int):
    return reward


def build_linear_sched(start, steps, end = 0.0):
    return LinearSchedule(start, end=end, steps=steps)


def build_cosine_sched(start, steps, end = 0.0):
    return CosineDecaySchedule(start, final_value=end, decay_steps=steps)


def env_builder() -> Env:
    return build_env(ENV_NAME, render=False)


if __name__ == '__main__':
    do_random = False
    seeds = list(range(10))
    gamma = 0.99
    state_bins = None
    num_parallel_workers = min(10, len(seeds))
    eval_num_episodes = 100

    # ENV_NAME = 'FrozenLake'
    # ENV_NAME = 'CliffWalking'
    ENV_NAME = 'Taxi'

    if ENV_NAME == 'FrozenLake':
        num_episodes = 10000
        T = 100
        evaluate_frequency = max(1, int(0.001 * num_episodes * T))

        q_init = -1.0

        epsilon_start = 1.0
        epsilon_end = 0.001
        eps_steps = num_episodes * T // 10
        eps_schedule_builder = build_linear_sched

        alpha_start = 0.3
        alpha_end = 0.1
        alpha_steps = num_episodes * T
        update_coefficient_builder = build_linear_sched # alpha / lr
    elif ENV_NAME == 'CliffWalking':
        num_episodes = 1000
        T = 100

        q_init = -200  # to show how it learns

        epsilon_start = 1.0
        epsilon_end = 0.01
        eps_steps = num_episodes * T
        eps_schedule_builder = build_linear_sched

        alpha_start = 0.5
        alpha_end = 0.01
        alpha_steps = num_episodes * T
        update_coefficient_builder = build_linear_sched  # alpha / lr

        # the smaller, the more frequent
        evaluate_frequency = max(1, int(0.05 * num_episodes * T))
    elif ENV_NAME == 'Taxi':
        num_episodes = 3000
        T = 50

        q_init = -150.

        epsilon_start = 0.3
        epsilon_end = 0.001
        eps_steps = num_episodes * T
        eps_schedule_builder = build_linear_sched

        alpha_start = 1.0
        alpha_end = 0.2
        alpha_steps = num_episodes * T
        update_coefficient_builder = build_linear_sched  # alpha / lr

        evaluate_frequency = max(1, int(0.02 * num_episodes * T))
    else:
        raise NotImplementedError

    env = env_builder()

    agent_class = nStepSarsa

    agent_kwargs = dict(
        action_space_dims=int(env.action_space.n),
        obs_space_dims=int(env.observation_space.n),
        discount=gamma,
        qval_init=q_init,
        eps_schedule_builder=eps_schedule_builder,
        eps_schedule_kwargs={
            "start": epsilon_start,
            "steps": eps_steps,
            "end": epsilon_end
        },
        update_coefficient_builder=update_coefficient_builder,
        update_coefficient_kwargs={
            "start": alpha_start,
            "steps": alpha_steps,
            "end": alpha_end
        }
    )

    processed_numerical_results = { }
    processed_distribution_results = { }

    for agent_name, agent_class in [
        ('Sarsa', Sarsa),
        ('ExpectedSarsa', ExpectedSarsa),
        ('QLearning', QLearning),
        ('nStepSarsa', nStepSarsa)
    ]:

        akwargs = agent_kwargs.copy()

        if agent_class == nStepSarsa:
            n = 4
            akwargs.update(dict(n = n))

        results = parallel_train(
            env_builder=env_builder,
            behavioral_agent_class=agent_class,
            behavioral_agent_kwargs=akwargs,
            T=T,
            num_episodes=num_episodes,
            reward_shaper=reward_shaper,
            train_seeds=seeds,
            do_eval=True,
            eval_num_episodes=eval_num_episodes,
            evaluate_frequency=evaluate_frequency,
            greedy_eval=True,
            parallel_eval=True,
            num_parallel_workers=num_parallel_workers,
        )


        processed_numerical_results.update({
            agent_name: preprocess_for_numerical_plots(
                results, agent_name)
        })

        processed_distribution_results.update({
            agent_name: preprocess_for_distribution_plots(
                results, agent_name)
        })

    # --- Plotting --- #
    all_algorithm_names = list(processed_distribution_results.keys())
    palette = buil_high_contrast_palette(n_colors=len(all_algorithm_names))

    color_map = {
        name: color for name, color in zip(
            list(processed_distribution_results.keys()),
            palette
        )
    }

    plot_training_metrics(
        results=processed_numerical_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/training_metrics',
    )

    plot_evaluation_metrics(
        results=processed_numerical_results,
        color_map=color_map,
        file_root = f'{ENV_NAME}',
        save_dir = 'images/evaluation_metrics',
    )

    plot_value_accuracy(
        results=processed_numerical_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/learning',
    )

    plot_action_distribution(
        results=processed_distribution_results,
        file_root = f'{ENV_NAME}',
        save_dir = 'images/behaviors',
    )

    # --- 5. Generate an EXAMPLE State Visitation heatmap ---
    # NOTE: This plot will only work if you ran the experiment with state_bins defined.
    # You must change x_dim_idx and y_dim_idx to match the state dimensions
    # you want to visualize for your specific environment.
    # This example assumes a 4D state space (like CartPole) and plots dimension 0 vs. 2.
    #
    # print("\n--- Generating State Visitation Heatmap (Example: Dim 0 vs 2) ---")
    # plot_state_visitation(
    #     results=processed_results,
    #     x_dim_idx=0,
    #     y_dim_idx=2
    # )

    print("Done !")
    exit(0)