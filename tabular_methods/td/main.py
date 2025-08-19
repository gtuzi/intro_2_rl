from typing import Callable

import numpy as np
import gymnasium as gym
from gymnasium import Env
import seaborn as sns
from scipy.stats import randint

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
    do_on_policy = True
    do_off_policy = False

    seeds =  list(range(5)) # list(range(10))
    state_bins = None
    num_parallel_workers = min(10, len(seeds))
    eval_num_episodes = 20 # 50

    # ENV_NAME = 'FrozenLake'
    # ENV_NAME = 'CliffWalking'
    ENV_NAME = 'Taxi'

    if ENV_NAME == 'FrozenLake':
        if do_on_policy:
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
        elif do_off_policy:
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
            update_coefficient_builder = build_linear_sched  # alpha / lr

    elif ENV_NAME == 'CliffWalking':
        if do_on_policy:
            num_episodes = 5000
            T = 20

            q_init = 0.

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
        elif do_off_policy:
            num_episodes = 5000
            T = 20

            q_init = 0.

            epsilon_start = 1.0
            epsilon_end = 0.01
            eps_steps = num_episodes * T
            eps_schedule_builder = build_linear_sched

            alpha_start = 0.3
            alpha_end = 0.01
            alpha_steps = num_episodes * T
            update_coefficient_builder = build_linear_sched  # alpha / lr

            # the smaller, the more frequent
            evaluate_frequency = max(1, int(0.05 * num_episodes * T))

    elif ENV_NAME == 'Taxi':
        if do_on_policy:
            num_episodes = 5000
            T = 50
            q_init = 0.

            epsilon_start = 0.3
            epsilon_end = 0.001
            eps_steps = num_episodes * T
            eps_schedule_builder = build_linear_sched

            alpha_start = 1.0
            alpha_end = 0.2
            alpha_steps = num_episodes * T
            update_coefficient_builder = build_linear_sched  # alpha / lr

            evaluate_frequency = max(1, int(0.02 * num_episodes * T))

        elif do_off_policy:
            num_episodes = 5000
            T = 50
            q_init = 0.

            epsilon_start = 1.0
            epsilon_end = 0.001
            eps_steps = num_episodes * T
            eps_schedule_builder = build_linear_sched

            alpha_start = 0.1
            alpha_end = 0.001
            alpha_steps = num_episodes * T
            update_coefficient_builder = build_cosine_sched  # alpha / lr

            evaluate_frequency = max(1, int(0.01 * num_episodes * T))

    else:
        raise NotImplementedError

    env = env_builder()

    gamma = 0.99
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

    # region On-Policy
    if do_on_policy:
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
                parallel_eval=True,
                soft_eval=True,
                hard_eval=True,
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
            save_dir='images/training_metrics/on_policy',
        )

        plot_evaluation_metrics(
            results=processed_numerical_results,
            color_map=color_map,
            file_root = f'{ENV_NAME}',
            save_dir = 'images/evaluation_metrics/on_policy',
        )

        plot_value_accuracy(
            results=processed_numerical_results,
            metrics_to_compare=['V0', 'soft_G0'],
            color_map=color_map,
            file_root=f'{ENV_NAME}',
            save_dir='images/learning/on_policy',
        )

        plot_action_distribution(
            results=processed_distribution_results,
            file_root = f'{ENV_NAME}',
            save_dir = 'images/behaviors/on_policy',
        )

    # endregion
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

    # region Off-Policy

    processed_numerical_results = {}
    processed_distribution_results = {}

    # behavioral_class = ExpectedSarsa
    behavioral_class = DiscreteActionRandomAgent

    if do_off_policy:
        for agent_name, target_class in [
            ('OffPolicyNStepSarsa-1', nStepsSarsaOffPolicy),
            # ('OffPolicyNStepSarsa-2', nStepsSarsaOffPolicy),
            ('OffPolicyNStepSarsa-3', nStepsSarsaOffPolicy),
            # ('OffPolicyNStepSarsa-5', nStepsSarsaOffPolicy)
        ]:

            behavioral_akwargs = dict(
                obs_space_dims = agent_kwargs['obs_space_dims'],  # Not used
                action_space_dims = agent_kwargs['action_space_dims'],
                distribution=randint,
                distribution_args=dict(
                    low=0,
                    high=agent_kwargs['action_space_dims']
                )
            )

            target_akwargs = agent_kwargs.copy()

            if agent_name == 'OffPolicyNStepSarsa-1':
                target_akwargs.update(dict(n=1))
            elif agent_name == 'OffPolicyNStepSarsa-2':
                target_akwargs.update(dict(n=2))
            elif agent_name == 'OffPolicyNStepSarsa-3':
                target_akwargs.update(dict(n=3))
            elif agent_name == 'OffPolicyNStepSarsa-4':
                target_akwargs.update(dict(n=4))
            elif agent_name == 'OffPolicyNStepSarsa-5':
                target_akwargs.update(dict(n=5))

            results = parallel_train(
                env_builder=env_builder,
                behavioral_agent_class=behavioral_class,
                behavioral_agent_kwargs=behavioral_akwargs,
                target_agent_class=target_class,
                target_agent_kwargs=target_akwargs,
                T=T,
                num_episodes=num_episodes,
                reward_shaper=reward_shaper,
                train_seeds=seeds,
                do_eval=True,
                eval_num_episodes=eval_num_episodes,
                evaluate_frequency=evaluate_frequency,
                parallel_eval=True,
                soft_eval=True,
                hard_eval=True,
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
        palette = buil_high_contrast_palette(
            n_colors=len(all_algorithm_names))

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
            save_dir='images/training_metrics/off_policy',
        )

        plot_evaluation_metrics(
            results=processed_numerical_results,
            color_map=color_map,
            file_root=f'{ENV_NAME}',
            save_dir='images/evaluation_metrics/off_policy',
        )

        plot_value_accuracy(
            results=processed_numerical_results,
            metrics_to_compare = ['V0', 'soft_G0'],
            color_map=color_map,
            file_root=f'{ENV_NAME}',
            save_dir='images/learning/off_policy',
        )

        plot_action_distribution(
            results=processed_distribution_results,
            eval_types_to_plot=['soft', 'hard'],
            file_root=f'{ENV_NAME}',
            save_dir='images/behaviors/off_policy',
        )

    # endregion

    print("Done !")
    exit(0)