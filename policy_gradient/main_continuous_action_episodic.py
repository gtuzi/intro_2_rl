import itertools
import warnings
from typing import Callable

import numpy as np

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*env\.shape to get variables from other wrappers is deprecated.*"
)
import gymnasium as gym
from gymnasium import Env

from shared.utils import LinearSchedule

from policy_gradient.train import (
    parallel_train,
    sequential_train,
    preprocess_for_training_plots,
    preprocess_for_evaluation_plots,
    print_health_report
)


from policy_gradient.agents import (
    ReinforceContinuousAction,
    ReinforceBaselineContinuousAction,
    ACWithEligibilityTracesContinuousAction
)


from policy_gradient.plot import (
    plot_training_metrics,
    plot_evaluation_metrics,
    plot_value_accuracy,
    buil_high_contrast_palette
)

global ENV_NAME


def build_env(name: str, **kwargs) -> Env:
    render = kwargs.pop('render', False)

    if 'MountainCar' in name:
        env = gym.make(
            'MountainCarContinuous-v0',
            render_mode="human" if render else None
        )
    elif 'Pendulum' in name:
        env = gym.make(
            "Pendulum-v1",
            render_mode="human" if render else None
        )
    elif 'LunarLander' in name:
        enable_wind = kwargs.pop('enable_wind', False)
        env = gym.make(
            "LunarLander-v3",
            continuous=True,
            enable_wind=enable_wind,
            render_mode="human" if render else None
        )
    else:
        raise NotImplementedError

    if 'max_episode_steps' in kwargs:
        env._max_episode_steps = kwargs.pop('max_episode_steps')

    return env


def env_builder(**kwargs) -> Env:
    global ENV_NAME
    return build_env(ENV_NAME, render=False, **kwargs)


def base_reward_shaper(reward: float, **kwargs):
    return reward


def build_linear_sched(start, steps, end = 0.0):
    return LinearSchedule(start, end=end, steps=steps)



def reinforce(
        num_train_seeds: int ,
        eval_num_episodes: int,
        reward_shaper: Callable,
        num_episodes: int,
        T: int,
        evaluate_frequency: int
):
    seeds = list(range(num_train_seeds))
    num_parallel_workers = min(10, len(seeds))

    # parameters for env_build()
    environment_kwargs = dict(max_episode_steps = T)
    env = env_builder()

    gamma = 0.99
    alpha_steps = num_episodes
    update_coefficient_builder = build_linear_sched

    state_size = env.unwrapped.observation_space.shape[0]
    action_size = env.unwrapped.action_space.shape[0]
    min_action = env.action_space.low.tolist()
    max_action = env.action_space.high.tolist()

    H = 4096
    h = (H // (state_size + action_size), )

    # Pattern: <thing>_builder, <thing>_kwargs
    agent_kwargs = dict(
        action_size=action_size,
        state_size=state_size,
        action_mins=min_action,
        action_maxs=max_action,
        discount=gamma,
        hidden_dims=h,
        normalize_input=False,
        grad_norm_threshold=1,
        update_coefficient_builder=update_coefficient_builder,
    )

    processed_training_results = {}
    processed_evaluation_results = {}

    agent_class = ReinforceContinuousAction

    for (
            alpha,
            normalize_reward,
            normalize_grad
    ) in itertools.product(
            [1e-4, 5e-4],
            [True],
            [True]
    ):
        akwargs = agent_kwargs.copy()

        akwargs.update(
            dict(
                update_coefficient_kwargs =
                {
                    "start": alpha,
                    "steps": alpha_steps,
                    "end"  : 0.2 * alpha
                },
                normalize_reward=normalize_reward,
                normalize_grad=normalize_grad,
            )
        )

        agent_name = 'Reinforce: ' + (
                f'$\\alpha={alpha:.1e}$, '
                f'norm_r: {int(normalize_reward)}, '
                f'norm_g: {int(normalize_grad)}'
        )

        results = parallel_train(
            env_builder=lambda: env_builder(**environment_kwargs),
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

        processed_training_results.update({
            agent_name: preprocess_for_training_plots(
                results, agent_name)
        })

        processed_evaluation_results.update({
            agent_name: preprocess_for_evaluation_plots(
                results, agent_name)
        })


    # ------- Health Report --------- #
    print_health_report(
        training_results=processed_training_results,
        evaluation_results=processed_evaluation_results
    )

    # region Present

    # --- Plotting --- #
    all_algorithm_names = list(processed_training_results.keys())
    palette = buil_high_contrast_palette(n_colors=len(all_algorithm_names))

    color_map = {
        name: color for name, color in zip(
            list(processed_training_results.keys()),
            palette
        )
    }

    plot_training_metrics(
        results=processed_training_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/training_metrics/reinforce_continuous_action',
    )

    plot_evaluation_metrics(
        results=processed_evaluation_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/evaluation_metrics/reinforce_continuous_action',
    )

    plot_value_accuracy(
        results=processed_evaluation_results,
        metrics_to_compare=['V0', 'soft_G0'],
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/learning/reinforce_continuous_action',
    )

    #endregion


def reinforce_baseline(
        num_train_seeds: int ,
        eval_num_episodes: int,
        reward_shaper: Callable,
        num_episodes: int,
        T: int,
        evaluate_frequency: int
):
    seeds = list(range(num_train_seeds))
    num_parallel_workers = min(10, len(seeds))

    # parameters for env_build()
    environment_kwargs = dict(max_episode_steps = T)
    env = env_builder()

    gamma = 0.99
    alpha_steps = num_episodes
    update_coefficient_builder = build_linear_sched

    state_size = env.unwrapped.observation_space.shape[0]
    action_size = env.unwrapped.action_space.shape[0]
    min_action = env.action_space.low.tolist()
    max_action = env.action_space.high.tolist()

    H = 4096
    h = (H // (state_size + action_size), )

    # Pattern: <thing>_builder, <thing>_kwargs
    agent_kwargs = dict(
        action_size=action_size,
        state_size=state_size,
        action_mins=min_action,
        action_maxs=max_action,
        discount=gamma,
        hidden_dims=h,
        normalize_input=False,
        grad_norm_threshold=1,
        update_coefficient_actor_builder=update_coefficient_builder,
        update_coefficient_critic_builder=update_coefficient_builder,
    )

    processed_training_results = {}
    processed_evaluation_results = {}

    agent_class = ReinforceBaselineContinuousAction

    for (
            alpha,
            normalize_reward,
            normalize_grad
    ) in itertools.product(
            [1e-4, 5e-4],
            [True],
            [True]
    ):
        akwargs = agent_kwargs.copy()

        akwargs.update(
            dict(
                update_coefficient_actor_kwargs =
                {
                    "start": alpha,
                    "steps": alpha_steps,
                    "end"  : 0.1 * alpha
                },

                update_coefficient_critic_kwargs=
                {
                    "start": 25 * alpha,
                    "steps": alpha_steps,
                    "end"  : 0.1 * 25 * alpha
                },

                normalize_reward=normalize_reward,
                normalize_grad=normalize_grad,
            )
        )

        agent_name = 'ReinforceBaseline: ' + (
                f'$\\alpha={alpha:.1e}$, '
                f'norm_r: {int(normalize_reward)}, '
                f'norm_g: {int(normalize_grad)}'
        )

        results = parallel_train(
            env_builder=lambda: env_builder(**environment_kwargs),
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

        processed_training_results.update({
            agent_name: preprocess_for_training_plots(
                results, agent_name)
        })

        processed_evaluation_results.update({
            agent_name: preprocess_for_evaluation_plots(
                results, agent_name)
        })

    # ------- Health Report --------- #
    print_health_report(
        training_results=processed_training_results,
        evaluation_results=processed_evaluation_results
    )

    # region Present

    # --- Plotting --- #
    all_algorithm_names = list(processed_training_results.keys())
    palette = buil_high_contrast_palette(n_colors=len(all_algorithm_names))

    color_map = {
        name: color for name, color in zip(
            list(processed_training_results.keys()),
            palette
        )
    }

    plot_training_metrics(
        results=processed_training_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/training_metrics/reinforcebaseline_continuous_action',
    )

    plot_evaluation_metrics(
        results=processed_evaluation_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/evaluation_metrics/reinforcebaseline_continuous_action',
    )

    plot_value_accuracy(
        results=processed_evaluation_results,
        metrics_to_compare=['V0', 'soft_G0'],
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/learning/reinforcebaseline_continuous_action',
    )

    #endregion


def ac_eligibility_traces(
        num_train_seeds: int ,
        eval_num_episodes: int,
        reward_shaper: Callable,
        num_episodes: int,
        T: int,
        evaluate_frequency: int
):
    seeds = list(range(num_train_seeds))
    num_parallel_workers = min(10, len(seeds))

    # parameters for env_build()
    environment_kwargs = dict(max_episode_steps = T)
    env = env_builder()

    gamma = 0.99
    alpha_steps = num_episodes * T
    update_coefficient_builder = build_linear_sched

    state_size = env.unwrapped.observation_space.shape[0]
    action_size = env.unwrapped.action_space.shape[0]
    min_action = env.action_space.low.tolist()
    max_action = env.action_space.high.tolist()

    H = 4096
    h = (H // (state_size + action_size), )

    # Pattern: <thing>_builder, <thing>_kwargs
    agent_kwargs = dict(
        action_size=action_size,
        state_size=state_size,
        action_mins=min_action,
        action_maxs=max_action,
        discount=gamma,
        hidden_dims=h,
        normalize_input=False,
        grad_norm_threshold=1,
        update_coefficient_actor_builder=update_coefficient_builder,
        update_coefficient_critic_builder=update_coefficient_builder,
    )

    processed_training_results = {}
    processed_evaluation_results = {}

    agent_class = ACWithEligibilityTracesContinuousAction

    for (alpha, lam, normalize_grad) in itertools.product(
            [1e-3, 5e-3],
            [0.2, 0.5, 0.8],
            [True]
    ):
        akwargs = agent_kwargs.copy()

        akwargs.update(
            dict(
                lam_actor = lam,

                lam_critic = lam,

                update_coefficient_actor_kwargs =
                {
                    "start": alpha,
                    "steps": alpha_steps,
                    "end"  : 0.1 * alpha
                },

                update_coefficient_critic_kwargs=
                {
                    "start": 25 * alpha,
                    "steps": alpha_steps,
                    "end"  : 0.1 * 25 * alpha
                },

                normalize_grad=normalize_grad,
            )
        )

        agent_name = 'AC-EligTraces: ' + (
                f'$\\alpha={alpha:.1e}$, '
                f'$\\lambda={lam:.1e}$, '
                f'clip_g: {int(normalize_grad)}'
        )

        results = parallel_train(
            env_builder=lambda: env_builder(**environment_kwargs),
            behavioral_agent_class=agent_class,
            behavioral_agent_kwargs=akwargs,
            T=T,
            num_episodes=num_episodes,
            reward_shaper=reward_shaper,
            train_seeds=seeds,
            do_eval=True,
            parallel_eval=True,
            eval_num_episodes=eval_num_episodes,
            evaluate_frequency=evaluate_frequency,
            soft_eval=True,
            hard_eval=True,
            num_parallel_workers=num_parallel_workers,
        )

        processed_training_results.update({
            agent_name: preprocess_for_training_plots(
                results, agent_name)
        })

        processed_evaluation_results.update({
            agent_name: preprocess_for_evaluation_plots(
                results, agent_name)
        })


    # ------- Health Report --------- #
    print_health_report(
        training_results=processed_training_results,
        evaluation_results=processed_evaluation_results
    )

    # region Present

    # --- Plotting --- #
    all_algorithm_names = list(processed_training_results.keys())
    palette = buil_high_contrast_palette(n_colors=len(all_algorithm_names))

    color_map = {
        name: color for name, color in zip(
            list(processed_training_results.keys()),
            palette
        )
    }

    plot_training_metrics(
        results=processed_training_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/training_metrics/ACwEligTrace_continuous_action',
    )

    plot_evaluation_metrics(
        results=processed_evaluation_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/evaluation_metrics/ACwEligTrace_continuous_action',
    )

    plot_value_accuracy(
        results=processed_evaluation_results,
        metrics_to_compare=['V0', 'soft_G0'],
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/learning/ACwEligTrace_continuous_action',
    )

    #endregion


if __name__ == '__main__':
    global ENV_NAME

    def get_env_params():
        global ENV_NAME

        reduce_episodes = 1.0
        reward_shaper = base_reward_shaper

        if ENV_NAME == 'MountainCar':
            num_episodes = int(200 * reduce_episodes)
            T = 999
            evaluate_frequency = max(1, int(0.1 * num_episodes * T))
        elif ENV_NAME == 'Pendulum':
            num_episodes = int(1000 * reduce_episodes)
            T = 100
            evaluate_frequency = max(1, int(0.1 * num_episodes * T))
        else:
            raise NotImplementedError

        return num_episodes, T, evaluate_frequency, reward_shaper

    num_train_seeds=10
    eval_num_episodes=5

    envs = [
        'MountainCar',
        'Pendulum',
    ]

    experiment_fns = [
        reinforce,
        reinforce_baseline,
        ac_eligibility_traces
    ]

    for env in envs:

        ENV_NAME = env

        (
            num_episodes,
            T,
            evaluate_frequency,
            reward_shaper
        ) = get_env_params()

        for fn in experiment_fns:
            fn(
                num_train_seeds=num_train_seeds,
                eval_num_episodes=eval_num_episodes,
                reward_shaper=reward_shaper,
                num_episodes=num_episodes,
                T=T,
                evaluate_frequency=evaluate_frequency
            )

    exit(0)

