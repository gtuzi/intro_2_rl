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
    preprocess_for_distribution_plots,
    preprocess_for_training_plots,
    preprocess_for_evaluation_plots,
    print_health_report
)


from policy_gradient.agents import (
    Reinforce_LinearApproximation,
    Reinforce,
    ReinforceBaseline,
    OneStepAC,
    ACWithEligibilityTraces,
    ACWithEligibilityTracesContinuing
)


from policy_gradient.plot import (
    plot_training_metrics,
    plot_evaluation_metrics,
    plot_value_accuracy,
    plot_action_distribution,
    buil_high_contrast_palette
)

global ENV_NAME


def build_env(name: str, **kwargs) -> Env:
    render = kwargs.pop('render', False)

    if 'MountainCar' in name:
        env = gym.make(
            'MountainCar-v0',
            render_mode="human" if render else None
        )
    elif 'AirRaid' in name:
        env = gym.make(
            "ALE/AirRaid-v5",
            obs_type="rgb",
            render_mode="human" if render else None
        )
    elif 'LunarLander' in name:
        enable_wind = kwargs.pop('enable_wind', False)
        env = gym.make(
            "LunarLander-v3",
            continuous=False,
            enable_wind=enable_wind,
            render_mode="human" if render else None
        )
    elif 'CartPole' in name:
        env = gym.make(
            "CartPole-v1",
            render_mode="human" if render else None
        )
    elif 'Pendulum' in name:
        env = gym.make(
            "Pendulum-v1",
            render_mode="human" if render else None
        )
    elif 'Acrobot' in name:
        env = gym.make(
            'Acrobot-v1',
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


def acrobot_reward_shaper(reward: float, state: np.ndarray, **kwargs) -> float:

    """
    A reward shaper for Acrobot that provides a dense reward based on height
    and penalizes excessive velocity to encourage smoother control.

    The state is: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), vel1, vel2]
    The height of the foot is: -cos(theta1) - cos(theta1 + theta2)
    """

    # If the original reward is 0 (or > -1), the goal has been reached. Return a large bonus.
    if reward > -1.0:
        return 10.0

    # The state vector components
    cos_theta1 = state[0]
    sin_theta1 = state[1]
    cos_theta2 = state[2]
    sin_theta2 = state[3]
    vel1 = state[4]
    vel2 = state[5]

    # Calculate the height of the foot using the angle sum identity for cosine
    height_of_foot = -cos_theta1 - (
                cos_theta1 * cos_theta2 - sin_theta1 * sin_theta2)

    # Penalty for high angular velocity  to encourage the agent to be
    # more controlled and stable.
    velocity_penalty_weight = 0.001
    velocity_penalty = -velocity_penalty_weight * (vel1 ** 2 + vel2 ** 2)

    # The final reward is the height reward plus the stability penalty
    return float(height_of_foot + velocity_penalty)


def build_linear_sched(start, steps, end = 0.0):
    return LinearSchedule(start, end=end, steps=steps)


#############################################################
# Experiments
#############################################################

# region REINFORCE

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
    temp_start = 1.0
    temp_end = 1.0
    temp_steps = num_episodes
    temp_builder = build_linear_sched
    alpha_steps = num_episodes
    update_coefficient_builder = build_linear_sched

    state_size = env.unwrapped.observation_space.shape[0]
    action_size = int(env.unwrapped.action_space.n)
    H = 4096
    h = (H // (state_size + action_size), )

    # Pattern: <thing>_builder, <thing>_kwargs
    agent_kwargs = dict(
        action_space_dims=action_size,
        state_size=state_size,
        discount=gamma,
        temp_builder=temp_builder,
        temp_kwargs={
            "start": temp_start,
            "steps": temp_steps,
            "end"  : temp_end
        },
        hidden_dims=h,
        normalize_grad=True,
        normalize_input=False,
        grad_norm_threshold=1.,
        update_coefficient_builder=update_coefficient_builder,
    )

    processed_training_results = {}
    processed_evaluation_results = {}
    processed_distribution_results = {}

    agent_class = Reinforce

    for alpha in [1e-4, 1e-3, 5e-3]:
        for normalize_reward in [False, True]:
            akwargs = agent_kwargs.copy()

            akwargs.update(
                dict(
                    update_coefficient_kwargs =
                    {
                        "start": alpha,
                        "steps": alpha_steps,
                        "end"  : 0.2 * alpha
                    },
                    normalize_reward=normalize_reward
                )
            )

            agent_name = 'Reinforce: ' + (
                    f'$\\alpha={alpha:.1e}$, '
                    f'norm_r: {int(normalize_reward)}'
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

            processed_distribution_results.update({
                agent_name: preprocess_for_distribution_plots(
                    results, agent_name)
            })


    # ------- Health Report --------- #
    print_health_report(
        training_results=processed_training_results,
        evaluation_results=processed_evaluation_results,
        distribution_results=processed_distribution_results
    )

    # region Present

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
        results=processed_training_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/training_metrics/reinforce',
    )

    plot_evaluation_metrics(
        results=processed_evaluation_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/evaluation_metrics/reinforce',
    )

    plot_value_accuracy(
        results=processed_evaluation_results,
        metrics_to_compare=['V0', 'soft_G0'],
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/learning/reinforce',
    )

    plot_action_distribution(
        results=processed_distribution_results,
        file_root=f'{ENV_NAME}',
        save_dir='images/behaviors/reinforce',
    )

    #endregion


def reinforce_with_baseline(
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
    temp_start = 1.0
    temp_end = 1.0
    temp_steps = num_episodes
    temp_builder = build_linear_sched
    alpha_steps = num_episodes
    update_coefficient_builder = build_linear_sched

    state_size = env.unwrapped.observation_space.shape[0]
    action_size = int(env.unwrapped.action_space.n)
    H = 4096
    h = (H // (state_size + action_size), )

    # Pattern: <thing>_builder, <thing>_kwargs
    agent_kwargs = dict(
        action_space_dims=action_size,
        state_size=state_size,
        discount=gamma,
        temp_builder=temp_builder,
        temp_kwargs={
            "start": temp_start,
            "steps": temp_steps,
            "end"  : temp_end
        },
        hidden_dims=h,
        normalize_grad=True,
        normalize_input=False,
        grad_norm_threshold=1.,
        update_coefficient_actor_builder=update_coefficient_builder,
        update_coefficient_critic_builder=update_coefficient_builder,
    )

    processed_training_results = {}
    processed_evaluation_results = {}
    processed_distribution_results = {}
    agent_class = ReinforceBaseline

    for alpha in [1e-4, 1e-3, 5e-3]:
        for normalize_reward in [True, False]:
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
                        "start": alpha,
                        "steps": alpha_steps,
                        "end"  : 0.1 * alpha
                    },
                    normalize_reward=normalize_reward
                )
            )

            agent_name = 'ReinforceBaseline: ' + (
                f'$\\alpha={alpha:.1e}$, '
                f'norm_r={int(normalize_reward)}'
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

            processed_distribution_results.update({
                agent_name: preprocess_for_distribution_plots(
                    results, agent_name)
            })

    # ------- Health Report --------- #
    print_health_report(
        training_results=processed_training_results,
        evaluation_results=processed_evaluation_results,
        distribution_results=processed_distribution_results
    )

    # region Present

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
        results=processed_training_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/training_metrics/reinforcebaseline',
    )

    plot_evaluation_metrics(
        results=processed_evaluation_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/evaluation_metrics/reinforcebaseline',
    )

    plot_value_accuracy(
        results=processed_evaluation_results,
        metrics_to_compare=['V0', 'soft_G0'],
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/learning/reinforcebaseline',
    )

    plot_action_distribution(
        results=processed_distribution_results,
        file_root=f'{ENV_NAME}',
        save_dir='images/behaviors/reinforcebaseline',
    )

    #endregion

#endregion


# region Actor-Critic
def one_step_ac(
        num_train_seeds: int,
        eval_num_episodes: int,
        reward_shaper: Callable,
        num_episodes: int,
        T: int,
        evaluate_frequency: int
):
    seeds = list(range(num_train_seeds))
    num_parallel_workers = min(10, len(seeds))

    # parameters for env_build()
    environment_kwargs = dict(max_episode_steps=T)
    env = env_builder()

    gamma = 0.99
    temp_start = 1.0
    temp_end = 1.0
    temp_steps = num_episodes * T
    temp_builder = build_linear_sched
    alpha_steps = num_episodes * T
    update_coefficient_builder = build_linear_sched

    state_size = env.unwrapped.observation_space.shape[0]
    action_size = int(env.unwrapped.action_space.n)
    H = 4096
    h = (H // (state_size + action_size),)

    # Pattern: <thing>_builder, <thing>_kwargs
    agent_kwargs = dict(
        action_space_dims=action_size,
        state_size=state_size,
        discount=gamma,
        temp_builder=temp_builder,
        temp_kwargs={
            "start": temp_start,
            "steps": temp_steps,
            "end"  : temp_end
        },
        hidden_dims=h,
        normalize_input=False,
        normalize_grad=True,
        grad_norm_threshold=1.,
        update_coefficient_actor_builder=update_coefficient_builder,
        update_coefficient_critic_builder=update_coefficient_builder,
    )

    processed_training_results = {}
    processed_evaluation_results = {}
    processed_distribution_results = {}
    agent_class = OneStepAC

    for alpha in [1e-3, 3e-3]:
        akwargs = agent_kwargs.copy()

        akwargs.update(
            dict(
                update_coefficient_actor_kwargs=
                {
                    "start": alpha,
                    "steps": alpha_steps,
                    "end"  : 0.001 * alpha
                },
                update_coefficient_critic_kwargs=
                {
                    "start": 25 * alpha,
                    "steps": alpha_steps,
                    "end"  : 0.001 * 25 * alpha
                }
            )
        )

        agent_name = '1StepAC: ' + f'$\\alpha={alpha:.1e}$'

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

        processed_distribution_results.update({
            agent_name: preprocess_for_distribution_plots(
                results, agent_name)
        })

    # ------- Health Report --------- #
    print_health_report(
        training_results=processed_training_results,
        evaluation_results=processed_evaluation_results,
        distribution_results=processed_distribution_results
    )

    # region Present

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
        results=processed_training_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/training_metrics/onestepac',
    )

    plot_evaluation_metrics(
        results=processed_evaluation_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/evaluation_metrics/onestepac',
    )

    plot_value_accuracy(
        results=processed_evaluation_results,
        metrics_to_compare=['V0', 'soft_G0'],
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/learning/onestepac',
    )

    plot_action_distribution(
        results=processed_distribution_results,
        file_root=f'{ENV_NAME}',
        save_dir='images/behaviors/onestepac',
    )

    # endregion


def ac_with_eligibility_traces(
        num_train_seeds: int,
        eval_num_episodes: int,
        reward_shaper: Callable,
        num_episodes: int,
        T: int,
        evaluate_frequency: int
):
    seeds = list(range(num_train_seeds))
    num_parallel_workers = min(10, len(seeds))

    # parameters for env_build()
    environment_kwargs = dict(max_episode_steps=T)

    env = env_builder()

    gamma = 0.99
    temp_start = 1.0
    temp_end = 1.0
    temp_steps = num_episodes * T
    temp_builder = build_linear_sched
    alpha_steps = num_episodes * T
    update_coefficient_builder = build_linear_sched

    state_size = env.unwrapped.observation_space.shape[0]
    action_size = int(env.unwrapped.action_space.n)
    H = 4096
    h = (H // (state_size + action_size),)

    # Pattern: <thing>_builder, <thing>_kwargs
    agent_kwargs = dict(
        action_space_dims=action_size,
        state_size=state_size,
        discount=gamma,
        temp_builder=temp_builder,
        temp_kwargs={
            "start": temp_start,
            "steps": temp_steps,
            "end"  : temp_end
        },
        hidden_dims=h,
        normalize_input=False,
        normalize_grad=True,
        grad_norm_threshold=1.,
        update_coefficient_actor_builder=update_coefficient_builder,
        update_coefficient_critic_builder=update_coefficient_builder,
    )

    processed_training_results = {}
    processed_evaluation_results = {}
    processed_distribution_results = {}
    agent_class = ACWithEligibilityTraces

    for alpha in [1e-3, 3e-3]:
        for lam in [0.1, 0.5, 0.9]:

            akwargs = agent_kwargs.copy()

            akwargs.update(
                dict(
                    update_coefficient_actor_kwargs=
                    {
                        "start": alpha,
                        "steps": alpha_steps,
                        "end"  : 0.001 * alpha
                    },
                    update_coefficient_critic_kwargs=
                    {
                        "start": 25 * alpha,
                        "steps": alpha_steps,
                        "end"  : 0.001 * 25 * alpha
                    },
                    lam_actor=lam,
                    lam_critic=lam,
                )
            )

            agent_name = 'ACwEligTrace: ' + (
                    f'$\\alpha={alpha: .1e}$, ' +
                    f'$\\lambda={lam}$'
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

            processed_distribution_results.update({
                agent_name: preprocess_for_distribution_plots(
                    results, agent_name)
            })

    # ------- Health Report --------- #
    print_health_report(
        training_results=processed_training_results,
        evaluation_results=processed_evaluation_results,
        distribution_results=processed_distribution_results
    )

    # region Present

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
        results=processed_training_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/training_metrics/ACEligTrace',
    )

    plot_evaluation_metrics(
        results=processed_evaluation_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/evaluation_metrics/ACEligTrace',
    )

    plot_value_accuracy(
        results=processed_evaluation_results,
        metrics_to_compare=['V0', 'soft_G0'],
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/learning/ACEligTrace',
    )

    plot_action_distribution(
        results=processed_distribution_results,
        file_root=f'{ENV_NAME}',
        save_dir='images/behaviors/ACEligTrace',
    )

    # endregion


# endregion

if __name__ == '__main__':
    global ENV_NAME

    def get_env_params():
        global ENV_NAME

        reduce_episodes = 1.0
        reward_shaper = base_reward_shaper

        if ENV_NAME == 'MountainCar':
            num_episodes = int(100 * reduce_episodes)
            T = 999
            evaluate_frequency = max(1, int(0.01 * num_episodes * T))
        elif ENV_NAME == 'CartPole':
            num_episodes = int(100 * reduce_episodes)
            T = 500
            evaluate_frequency = max(1, int(0.01 * num_episodes * T))
        elif ENV_NAME == 'Acrobot':
            num_episodes = int(100 * reduce_episodes)
            T = 500
            reward_shaper = acrobot_reward_shaper
            evaluate_frequency = max(1, int(0.05 * num_episodes * T))
        elif ENV_NAME == 'LunarLander':
            num_episodes = int(1500 * reduce_episodes)
            T = 500
            evaluate_frequency = max(1, int(0.1 * num_episodes * T))
        else:
            raise NotImplementedError


        return num_episodes, T, evaluate_frequency, reward_shaper

    num_train_seeds=5
    eval_num_episodes=10
    envs = [
        'CartPole',
        'LunarLander',
        'MountainCar',
        'Acrobot',
    ]

    experiment_fns = [
        reinforce,
        reinforce_with_baseline,
        one_step_ac,
        ac_with_eligibility_traces
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



