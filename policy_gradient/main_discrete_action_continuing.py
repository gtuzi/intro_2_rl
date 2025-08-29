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

from shared.utils import (LinearSchedule, LinearWarmupDecaySchedule)

from policy_gradient.train import (
    parallel_train_continuing,
    sequential_train_continuing,
    preprocess_for_continuing_distribution_plots,
    preprocess_for_continuing_training_plots,
    preprocess_for_continuing_evaluation_plots,
    print_continuing_health_report
)

from policy_gradient.agents import ACWithEligibilityTracesContinuing


from policy_gradient.plot import (
    plot_action_distribution,
    plot_continuing_training_metrics,
    plot_continuing_evaluation_metrics,
    plot_continuing_value_accuracy,
    buil_high_contrast_palette
)

global ENV_NAME


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
        continuous = kwargs.pop('continuous', False)
        enable_wind = kwargs.pop('enable_wind', False)

        env = gym.make(
            "LunarLander-v3",
            continuous=continuous,
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


def build_linear_sched(start, steps, end = 0.0):
    return LinearSchedule(start, end=end, steps=steps)


def build_linear_warmup_decay_sched(start, steps, end = 0.0):
    warmup_steps = 0.3 * steps
    decay_steps = steps - warmup_steps

    return LinearWarmupDecaySchedule(
        start_lr = start * 0.1,
        peak_lr=start,
        end_lr=end,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps
    )


def ac_with_eligibility_traces(
        num_train_seeds: int,
        eval_num_episodes: int,
        reward_shaper: Callable,
        T: int,
        evaluate_frequency: int
):
    seeds = list(range(num_train_seeds))
    num_parallel_workers = min(10, len(seeds))

    # parameters for env_build()
    environment_kwargs = dict(max_episode_steps=T)

    env = env_builder()

    temp_start = 1.0
    temp_end = 1.0
    temp_steps = T
    alpha_steps = T

    state_size = env.unwrapped.observation_space.shape[0]
    action_size = int(env.unwrapped.action_space.n)
    H = 4096
    h = (H // (state_size + action_size),)

    # Pattern: <thing>_builder, <thing>_kwargs
    agent_kwargs = dict(
        action_space_dims=action_size,
        state_size=state_size,
        temp_builder=build_linear_sched,
        temp_kwargs={
            "start": temp_start,
            "steps": temp_steps,
            "end"  : temp_end
        },
        hidden_dims=h,
        normalize_input=False,
        norm_grad=False,
        norm_threshold=1.,
        update_coefficient_actor_builder=build_linear_sched,
        update_coefficient_critic_builder=build_linear_sched,
        update_coefficient_avg_reward_builder=build_linear_sched,
    )

    processed_training_results = {}
    processed_evaluation_results = {}
    processed_distribution_results = {}
    agent_class = ACWithEligibilityTracesContinuing

    for alpha in [1e-4, 5e-4]:
        for lam in [0.2, 0.5, 0.8]:

            akwargs = agent_kwargs.copy()

            akwargs.update(
                dict(
                    update_coefficient_actor_kwargs=
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
                    update_coefficient_avg_reward_kwargs={
                        "start": 20 * alpha,
                        "steps": alpha_steps,
                        "end" : 0.1 * 20 * alpha
                    },
                    lam_actor=lam,
                    lam_critic=lam,
                )
            )

            agent_name = 'ACwEligTraceContinuing: ' + (
                    f'$\\alpha={alpha: .1e}$, ' +
                    f'$\\lambda={lam}$'
            )

            results = parallel_train_continuing(
                env_builder=lambda: env_builder(**environment_kwargs),
                behavioral_agent_class=agent_class,
                behavioral_agent_kwargs=akwargs,
                T=T,
                reward_shaper=reward_shaper,
                train_seeds=seeds,
                do_eval=True,
                eval_num_episodes=eval_num_episodes,
                eval_max_steps = max(10, int(0.01 * T)),
                evaluate_frequency=evaluate_frequency,
                parallel_eval=True,
                soft_eval=True,
                hard_eval=True,
                num_parallel_workers=num_parallel_workers,
            )

            processed_training_results.update({
                agent_name: preprocess_for_continuing_training_plots(
                    results, agent_name)
            })

            processed_evaluation_results.update({
                agent_name: preprocess_for_continuing_evaluation_plots(
                    results, agent_name)
            })

            processed_distribution_results.update({
                agent_name: preprocess_for_continuing_distribution_plots(
                    results, agent_name)
            })


    # ------- Health Report --------- #
    print_continuing_health_report(
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

    plot_continuing_training_metrics(
        results=processed_training_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/training_metrics/ACwEligTraceContinuing',
    )

    plot_continuing_evaluation_metrics(
        results=processed_evaluation_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/evaluation_metrics/ACwEligTraceContinuing',
    )

    plot_continuing_value_accuracy(
        training_results=processed_training_results,
        evaluation_results=processed_evaluation_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/learning/ACwEligTraceContinuing',
    )

    plot_action_distribution(
        results=processed_distribution_results,
        file_root=f'{ENV_NAME}',
        save_dir='images/behaviors/ACwEligTraceContinuing',
    )

    # endregion


if __name__ == '__main__':
    global ENV_NAME

    def get_env_params():
        global ENV_NAME

        reducer = 1.
        if ENV_NAME == 'Acrobot':
            T = int(1 * int(5e5) * reducer)
            reward_shaper = acrobot_reward_shaper
        else:
            raise NotImplementedError

        # the smaller, the more frequent.
        # This is anchored on train-steps.
        evaluate_frequency = max(1, int(0.1 * T))

        return T, evaluate_frequency, reward_shaper


    num_train_seeds=5
    eval_num_episodes=10

    experiment_fns = [ac_with_eligibility_traces]

    for env in ['Acrobot']:
        ENV_NAME = env

        (
            T,
            evaluate_frequency,
            reward_shaper
        ) = get_env_params()

        for fn in experiment_fns:
            fn(
                num_train_seeds=num_train_seeds,
                eval_num_episodes=eval_num_episodes,
                reward_shaper=reward_shaper,
                T=T,
                evaluate_frequency=evaluate_frequency
            )

            exit(0)



