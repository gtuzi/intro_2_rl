import gymnasium as gym
from gymnasium import Env
from scipy.stats import randint

from shared.utils import (
    LinearSchedule,
    CosineDecaySchedule,
    ExponentialSchedule
)


from tabular_methods.train import (
    parallel_train,
    preprocess_for_distribution_plots,
    preprocess_for_training_plots,
    preprocess_for_evaluation_plots,
    print_health_report,
    sequential_train
)


from tabular_methods.plot import (
    plot_training_metrics,
    plot_evaluation_metrics,
    plot_value_accuracy,
    plot_action_distribution,
    buil_high_contrast_palette
)

from agents import MCOnPolicyFirstVisitGLIE, MCOffPolicy
from tabular_methods.utils import DiscreteActionRandomAgent

global ENV_NAME


def build_env(name: str, **kwargs) -> Env:
    render = kwargs.pop('render', False)

    if 'FrozenLake' in name:
        is_slippery = kwargs.pop('is_slippery', False)

        env = gym.make(
            'FrozenLake-v1',
            render_mode="human" if render else None,
            desc=None,
            map_name="4x4",
            is_slippery=is_slippery,
            **kwargs
        )
    elif 'Taxi' in name:
        env = gym.make(
            'Taxi-v3',
            render_mode="human" if render else None,
            **kwargs
        )
    elif 'CliffWalking' in name:
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


def build_exponential_sched(start, steps, end = 0.0):
    return ExponentialSchedule(start, end=end, steps=steps)


def env_builder(**kwargs) -> Env:
    return build_env(ENV_NAME, render=False, **kwargs)


def run_on_policy(
        num_train_seeds: int = 10,
        eval_num_episodes=50,
):
    global ENV_NAME

    seeds = list(range(num_train_seeds))
    num_parallel_workers = min(10, len(seeds))
    env_kwargs = { }

    if ENV_NAME == 'FrozenLake' or ENV_NAME == 'FrozenLake-Slippery':
        if ENV_NAME.lower() == 'FrozenLake-Slippery'.lower():
            env_kwargs['is_slippery'] = True
        else:
            env_kwargs['is_slippery'] = False

        num_episodes = 500
        T = 50

        eps_steps = num_episodes # eps.step() at end of episode
        eps_schedule_builder = build_linear_sched

        alpha_start = 0.3
        alpha_end = 0.01
        alpha_steps = num_episodes

        update_coefficient_builder = build_linear_sched  # alpha / lr

    elif ENV_NAME == 'CliffWalking':
        num_episodes = 5000
        T = 20

        eps_steps = num_episodes # eps.step() at end of episode
        eps_schedule_builder = build_linear_sched

        alpha_start = 0.5
        alpha_end = 0.01
        alpha_steps = num_episodes
        update_coefficient_builder = build_linear_sched  # alpha / lr

    elif ENV_NAME == 'Taxi':
        num_episodes = 5000
        T = 50

        eps_steps = num_episodes # eps.step() at end of episode
        eps_schedule_builder = build_linear_sched

        alpha_start = 1.0
        alpha_end = 0.2
        alpha_steps = num_episodes
        update_coefficient_builder = build_linear_sched  # alpha / lr

    else:
        raise NotImplementedError

    evaluate_frequency = max(1, int(0.01 * num_episodes * T))
    env = env_builder()
    gamma = 0.99

    agent_kwargs = dict(
        action_space_dims=int(env.action_space.n),
        obs_space_dims=int(env.observation_space.n),
        discount=gamma,
        eps_schedule_builder=eps_schedule_builder,
        update_coefficient_builder=update_coefficient_builder,
        update_coefficient_kwargs={
            "start": alpha_start,
            "steps": alpha_steps,
            "end"  : alpha_end
        }
    )

    processed_training_results = {}
    processed_evaluation_results = {}
    processed_distribution_results = {}

    for q_init in [-1, 1]:
        for eps in [0.1, 0.05]:

            for agent_name, agent_class in [
                ('MC: $1_{st}$-visit', MCOnPolicyFirstVisitGLIE),
            ]:
                akwargs = agent_kwargs.copy()

                agent_name += ', $q_0$: ' + f'{q_init}'
                agent_name += ', $\\varepsilon$: ' + f'{eps}'

                akwargs.update(
                    dict(
                        eps_schedule_kwargs={
                            "start": eps,
                            "steps": eps_steps,
                            "end"  : 0.001 * eps
                        },
                        q_init=q_init
                    )
                )

                results = parallel_train(
                    env_builder= lambda: env_builder(**env_kwargs),
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

    # region Present

    # ------- Health Report --------- #
    print_health_report(
        training_results=processed_training_results,
        evaluation_results=processed_evaluation_results,
        distribution_results=processed_distribution_results
    )

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
        save_dir='images/training_metrics/on_policy',
    )

    plot_evaluation_metrics(
        results=processed_evaluation_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/evaluation_metrics/on_policy',
    )

    plot_value_accuracy(
        results=processed_evaluation_results,
        metrics_to_compare=['V0', 'soft_G0'],
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/learning/on_policy',
    )

    plot_action_distribution(
        results=processed_distribution_results,
        file_root=f'{ENV_NAME}',
        save_dir='images/behaviors/on_policy',
    )

    #endregion


def run_off_policy_uniform_behavioral(
        num_train_seeds: int = 10,
        eval_num_episodes=50,
):
    global ENV_NAME

    seeds = list(range(num_train_seeds))
    num_parallel_workers = min(10, len(seeds))
    env_kwargs = {}

    if ENV_NAME == 'FrozenLake' or ENV_NAME == 'FrozenLake-Slippery':
        if ENV_NAME.lower() == 'FrozenLake-Slippery'.lower():
            env_kwargs['is_slippery'] = True
        else:
            env_kwargs['is_slippery'] = False

        num_episodes = 500
        T = 50

        # --- Target Agent Parameters
        t_eps_steps = num_episodes
        t_eps_schedule_builder = build_linear_sched
        t_alpha_start = 1e-1
        t_alpha_end = 1e-5
        t_alpha_steps = num_episodes
        t_update_coefficient_builder = build_linear_sched

    elif ENV_NAME == 'CliffWalking':
        num_episodes = 5000
        T = 20

        # --- Target Agent Params --- #
        t_eps_steps = num_episodes
        t_eps_schedule_builder = build_linear_sched

        t_alpha_start = 0.3
        t_alpha_end = 0.01
        t_alpha_steps = num_episodes
        t_update_coefficient_builder = build_linear_sched

    elif ENV_NAME == 'Taxi':
        num_episodes = 5000
        T = 50

        # --- Target Agent Params --- #
        t_eps_steps = num_episodes
        t_eps_schedule_builder = build_linear_sched
        t_alpha_start = 1e-1
        t_alpha_end = 1e-3
        t_alpha_steps = num_episodes
        t_update_coefficient_builder = build_linear_sched
    else:
        raise NotImplementedError

    # The smaller, the more frequent the evalutation,
    # pegged to global train steps
    evaluate_frequency = max(1, int(0.01 * num_episodes * T))
    env = env_builder()

    gamma = 0.99

    # ---- Setup Behavioral Agent ----- #
    behavioral_akwargs = dict(
        action_space_dims=int(env.action_space.n),
        obs_space_dims=int(env.observation_space.n),
        distribution=randint,
        distribution_args=dict(
            low=0,
            high=int(env.action_space.n)
        )
    )

    behavioral_class = DiscreteActionRandomAgent

    # ----- Setup Target Agent ----- #
    target_agent_kwargs = dict(
        action_space_dims=int(env.action_space.n),
        obs_space_dims=int(env.observation_space.n),
        discount=gamma,
        eps_schedule_builder=t_eps_schedule_builder,
        update_coefficient_builder=t_update_coefficient_builder,
        update_coefficient_kwargs={
            "start": t_alpha_start,
            "steps": t_alpha_steps,
            "end"  : t_alpha_end
        }
    )

    processed_training_results = {}
    processed_evaluation_results = {}
    processed_distribution_results = {}

    for q_init in [-1, 1]:
        for eps in [0.1, 0.05]:
            for agent_name, target_class in [
                ('MC-OffPolicy:', MCOffPolicy)]:

                agent_name += ', $q_0$: ' + f'{q_init}'
                agent_name += ', $\\varepsilon$: ' + f'{eps}'

                behaviora_kwargs = behavioral_akwargs.copy()
                targeta_akwargs = target_agent_kwargs.copy()

                targeta_akwargs.update(
                    dict(
                        eps_schedule_kwargs={
                            "start": eps,
                            "steps": t_eps_steps,
                            "end"  : 0.001 * eps
                        },
                        q_init=q_init
                    )
                )

                results = parallel_train(
                    env_builder=lambda: env_builder(**env_kwargs),
                    behavioral_agent_class=behavioral_class,
                    behavioral_agent_kwargs=behaviora_kwargs,
                    target_agent_class=target_class,
                    target_agent_kwargs=targeta_akwargs,
                    T=T,
                    num_episodes=num_episodes,
                    reward_shaper=reward_shaper,
                    train_seeds=seeds,
                    eval_num_episodes=eval_num_episodes,
                    evaluate_frequency=evaluate_frequency,
                    do_eval=True,
                    parallel_eval=True,
                    soft_eval=True,
                    hard_eval=True,
                    num_parallel_workers=num_parallel_workers,
                )

                # region post-process
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

                # endregion

    # region  Present

    # ----------- Health Status ----------- #
    print_health_report(
        training_results=processed_training_results,
        evaluation_results=processed_evaluation_results,
        distribution_results=processed_distribution_results
    )

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
        results=processed_training_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/training_metrics/off_policy_uniform_behavioral',
    )

    plot_evaluation_metrics(
        results=processed_evaluation_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/evaluation_metrics/off_policy_uniform_behavioral',
    )

    plot_value_accuracy(
        results=processed_evaluation_results,
        metrics_to_compare=['V0', 'soft_G0'],
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/learning/off_policy_uniform_behavioral',
    )

    plot_action_distribution(
        results=processed_distribution_results,
        eval_types_to_plot=['soft', 'hard'],
        file_root=f'{ENV_NAME}',
        save_dir='images/behaviors/off_policy_uniform_behavioral',
    )

    # endregion


def run_off_policy_greedy_behavioral(
        num_train_seeds: int = 10,
        eval_num_episodes=50,
):
    global ENV_NAME

    seeds = list(range(num_train_seeds))
    num_parallel_workers = min(10, len(seeds))
    env_kwargs = {}

    if ENV_NAME == 'FrozenLake' or ENV_NAME == 'FrozenLake-Slippery':
        if ENV_NAME.lower() == 'FrozenLake-Slippery'.lower():
            env_kwargs['is_slippery'] = True
        else:
            env_kwargs['is_slippery'] = False

        num_episodes = 500
        T = 50

        # --- Behavioral Agent Params
        b_q_init = 1.0
        b_epsilon_start = 1.0
        b_epsilon_end = 0.2
        b_eps_steps = num_episodes
        b_eps_schedule_builder = build_linear_sched
        b_alpha_start = 0.3
        b_alpha_end = 0.1
        b_alpha_steps = num_episodes
        b_update_coefficient_builder = build_linear_sched

        # --- Target Agent Parameters
        t_eps_steps = num_episodes
        t_eps_schedule_builder = build_linear_sched
        t_alpha_start = 1e-1
        t_alpha_end = 1e-5
        t_alpha_steps = num_episodes
        t_update_coefficient_builder = build_linear_sched

    elif ENV_NAME == 'CliffWalking':
        num_episodes = 5000
        T = 20

        # -- Behavioral Agent Params --- #
        b_q_init = 0.
        b_epsilon_start = 1.0
        b_epsilon_end = 0.2
        b_eps_steps = num_episodes
        b_eps_schedule_builder = build_linear_sched
        b_alpha_start = 0.3
        b_alpha_end = 0.01
        b_alpha_steps = num_episodes
        b_update_coefficient_builder = build_linear_sched

        # --- Target Agent Params --- #
        t_eps_steps = num_episodes
        t_eps_schedule_builder = build_linear_sched

        t_alpha_start = 0.3
        t_alpha_end = 0.01
        t_alpha_steps = num_episodes
        t_update_coefficient_builder = build_linear_sched

    elif ENV_NAME == 'Taxi':
        num_episodes = 5000
        T = 50

        # --- Behavioral Agent Params --- #
        b_q_init = 0.
        b_epsilon_start = 1.0
        b_epsilon_end = 0.2
        b_eps_steps = num_episodes
        b_eps_schedule_builder = build_linear_sched
        b_alpha_start = 0.1
        b_alpha_end = 0.001
        b_alpha_steps = num_episodes * T
        b_update_coefficient_builder = build_linear_sched

        # --- Target Agent Params --- #
        t_eps_steps = num_episodes
        t_eps_schedule_builder = build_linear_sched
        t_alpha_start = 1e-1
        t_alpha_end = 1e-3
        t_alpha_steps = num_episodes
        t_update_coefficient_builder = build_linear_sched

    else:
        raise NotImplementedError

    # the smaller, the more frequent
    evaluate_frequency = max(1, int(0.01 * num_episodes * T))
    env = env_builder()

    gamma = 0.99

    # ---- Setup Behavioral Agent ----- #
    behavioral_agent_kwargs = dict(
        action_space_dims=int(env.action_space.n),
        obs_space_dims=int(env.observation_space.n),
        discount=gamma,
        q_init=b_q_init,
        eps_schedule_builder=b_eps_schedule_builder,
        eps_schedule_kwargs={
            "start": b_epsilon_start,
            "steps": b_eps_steps,
            "end"  : b_epsilon_end
        },
        update_coefficient_builder=b_update_coefficient_builder,
        update_coefficient_kwargs={
            "start": b_alpha_start,
            "steps": b_alpha_steps,
            "end"  : b_alpha_end
        }
    )

    behavioral_class = MCOnPolicyFirstVisitGLIE

    # ----- Setup Target Agent ----- #
    target_agent_kwargs = dict(
        action_space_dims=int(env.action_space.n),
        obs_space_dims=int(env.observation_space.n),
        discount=gamma,
        eps_schedule_builder=t_eps_schedule_builder,
        update_coefficient_builder=t_update_coefficient_builder,
        update_coefficient_kwargs={
            "start": t_alpha_start,
            "steps": t_alpha_steps,
            "end"  : t_alpha_end
        }
    )

    processed_training_results = {}
    processed_evaluation_results = {}
    processed_distribution_results = {}

    for q_init in [-1, 1]:
        for eps in [0.1, 0.05]:
            for agent_name, target_class in [
                ('MC-OffPolicy:', MCOffPolicy)]:

                agent_name += ', $q_0$: ' + f'{q_init}'
                agent_name += ', $\\varepsilon$: ' + f'{eps}'

                behaviora_kwargs = behavioral_agent_kwargs.copy()
                targeta_akwargs = target_agent_kwargs.copy()

                targeta_akwargs.update(
                    dict(
                        eps_schedule_kwargs={
                            "start": eps,
                            "steps": t_eps_steps,
                            "end"  : 0.001 * eps
                        },
                        q_init=q_init
                    )
                )

                results = parallel_train(
                    env_builder=lambda: env_builder(**env_kwargs),
                    behavioral_agent_class=behavioral_class,
                    behavioral_agent_kwargs=behaviora_kwargs,
                    target_agent_class=target_class,
                    target_agent_kwargs=targeta_akwargs,
                    T=T,
                    num_episodes=num_episodes,
                    reward_shaper=reward_shaper,
                    train_seeds=seeds,
                    eval_num_episodes=eval_num_episodes,
                    evaluate_frequency=evaluate_frequency,
                    do_eval=True,
                    parallel_eval=True,
                    soft_eval=True,
                    hard_eval=True,
                    num_parallel_workers=num_parallel_workers,
                )

                # region post-process
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

                # endregion

    # region  Present

    # ----------- Health Status ----------- #
    print_health_report(
        training_results=processed_training_results,
        evaluation_results=processed_evaluation_results,
        distribution_results=processed_distribution_results
    )

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
        results=processed_training_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/training_metrics/off_policy_greedy_behavioral',
    )

    plot_evaluation_metrics(
        results=processed_evaluation_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/evaluation_metrics/off_policy_greedy_behavioral',
    )

    plot_value_accuracy(
        results=processed_evaluation_results,
        metrics_to_compare=['V0', 'soft_G0'],
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/learning/off_policy_greedy_behavioral',
    )

    plot_action_distribution(
        results=processed_distribution_results,
        eval_types_to_plot=['soft', 'hard'],
        file_root=f'{ENV_NAME}',
        save_dir='images/behaviors/off_policy_greedy_behavioral',
    )

    # endregion


if __name__ == '__main__':
    global ENV_NAME

    num_train_seeds=10
    eval_num_episodes=100

    for env_name in [
                'FrozenLake',
                'FrozenLake-Slippery',
                'CliffWalking',
                # 'Taxi'
    ]:
        ENV_NAME = env_name

        run_on_policy(
            num_train_seeds=num_train_seeds,
            eval_num_episodes=eval_num_episodes
        )

        run_off_policy_uniform_behavioral(
            num_train_seeds=num_train_seeds,
            eval_num_episodes=eval_num_episodes
        )

        run_off_policy_greedy_behavioral(
            num_train_seeds=num_train_seeds,
            eval_num_episodes=eval_num_episodes
        )

    exit(0)