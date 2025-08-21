import gymnasium as gym
from gymnasium import Env
from scipy.stats import randint

from shared.utils import (
    LinearSchedule,
    CosineDecaySchedule,
    ExponentialSchedule
)

from tabular_methods.train import (
    parallel_train, sequential_train,
    preprocess_for_distribution_plots,
    preprocess_for_training_plots,
    preprocess_for_evaluation_plots, print_health_report
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
    nStepsQSigmaOffPolicy
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


def build_exponential_sched(start, steps, end = 0.0):
    return ExponentialSchedule(start, end=end, steps=steps)


def env_builder() -> Env:
    return build_env(ENV_NAME, render=False)



def run_on_policy_experiments(
        num_train_seeds: int = 10,
        eval_num_episodes=50,
):
    global ENV_NAME

    seeds = list(range(num_train_seeds))
    state_bins = None

    num_parallel_workers = min(10, len(seeds))

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
        alpha_end = 0.01
        alpha_steps = num_episodes * T
        update_coefficient_builder = build_linear_sched  # alpha / lr

    elif ENV_NAME == 'CliffWalking':
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

    elif ENV_NAME == 'Taxi':
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
            "end"  : epsilon_end
        },
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

    for agent_name, agent_class in [
        ('Sarsa', Sarsa),
        ('ExpectedSarsa', ExpectedSarsa),
        ('QLearning', QLearning),
        ('nStepSarsa-4', nStepSarsa)
    ]:

        akwargs = agent_kwargs.copy()

        if 'nStepSarsa' in agent_name:
            n = int(agent_name.split('-')[1])
            akwargs.update(dict(n=n))

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


def run_off_policy(
        num_train_seeds: int = 10,
        eval_num_episodes=50,
):
    global ENV_NAME

    seeds = list(range(num_train_seeds))
    state_bins = None

    num_parallel_workers = min(10, len(seeds))

    if ENV_NAME == 'FrozenLake':
        num_episodes = 10000
        T = 100
        evaluate_frequency = max(1, int(0.001 * num_episodes * T))

        q_init = 1.

        # --- Behavioral Agent Params
        b_epsilon_start = 1.0
        b_epsilon_end = 0.3
        b_eps_steps = num_episodes * T // 10
        b_eps_schedule_builder = build_linear_sched

        b_alpha_start = 0.3
        b_alpha_end = 0.1
        b_alpha_steps = num_episodes * T
        b_update_coefficient_builder = build_linear_sched


        # --- Target Agent Parameters
        t_epsilon_start = 1e-3
        t_epsilon_end = 1e-5
        t_eps_steps = num_episodes * T
        t_eps_schedule_builder = build_linear_sched

        t_alpha_start = 1e-1
        t_alpha_end = 1e-5
        t_alpha_steps = num_episodes * T
        t_update_coefficient_builder = build_linear_sched

    elif ENV_NAME == 'CliffWalking':
        num_episodes = 5000
        T = 20

        q_init = 0.

        b_epsilon_start = 1.0
        b_epsilon_end = 0.3
        b_eps_steps = num_episodes * T
        b_eps_schedule_builder = build_linear_sched

        b_alpha_start = 0.3
        b_alpha_end = 0.01
        b_alpha_steps = num_episodes * T
        b_update_coefficient_builder = build_linear_sched

        t_epsilon_start = 1.0
        t_epsilon_end = 0.01
        t_eps_steps = num_episodes * T
        t_eps_schedule_builder = build_linear_sched

        t_alpha_start = 0.3
        t_alpha_end = 0.01
        t_alpha_steps = num_episodes * T
        t_update_coefficient_builder = build_linear_sched

        # the smaller, the more frequent
        evaluate_frequency = max(1, int(0.05 * num_episodes * T))

    elif ENV_NAME == 'Taxi':
        num_episodes = 5000
        T = 50
        q_init = 0.

        b_epsilon_start = 1.0
        b_epsilon_end = 0.3
        b_eps_steps = num_episodes * T
        b_eps_schedule_builder = build_linear_sched

        b_alpha_start = 0.1
        b_alpha_end = 0.001
        b_alpha_steps = num_episodes * T
        b_update_coefficient_builder = build_linear_sched

        t_epsilon_start = 1e-3
        t_epsilon_end = 1e-5
        t_eps_steps = num_episodes * T
        t_eps_schedule_builder = build_linear_sched

        t_alpha_start = 1e-1
        t_alpha_end = 1e-3
        t_alpha_steps = num_episodes * T
        t_update_coefficient_builder = build_linear_sched

        evaluate_frequency = max(1, int(0.01 * num_episodes * T))

    else:
        raise NotImplementedError

    env = env_builder()

    gamma = 0.99

    # ---- Setup Behavioral Agent ----- #
    behavioral_agent_kwargs = dict(
        action_space_dims=int(env.action_space.n),
        obs_space_dims=int(env.observation_space.n),
        discount=gamma,
        qval_init=q_init,
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

    uniform_behavioral_akwargs = dict(
        obs_space_dims=behavioral_agent_kwargs['obs_space_dims'],
        action_space_dims=behavioral_agent_kwargs['action_space_dims'],
        distribution=randint,
        distribution_args=dict(
            low=0,
            high=behavioral_agent_kwargs['action_space_dims']
        )
    )

    behavioral_class = QLearning


    # ----- Setup Target Agent ----- #
    target_agent_kwargs = dict(
        action_space_dims=int(env.action_space.n),
        obs_space_dims=int(env.observation_space.n),
        discount=gamma,
        qval_init=q_init,
        eps_schedule_builder=t_eps_schedule_builder,
        eps_schedule_kwargs={
            "start": t_epsilon_start,
            "steps": t_eps_steps,
            "end"  : t_epsilon_end
        },
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

    for agent_name, target_class in [
        # ('OffPolicyNStepSarsa-4', nStepsSarsaOffPolicy),
        ('OffPolicyNStepQSigma-2', nStepsQSigmaOffPolicy),
        # ('OffPolicyNStepQSigma-4', nStepsQSigmaOffPolicy),
        ('OffPolicyNStepQSigma-4', nStepsQSigmaOffPolicy)
    ]:

        behaviora_kwargs = behavioral_agent_kwargs.copy()
        targeta_akwargs = target_agent_kwargs.copy()
        train_kwargs = dict()  # needed for sigma function

        if 'OffPolicyNStepSarsa-' in agent_name:
            n = int(agent_name.split('-')[1])
            targeta_akwargs.update(dict(n=n))
        elif 'OffPolicyNStepQSigma-' in agent_name:
            n = int(agent_name.split('-')[1])
            targeta_akwargs.update(dict(n=n))
            sigma_fn = lambda t: 0 if (t % 2 == 0) else 1.
            # sigma_fn = lambda t: 0.5
            train_kwargs.update(dict(sigma_fn=sigma_fn))

        results = parallel_train(
            env_builder=env_builder,
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
            **train_kwargs
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
        save_dir='images/training_metrics/off_policy',
    )

    plot_evaluation_metrics(
        results=processed_evaluation_results,
        color_map=color_map,
        file_root=f'{ENV_NAME}',
        save_dir='images/evaluation_metrics/off_policy',
    )

    plot_value_accuracy(
        results=processed_evaluation_results,
        metrics_to_compare=['V0', 'soft_G0'],
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


if __name__ == '__main__':
    global ENV_NAME

    num_train_seeds=10
    eval_num_episodes=100

    for env_name in ['FrozenLake', 'CliffWalking', 'Taxi']:

        ENV_NAME = env_name

        if 1:
            run_on_policy_experiments(
                num_train_seeds=num_train_seeds,
                eval_num_episodes=eval_num_episodes
            )

        if 0:
            run_off_policy(
                num_train_seeds=num_train_seeds,
                eval_num_episodes=eval_num_episodes
            )

    exit(0)