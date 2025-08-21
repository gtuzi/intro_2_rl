
import os
import random
import copy
import pickle
from typing import List, Callable, Dict, Any, Optional
from collections import Counter

from joblib import Parallel, delayed

from tqdm import tqdm
import pandas as pd
import numpy as np

from gymnasium import Env

from tabular_methods.utils import (
    Experience,
    QEpsGreedyAgent,
    SoftPolicy
)

# ************************************* #
# ************************************* #
# ************************************* #


# region Data Handling

def save_results(results: List[Dict[str, Any]], filepath: str):
    """
    Saves the experiment results object to a file using pickle.

    Args:
        results: The data structure returned by the training function (e.g., all_seeds_data).
        filepath: The full path to the file where results will be saved (e.g., 'results/sarsa.pkl').
    """
    # Ensure the directory exists
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results successfully saved to: {filepath}")


def load_results(filepath: str) -> List[Dict[str, Any]]:
    """
    Loads experiment results from a pickle file.

    Args:
        filepath: The full path to the file to load.

    Returns:
        The data structure containing the experiment results.
    """
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    print(f"Results successfully loaded from: {filepath}")
    return results


def preprocess_for_training_plots(
        all_seeds_data: List[Dict[str, Any]],
        algorithm_name: str
) -> pd.DataFrame:
    """
    Creates a DataFrame with only training metrics, one record per episode.
    """
    records = []
    for seed_run in all_seeds_data:
        seed = seed_run['seed']
        for episode_data in seed_run['training_episodes']:
            episode_steps_df = pd.DataFrame.from_dict(
                episode_data['steps_data'], orient='index')

            if not episode_steps_df.empty:
                final_step_time = episode_steps_df.index.max()
                record = {
                    'algorithm'           : algorithm_name, 'seed': seed,
                    'train_steps'         : final_step_time,
                    'episode_num'         : episode_data['episode_num'],
                    'mean_entropy'        : episode_steps_df['entropy'].mean(),
                    'mean_behavioral_loss': episode_steps_df[
                        'behavioral_agent_loss'].mean(),
                    'mean_target_loss'    : episode_steps_df[
                        'target_agent_loss'].mean(),
                }
                records.append(record)
    return pd.DataFrame(records)


def preprocess_for_evaluation_plots(
        all_seeds_data: List[Dict[str, Any]],
        algorithm_name: str
) -> pd.DataFrame:
    """
    Creates a DataFrame with only evaluation metrics, one record per evaluation event.
    """
    records = []
    for seed_run in all_seeds_data:
        seed = seed_run['seed']
        for episode_data in seed_run['training_episodes']:
            for step_time, step_metrics in episode_data['steps_data'].items():
                eval_results_dict = step_metrics.get("evaluation_results")
                if eval_results_dict:
                    record = {
                        'algorithm'  : algorithm_name, 'seed': seed,
                        'train_steps': step_time
                    }
                    for eval_type in ['soft_eval', 'hard_eval']:
                        if eval_type in eval_results_dict:
                            eval_data = eval_results_dict[eval_type]
                            eval_df = pd.DataFrame(eval_data["episodes"])
                            prefix = f"{eval_type.split('_')[0]}_"

                            record[f'mean_{prefix}eval_G0'] = eval_df[
                                'G0'].mean()
                            record[f'mean_{prefix}eval_sum_raw_rewards'] = \
                            eval_df['sum_raw_rewards'].mean()
                            record[f'mean_{prefix}eval_episode_length'] = \
                            eval_df['episode_length'].mean()
                            record[f'{prefix}eval_best_seen_score'] = \
                            eval_data['best_seen_score']
                            if 'mean_eval_V0' not in record:
                                record['mean_eval_V0'] = eval_df['V0'].mean()
                    records.append(record)
    return pd.DataFrame(records)


def preprocess_for_distribution_plots(
        all_seeds_data: List[Dict[str, Any]],
        algorithm_name: str
) -> pd.DataFrame:
    """
    Creates a DataFrame for distribution plots, handling both soft and hard
    evaluation types by creating an 'eval_type' column.
    """
    records = []
    for seed_run in all_seeds_data:
        seed = seed_run['seed']
        for episode_data in seed_run['training_episodes']:
            for step_time, step_metrics in episode_data['steps_data'].items():
                eval_results_dict = step_metrics.get("evaluation_results")

                if eval_results_dict:
                    # Iterate through both soft and hard evaluation results
                    for eval_type in ['soft_eval', 'hard_eval']:
                        if eval_type in eval_results_dict:
                            eval_data = eval_results_dict[eval_type]
                            for eval_episode in eval_data["episodes"]:
                                record = {
                                    'algorithm': algorithm_name,
                                    'seed': seed,
                                    'train_steps': step_time,
                                    'eval_type': eval_type.split('_')[0],  # 'soft' or 'hard'
                                    'eval_action_distribution': eval_episode.get('action_distribution'),
                                    'eval_state_visitation': eval_episode.get('state_visitation')
                                }
                                records.append(record)

    return pd.DataFrame(records)


def print_health_report(
        training_results: Dict[str, pd.DataFrame],
        evaluation_results: Dict[str, pd.DataFrame],
        distribution_results: Optional[Dict[str, pd.DataFrame]] = None
):
    """
    Analyzes preprocessed results and prints a comprehensive data health report,
    distinguishing between valid numbers, missing values (NaN/None), and Infs.
    """
    print("\n" + "=" * 70)
    print(" " * 23 + "EXPERIMENT HEALTH REPORT")
    print("=" * 70)

    all_algorithms = set(training_results.keys())
    all_algorithms.update(evaluation_results.keys())
    if distribution_results:
        all_algorithms.update(distribution_results.keys())

    for name in sorted(list(all_algorithms)):
        print(f"\n--- Algorithm: {name} ---")

        # --- Training Metrics Report ---
        print("  Training Data:")
        if name not in training_results or training_results[name].empty:
            print("    No data found.")
        else:
            df = training_results[name]
            metrics_to_check = [
                'mean_entropy', 'mean_behavioral_loss', 'mean_target_loss'
            ]

            for metric in metrics_to_check:
                if metric in df.columns:
                    total_points = len(df)
                    inf_count = np.isinf(df[metric]).sum()
                    nan_count = df[metric].isnull().sum()
                    numeric_count = total_points - inf_count - nan_count
                    print(
                        f"    - {metric:<35}: Numeric: {numeric_count:>4}, Missing (NaN/None): {nan_count:>4}, Infs: {inf_count:>4}")

        # --- Evaluation Metrics Report ---
        print("  Evaluation Data:")
        if name not in evaluation_results or evaluation_results[name].empty:
            print("    No data found.")
        else:
            df = evaluation_results[name]
            metrics_to_check = [
                'mean_soft_eval_G0', 'mean_hard_eval_G0',
                'mean_soft_eval_sum_raw_rewards',
                'mean_hard_eval_sum_raw_rewards',
                'mean_soft_eval_episode_length',
                'mean_hard_eval_episode_length',
                'soft_eval_best_seen_score', 'hard_eval_best_seen_score',
                'mean_eval_V0'
            ]

            for metric in metrics_to_check:
                if metric in df.columns:
                    total_points = len(df)
                    inf_count = np.isinf(df[metric]).sum()
                    nan_count = df[metric].isnull().sum()
                    numeric_count = total_points - inf_count - nan_count
                    print(
                        f"    - {metric:<35}: Numeric: {numeric_count:>4}, Missing (NaN/None): {nan_count:>4}, Infs: {inf_count:>4}")

        # --- Distribution Metrics Report ---
        print("  Distribution Data:")
        if not distribution_results or name not in distribution_results or \
                distribution_results[name].empty:
            print("    No data found.")
        else:
            df = distribution_results[name]
            metrics_to_check = [
                'eval_action_distribution', 'eval_state_visitation'
            ]

            for metric in metrics_to_check:
                if metric in df.columns:
                    total_points = len(df)
                    nan_count = df[metric].isnull().sum()
                    successful_count = total_points - nan_count
                    print(
                        f"    - {metric:<35}: Valid: {successful_count:>4}, Missing (NaN/None): {nan_count:>4}")

    print("\n" + "=" * 70 + "\n")


def discretize_state(state: np.ndarray, bins: List[np.ndarray]) -> tuple:
    """Converts a continuous state vector into a discrete tuple of bin indices."""
    # np.digitize finds which bin each state component falls into
    return tuple(np.digitize(s, b) for s, b in zip(state, bins))

# endregion


# ************************************* #
# ************************************* #
# ************************************* #


# region Evaluation Methods
def evaluate_single_episode(
        env_builder: Callable[[], Env],
        agent: QEpsGreedyAgent,
        seed: int,
        T: int,
        greedy_eval: bool = True,
        reward_shaper: Callable = lambda reward, done, t: reward,
        state_bins: List[np.ndarray] = None,
        **kwargs
) -> Dict[str, Any]:
    """
        Runs a single evaluation episode and returns its metrics.
    """

    env = env_builder()

    random.seed(seed)
    np.random.seed(seed)
    state, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    state_0 = state

    raw_rewards_over_time = []
    G0 = 0.
    gamma = agent.discount
    episode_length = 0
    action_counts = Counter()
    state_visitation = Counter()

    for t in range(T):
        episode_length += 1

        if state_bins:
            # Discretize the state into a hashable tuple and count it
            # Continuous case: Discretize the state
            # Use string for JSON safety
            tracked_state = str(discretize_state(state=state, bins=state_bins))
        else:
            tracked_state = state

        state_visitation.update([tracked_state])

        action, _ = agent.get_greedy_action(state) if greedy_eval else agent.act(state)

        action_counts.update([action])

        next_state, raw_reward, terminated, truncated, info = env.step(action)
        raw_rewards_over_time.append(raw_reward)

        done = terminated or truncated or (t + 1 == T)
        shaped_reward = reward_shaper(reward=raw_reward, done=done, t=t)
        G0 += (gamma ** t) * shaped_reward

        if done:
            break
        else:
            state = next_state

    V0 = agent.optimal_state_value(state_0) if greedy_eval else agent.state_value(state_0)

    env.close()

    # Return metrics for this single episode
    return {
        "seed": seed,
        "episode_length": episode_length,
        "sum_raw_rewards": sum(raw_rewards_over_time),
        "G0": G0,
        "V0": V0,
        "action_distribution": dict(action_counts),
        "state_visitation"   : dict(state_visitation)
    }


def sequential_evaluate(
        env_builder: Callable[[], Env],
        agent: QEpsGreedyAgent,
        seeds: List[int],
        T: int = 30,
        reward_shaper: Callable = lambda reward, done, t: reward,
        state_bins: Optional[List[np.ndarray]] = None,
        soft_eval: bool = False,
        hard_eval: bool = False,
        **kwargs
) -> Dict[str, Any]:
    """
    Runs both a soft (epsilon-greedy) and hard (greedy) evaluation.
    """
    results = {}
    for is_greedy in [False, True]:

        if is_greedy and (not hard_eval):
            continue # skip
        elif (not is_greedy) and (not soft_eval):
            continue

        eval_type = 'hard_eval' if is_greedy else 'soft_eval'

        evaluation_episodes = []
        best_seen_score = -float('inf')

        with tqdm(
                total=len(seeds),
                desc=f'Eval ({eval_type})',
                ncols=100
        ) as pbar:
            for seed in seeds:
                episode_data = evaluate_single_episode(
                    env_builder=env_builder,
                    agent=copy.deepcopy(agent),
                    seed=seed,
                    T=T,
                    greedy_eval=is_greedy,
                    reward_shaper=reward_shaper,
                    state_bins=state_bins,
                    **kwargs
                )

                current_score = episode_data['sum_raw_rewards']
                best_seen_score = max(best_seen_score, current_score)
                evaluation_episodes.append(episode_data)
                pbar.update(1)

        results[eval_type] = {
            "episodes"       : evaluation_episodes,
            "best_seen_score": best_seen_score
        }
    return results


def parallel_evaluate(
        env_builder: Callable[[], Env],
        agent: QEpsGreedyAgent,
        seeds: List[int],
        T: int = 30,
        reward_shaper: Callable = lambda reward, done, t: reward,
        state_bins: Optional[List[np.ndarray]] = None,
        max_workers: Optional[int] = None,
        soft_eval: bool = False,
        hard_eval: bool = False,
        **kwargs
) -> Dict[str, Any]:
    """
    Runs both soft and hard evaluations in parallel using joblib.
    """
    results = {}
    for is_greedy in [False, True]:

        if is_greedy and (not hard_eval):
            continue # skip
        elif (not is_greedy) and (not soft_eval):
            continue

        eval_type = 'hard_eval' if is_greedy else 'soft_eval'
        n_jobs = max_workers if max_workers is not None else -1

        evaluation_episodes = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_single_episode)(
                env_builder=env_builder,
                agent=copy.deepcopy(agent),
                seed=seed,
                T=T,
                greedy_eval=is_greedy,
                reward_shaper=reward_shaper,
                state_bins=state_bins,
                **kwargs
            ) for seed in tqdm(seeds, desc=f"Parallel Eval ({eval_type})")
        )

        best_seen_score = -float('inf')
        if evaluation_episodes:
            best_seen_score = max(ep['sum_raw_rewards'] for ep in evaluation_episodes)

        results[eval_type] = {
            "episodes": evaluation_episodes,
            "best_seen_score": best_seen_score
        }
    return results

# endregion eval

# ************************************* #
# ************************************* #
# ************************************* #

# region Train Methods
def train_single_seed(
        seed: int,
        env_builder: Callable[[], Env],
        behavioral_agent_class: type,
        behavioral_agent_kwargs: dict,
        target_agent_class: Optional[type] = None,
        target_agent_kwargs: Optional[dict] = None,
        reward_shaper: Callable = lambda reward, done, t: reward,
        sigma_fn: Callable[[int], float] = lambda t: 1.,
        T: int = 30,
        num_episodes: int = 10,
        do_eval: bool = True,
        eval_num_episodes: int = 10,
        evaluate_frequency: int = 1,
        state_bins: Optional[List[np.ndarray]] = None,
        parallel_eval: bool = False,
        soft_eval: bool = False,
        hard_eval: bool = False,
        eval_max_workers: Optional[int] = None,
        **kwargs
) -> Dict[str, Any]:
    """
        Runs a full training and evaluation process for a single seed.
    """

    env = env_builder()

    random.seed(seed)
    np.random.seed(seed)
    # Seed the environment's RNGs ONCE at the start of the entire run.
    env.reset(seed=seed)
    env.action_space.seed(seed)

    bkwargs = behavioral_agent_kwargs.copy()

    if issubclass(behavioral_agent_class, QEpsGreedyAgent):
        bkwargs['seed'] = seed

        if 'eps_schedule_builder' in bkwargs:
            eps_schedule_builder = bkwargs.pop('eps_schedule_builder')
            eps_schedule_kwargs = bkwargs.pop('eps_schedule_kwargs')
            bkwargs['eps'] = eps_schedule_builder(**eps_schedule_kwargs)
        else:
            assert 'eps'in bkwargs, 'Provide eps'

        if 'update_coefficient_builder' in bkwargs:
            update_coefficient_builder = bkwargs.pop('update_coefficient_builder')
            update_coefficient_kwargs = bkwargs.pop('update_coefficient_kwargs')
            bkwargs['update_coefficient'] = update_coefficient_builder(**update_coefficient_kwargs)
        else:
            assert 'update_coefficient' in bkwargs, \
                'Provide update_coefficient for behavioral agent'

    behavioral_agent = behavioral_agent_class(**bkwargs)

    target_agent = None
    if target_agent_class:
        tkwargs = target_agent_kwargs.copy()
        tkwargs['seed'] = seed

        if 'eps_schedule_builder' in tkwargs:
            eps_schedule_builder = tkwargs.pop('eps_schedule_builder')
            eps_schedule_kwargs = tkwargs.pop('eps_schedule_kwargs')
            tkwargs['eps'] = eps_schedule_builder(**eps_schedule_kwargs)
        else:
            assert 'eps'in tkwargs, 'Provide eps'

        if 'update_coefficient_builder' in tkwargs:
            update_coefficient_builder = tkwargs.pop('update_coefficient_builder')
            update_coefficient_kwargs = tkwargs.pop('update_coefficient_kwargs')
            tkwargs['update_coefficient'] = update_coefficient_builder(**update_coefficient_kwargs)
        else:
            assert 'update_coefficient' in tkwargs, \
                'Provide update_coefficient for target agent'

        target_agent = target_agent_class(**tkwargs)

    behavioral_agent.initialize()
    if target_agent is not None:
        target_agent.initialize()

    gamma = getattr(target_agent or behavioral_agent, 'discount', 1.0)
    seed_data = {"seed": seed, "training_episodes": []}

    with tqdm(
            total=num_episodes,
            desc=f'Train - seed_{seed}',
            ncols=100
    ) as pbar:

        global_time = 0
        last_eval_step = 0

        for episode in range(num_episodes):
            behavioral_agent.reset()
            if target_agent:
                target_agent.reset()

            # --- RESET FOR NEW EPISODE (NO SEED) ---
            # This call now advances the environment's RNG to a new state
            # for the new episode, ensuring variety between episodes.
            state, info = env.reset()
            state_0 = state

            episode_data = {
                "episode_num"       : episode,
                "steps_data"        : { },
                "evaluation_results": None
            }
            G0 = 0.
            action, p = behavioral_agent.act(state)

            for t in range(T):
                (
                    next_state,
                    raw_reward,
                    terminated,
                    truncated,
                    info
                ) = env.step(action)

                done = terminated or truncated or (t + 1 == T)

                shaped_reward = reward_shaper(
                    reward=raw_reward,
                    done=done,
                    t=t
                )

                G0 += (gamma ** t) * shaped_reward
                next_action, next_p = behavioral_agent.act(next_state)

                # --- Experience object creation (as before) ---
                sigmap = sigma_fn(t + 1)
                rhop = None

                if (
                    (target_agent is not None) and
                    (isinstance(target_agent, SoftPolicy))
                ):
                    target_next_p = target_agent.get_sa_probability(
                        next_state, next_action)
                    rhop = target_next_p / (next_p + 1e-8)

                e = Experience(
                    s=state,
                    a=action,
                    p=p,
                    r=shaped_reward,
                    sp=next_state,
                    ap=next_action,
                    pp=next_p,
                    done=int(done),
                    sigmap=sigmap,
                    rhop=rhop,
                    t=t
                )

                # --- Update Agents ---
                # agent.step() returns a loss value
                b_loss = behavioral_agent.step(e)

                t_loss = (
                    target_agent.step(e)
                    if target_agent is not None else None
                )

                # --- Evaluation Logic ---
                evaluation_results = None
                if do_eval and ((global_time - last_eval_step >= evaluate_frequency)):
                    # Must generate this list as it controls the number
                    # of episodes.
                    eval_seeds = [
                        seed + 1 + i for i in range(eval_num_episodes)
                    ]

                    eval_agent = behavioral_agent
                    if target_agent:
                        eval_agent = target_agent

                    if parallel_eval:
                        evaluation_results = parallel_evaluate(
                            env_builder=env_builder,
                            agent=eval_agent,
                            seeds=eval_seeds,
                            T=T,
                            reward_shaper=reward_shaper,
                            state_bins=state_bins,
                            max_workers=eval_max_workers,
                            soft_eval=soft_eval,
                            hard_eval=hard_eval,
                            **kwargs
                        )
                    else:
                        evaluation_results = sequential_evaluate(
                            env_builder=env_builder,
                            agent=eval_agent,
                            seeds=eval_seeds,
                            T=T,
                            reward_shaper=reward_shaper,
                            state_bins=state_bins,
                            soft_eval=soft_eval,
                            hard_eval=hard_eval,
                            **kwargs
                        )

                    last_eval_step = global_time

                # --- Per-step data collection ---
                step_metrics = {
                    "raw_reward"            : raw_reward,
                    "behavioral_agent_loss" : b_loss,
                    "target_agent_loss"     : t_loss,
                    "entropy"               : behavioral_agent.entropy(state),
                    "evaluation_results"    : evaluation_results
                }

                episode_data["steps_data"][global_time] = step_metrics
                global_time += 1

                if done:
                    break
                else:
                    state, action, p = next_state, next_action, next_p

            # --- Post-episode data finalization ---

            # -- Final Episode ----
            # This ensures an evaluation is run at the very end of training for this seed.
            last_recorded_step = global_time - 1
            if (
                    do_eval and
                    (global_time != last_eval_step) and
                    (episode == num_episodes - 1)  # Last episode
            ):
                eval_seeds = [
                    seed + 1 + i for i in range(eval_num_episodes)
                ]

                eval_agent = behavioral_agent
                if target_agent:
                    eval_agent = target_agent

                if parallel_eval:
                    evaluation_results = parallel_evaluate(
                        env_builder=env_builder,
                        agent=eval_agent,
                        seeds=eval_seeds,
                        T=T,
                        reward_shaper=reward_shaper,
                        state_bins=state_bins,
                        max_workers=eval_max_workers,
                        soft_eval=soft_eval,
                        hard_eval=hard_eval,
                        **kwargs
                    )
                else:
                    evaluation_results = sequential_evaluate(
                        env_builder=env_builder,
                        agent=eval_agent,
                        seeds=eval_seeds,
                        T=T,
                        reward_shaper=reward_shaper,
                        state_bins=state_bins,
                        soft_eval=soft_eval,
                        hard_eval=hard_eval,
                        **kwargs
                    )

                # Update the existing metrics from the last step with the new
                # evaluation results.
                # This does not overwrite the training data from that step.
                if last_recorded_step in episode_data["steps_data"]:
                    episode_data["steps_data"][last_recorded_step][
                        "evaluation_results"] = evaluation_results

            V0 = (
                    behavioral_agent.state_value(state_0)
                    if isinstance(behavioral_agent, QEpsGreedyAgent)
                    else None
                )

            episode_data["episode_length"] = t + 1
            episode_data["sum_raw_rewards"] = sum(
                s["raw_reward"] for s in episode_data["steps_data"].values()
            )
            episode_data["G0"] = G0
            episode_data["V0"] = V0

            seed_data["training_episodes"].append(episode_data)

            pbar.set_postfix(
                {"G0": f" {G0:.2f}", "V[0]": f"{V0 if V0 is not None else 0:.2f}"}
            )

            pbar.update(1)

    env.close()

    return seed_data


def sequential_train(
        env_builder: Callable[[], Env],
        behavioral_agent_class: type,
        behavioral_agent_kwargs: dict,
        target_agent_class: Optional[type] = None,
        target_agent_kwargs: Optional[dict] = None,
        reward_shaper: Callable = lambda reward, done, t: reward,
        sigma_fn: Callable[[int], float] = lambda t: 1.,
        T: int = 30,
        num_episodes: int = 10,
        do_eval: bool = True,
        eval_num_episodes: int = 10,
        evaluate_frequency: int = 1,
        train_seeds=(1, 2, 3, 4),
        soft_eval: bool = False,
        hard_eval: bool = False,
        state_bins: Optional[List[np.ndarray]] = None,
        **kwargs
) -> List[Dict[str, Any]]:
    """
    Trains an agent over multiple seeds by coordinating single-seed training runs.
    """
    all_seeds_data = []
    for seed in train_seeds:
        # Call the core single-seed function
        seed_data = train_single_seed(
            seed=seed,
            env_builder=env_builder,
            behavioral_agent_class=behavioral_agent_class,
            behavioral_agent_kwargs=behavioral_agent_kwargs,
            target_agent_class=target_agent_class,
            target_agent_kwargs=target_agent_kwargs,
            reward_shaper=reward_shaper,
            sigma_fn=sigma_fn,
            T=T,
            num_episodes=num_episodes,
            do_eval=do_eval,
            eval_num_episodes=eval_num_episodes,
            evaluate_frequency=evaluate_frequency,
            soft_eval=soft_eval,
            hard_eval=hard_eval,
            state_bins=state_bins,
            **kwargs
        )

        all_seeds_data.append(seed_data)

    return all_seeds_data


def parallel_train(
        env_builder: Callable[[], Env],
        num_parallel_workers: int,
        behavioral_agent_class: type,
        behavioral_agent_kwargs: dict,
        target_agent_class: Optional[type] = None,
        target_agent_kwargs: Optional[dict] = None,
        reward_shaper: Callable = lambda reward, done, t: reward,
        sigma_fn: Callable[[int], float] = lambda t: 1.,
        T: int = 30,
        num_episodes: int = 10,
        do_eval: bool = True,
        eval_num_episodes: int = 10,
        evaluate_frequency: int = 1,
        train_seeds=(1, 2, 3, 4),
        state_bins: Optional[List[np.ndarray]] = None,
        parallel_eval: bool = False,
        soft_eval: bool = False,
        hard_eval: bool = False,
        **kwargs
) -> List[Dict[str, Any]]:
    """
    Trains an agent over multiple seeds in parallel using joblib.
    """

    total_cpus = os.cpu_count() or 1
    available_for_eval = total_cpus - num_parallel_workers
    actual_eval_workers = max(1, available_for_eval)

    # Use tqdm for the progress bar directly on the iterable
    all_seeds_data = Parallel(n_jobs=num_parallel_workers)(
        delayed(train_single_seed)(
            seed=seed,
            env_builder=env_builder,
            behavioral_agent_class=behavioral_agent_class,
            behavioral_agent_kwargs=behavioral_agent_kwargs,
            target_agent_class=target_agent_class,
            target_agent_kwargs=target_agent_kwargs,
            reward_shaper=reward_shaper,
            sigma_fn=sigma_fn,
            T=T,
            num_episodes=num_episodes,
            do_eval=do_eval,
            eval_num_episodes=eval_num_episodes,
            evaluate_frequency=evaluate_frequency,
            parallel_eval=parallel_eval,
            eval_max_workers=actual_eval_workers,
            soft_eval=soft_eval,
            hard_eval=hard_eval,
            state_bins = state_bins,
            **kwargs
        ) for seed in tqdm(train_seeds, desc="Total Progress (Seeds)")
    )

    return all_seeds_data

# endregion train