
import os
import random
import copy
import pickle
import warnings
from typing import List, Callable, Dict, Any, Optional, Union, Tuple
from collections import Counter

from joblib import Parallel, delayed

from tqdm import tqdm
import pandas as pd
import numpy as np

from gymnasium import Env

from policy_gradient.utils import (
    DiscreteActionSoftPolicy,
    DiscreteActionCriticStateValue, DiscreteActionCriticActionValue,
    DiscreteActionAgent, ContinuousActionSoftPolicy,
    ContinuousActionCriticStateValue
)
from shared.utils import Experience


# ************************************************************************** #
# Episodic
# ************************************************************************** #


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
                    'algorithm': algorithm_name, 'seed': seed,
                    'train_steps': final_step_time,
                    'episode_num': episode_data['episode_num'],
                    'mean_raw_reward': episode_steps_df['raw_reward'].mean(),
                    'mean_shaped_reward': episode_steps_df['shaped_reward'].mean(),
                    'mean_entropy': episode_steps_df['entropy'].mean(),
                    'mean_behavioral_loss': episode_steps_df['behavioral_agent_loss'].mean(),
                    'mean_target_loss': episode_steps_df['target_agent_loss'].mean(),
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
    Analyzes preprocessed episodic results and prints a comprehensive data health report.
    """
    print("\n" + "=" * 70)
    print(" " * 23 + "EPISODIC EXPERIMENT HEALTH REPORT")
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
                'mean_raw_reward', 'mean_shaped_reward',
                'mean_entropy', 'mean_behavioral_loss', 'mean_target_loss'
            ]
            for metric in metrics_to_check:
                if metric in df.columns:
                    total_points = len(df)
                    inf_count = np.isinf(df[metric]).sum()
                    nan_count = df[metric].isnull().sum()
                    numeric_count = total_points - inf_count - nan_count
                    print(
                        f"    - {metric:<35}: Numeric: {numeric_count:>6}, Missing (NaN/None): {nan_count:>6}, Infs: {inf_count:>6}")

        # --- Evaluation Metrics Report ---
        print("  Evaluation Data:")
        if name not in evaluation_results or evaluation_results[name].empty:
            print("    No data found.")
        else:
            df = evaluation_results[name]
            metrics_to_check = [
                'mean_soft_eval_G0', 'mean_hard_eval_G0',
                'mean_soft_eval_sum_raw_rewards', 'mean_hard_eval_sum_raw_rewards',
                'mean_soft_eval_sum_shaped_rewards', 'mean_hard_eval_sum_shaped_rewards',
                'mean_soft_eval_episode_length', 'mean_hard_eval_episode_length',
                'soft_eval_best_seen_score', 'hard_eval_best_seen_score', 'mean_eval_V0'
            ]
            for metric in metrics_to_check:
                if metric in df.columns:
                    total_points = len(df)
                    inf_count = np.isinf(df[metric]).sum()
                    nan_count = df[metric].isnull().sum()
                    numeric_count = total_points - inf_count - nan_count
                    print(
                        f"    - {metric:<35}: Numeric: {numeric_count:>6}, Missing (NaN/None): {nan_count:>6}, Infs: {inf_count:>6}")

        # --- Distribution Metrics Report ---
        print("  Distribution Data:")
        if not distribution_results or name not in distribution_results or distribution_results[name].empty:
            print("    No data found.")
        else:
            df = distribution_results[name]
            metrics_to_check = ['eval_action_distribution']
            for metric in metrics_to_check:
                if metric in df.columns:
                    total_points = len(df)
                    nan_count = df[metric].isnull().sum()
                    valid_count = total_points - nan_count
                    print(
                        f"    - {metric:<35}: Valid: {valid_count:>6}, Missing (NaN/None): {nan_count:>6}")

    print("\n" + "=" * 70 + "\n")


def discretize_state(state: np.ndarray, bins: List[np.ndarray]) -> tuple:
    """Converts a continuous state vector into a discrete tuple of bin indices."""
    # np.digitize finds which bin each state component falls into
    return tuple(np.digitize(s, b) for s, b in zip(state, bins))

# endregion


# region Evaluation Methods
def evaluate_single_episode(
        env_builder: Callable[[], Env],
        agent: Union[
            DiscreteActionSoftPolicy,
            DiscreteActionCriticStateValue,
            DiscreteActionCriticActionValue,
            ContinuousActionSoftPolicy,
            ContinuousActionCriticStateValue,
        ],
        seed: int,
        T: int,
        greedy_eval: bool = True,
        reward_shaper: Callable = lambda reward, state, done, t: reward,
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

    for t in range(T):
        episode_length += 1

        action, _ = (
            agent.get_greedy_action(state)
            if greedy_eval
            else agent.act(state)
        )

        if isinstance(agent, DiscreteActionAgent):
            action_counts.update([action])

        next_state, raw_reward, terminated, truncated, info = env.step(action)
        raw_rewards_over_time.append(raw_reward)

        done = terminated or truncated or (t + 1 == T)
        shaped_reward = reward_shaper(
            reward=raw_reward,
            state=next_state,
            done=done,
            t=t)

        G0 += (gamma ** t) * shaped_reward

        if done:
            break
        else:
            state = next_state

    V0 = None

    if isinstance(agent, DiscreteActionCriticActionValue):
        V0 = (
            agent.optimal_state_value(state_0) if greedy_eval
            else agent.state_value(state_0)
        )
    elif isinstance(
            agent,
            (
                    DiscreteActionCriticStateValue,
                    ContinuousActionCriticStateValue
            )
    ):
        V0 = agent.state_value(state_0)

    env.close()

    # Return metrics for this single episode
    return {
        "seed": seed,
        "episode_length": episode_length,
        "sum_raw_rewards": sum(raw_rewards_over_time),
        "G0": G0,
        "V0": V0,
        "action_distribution": dict(action_counts)
    }


def sequential_evaluate(
        env_builder: Callable[[], Env],
        agent: Union[DiscreteActionSoftPolicy, DiscreteActionCriticStateValue],
        seeds: List[int],
        T: int = 30,
        reward_shaper: Callable = lambda reward, state, done, t: reward,
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
                try:
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

                except Exception as e:
                    print(f"\n!!!!!! ERROR: Evaluation failed for seed {seed} !!!!!!")
                    print(f"Exception: {e}\n")
                    import traceback
                    traceback.print_exc()

                    pbar.update(1)
                    continue

        results[eval_type] = {
            "episodes"       : evaluation_episodes,
            "best_seen_score": best_seen_score
        }

    return results


def parallel_evaluate(
        env_builder: Callable[[], Env],
        agent: Union[DiscreteActionSoftPolicy, DiscreteActionCriticStateValue],
        seeds: List[int],
        T: int = 30,
        reward_shaper: Callable = lambda reward, state, done, t: reward,
        state_bins: Optional[List[np.ndarray]] = None,
        max_workers: Optional[int] = None,
        soft_eval: bool = False,
        hard_eval: bool = False,
        **kwargs
) -> Dict[str, Any]:
    """
        Runs both soft and hard evaluations in parallel using joblib.
    """

    def safe_single_seed_run(*args, **kwargs):
        seed = kwargs.get('seed', 'N/A')
        try:
            return evaluate_single_episode(*args, **kwargs)
        except Exception as e:
            print(f"\n!!!!!! ERROR: Evaluation failed for seed {seed} !!!!!!")
            print(f"Exception: {e}\n")
            import traceback
            traceback.print_exc()
            return None

    results = {}

    for is_greedy in [False, True]:

        if is_greedy and (not hard_eval):
            continue # skip
        elif (not is_greedy) and (not soft_eval):
            continue

        eval_type = 'hard_eval' if is_greedy else 'soft_eval'

        n_jobs = max_workers if max_workers is not None else -1

        evaluation_episodes = Parallel(n_jobs=n_jobs)(
            delayed(safe_single_seed_run)(
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

        # Filter out failed seeds/evaluation episodes
        evaluation_episodes = [r for r in evaluation_episodes if r is not None]

        best_seen_score = -float('inf')
        if evaluation_episodes:
            best_seen_score = max(ep['sum_raw_rewards'] for ep in evaluation_episodes)

        results[eval_type] = {
            "episodes": evaluation_episodes,
            "best_seen_score": best_seen_score
        }

    return results

# endregion eval


# region Train Methods
def train_single_seed(
        seed: int,
        env_builder: Callable[[], Env],
        behavioral_agent_class: type,
        behavioral_agent_kwargs: dict,
        target_agent_class: Optional[type] = None,
        target_agent_kwargs: Optional[dict] = None,
        reward_shaper: Callable = lambda reward, state, done, t: reward,
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
    env.reset(seed=seed)
    env.action_space.seed(seed)

    # --- Agent Construction ---
    bkwargs = behavioral_agent_kwargs.copy()
    if behavioral_agent_class:
        bkwargs['seed'] = seed
        for key in list(bkwargs.keys()):
            if key.endswith('_builder'):
                base_key = key.replace('_builder', '')
                builder = bkwargs.pop(key)
                builder_kwargs = bkwargs.pop(f"{base_key}_kwargs", {})
                bkwargs[base_key] = builder(**builder_kwargs)
    behavioral_agent = behavioral_agent_class(**bkwargs)

    target_agent = None
    if target_agent_class:
        tkwargs = target_agent_kwargs.copy()
        tkwargs['seed'] = seed
        for key in list(tkwargs.keys()):
            if key.endswith('_builder'):
                base_key = key.replace('_builder', '')
                builder = tkwargs.pop(key)
                builder_kwargs = tkwargs.pop(f"{base_key}_kwargs", {})
                tkwargs[base_key] = builder(**builder_kwargs)
        target_agent = target_agent_class(**tkwargs)

    behavioral_agent.initialize()
    if target_agent is not None:
        target_agent.initialize()

    gamma = getattr(target_agent or behavioral_agent, 'discount', 1.0)
    seed_data = {"seed": seed, "training_episodes": []}
    global_time = 0
    last_eval_step = 0

    with tqdm(
            total=num_episodes,
            desc=f'Train - seed_{seed}',
            ncols=100
    ) as pbar:

        for episode in range(num_episodes):
            behavioral_agent.reset()
            if target_agent:
                target_agent.reset()

            state, info = env.reset()
            state_0 = state
            episode_data = {
                "episode_num"       : episode,
                "steps_data"        : {},
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
                    state=next_state,
                    done=done,
                    t=t
                )

                G0 += (gamma ** t) * shaped_reward
                next_action, next_p = behavioral_agent.act(next_state)

                sigmap = sigma_fn(t + 1)
                rhop = None
                if (
                        (target_agent is not None) and
                        (isinstance(target_agent, DiscreteActionSoftPolicy))
                ):
                    target_next_p = target_agent.get_sa_probability(next_state, next_action)
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

                b_loss = behavioral_agent.step(e)
                t_loss = target_agent.step(e) if target_agent is not None else None

                evaluation_results = None

                if (
                        do_eval and
                        (global_time - last_eval_step >= evaluate_frequency)
                ):
                    eval_seeds = [
                        seed + 1000 + i
                        for i in range(eval_num_episodes)
                    ]
                    eval_agent = target_agent or behavioral_agent
                    eval_func = parallel_evaluate if parallel_eval else sequential_evaluate
                    evaluation_results = eval_func(
                        env_builder=env_builder, agent=eval_agent,
                        seeds=eval_seeds, T=T,
                        reward_shaper=reward_shaper, state_bins=state_bins,
                        max_workers=eval_max_workers,
                        soft_eval=soft_eval, hard_eval=hard_eval, **kwargs
                    )
                    last_eval_step = global_time

                step_metrics = {
                    "raw_reward"           : raw_reward,
                    "shaped_reward"        : shaped_reward,
                    "behavioral_agent_loss": b_loss,
                    "target_agent_loss"    : t_loss,
                    "entropy"              : behavioral_agent.entropy(state),
                    "evaluation_results"   : evaluation_results
                }
                episode_data["steps_data"][global_time] = step_metrics
                global_time += 1

                if done:
                    break
                else:
                    state, action, p = next_state, next_action, next_p

            current_episode_length = t + 1
            episode_data["episode_length"] = current_episode_length
            if global_time - 1 in episode_data["steps_data"]:
                episode_data["steps_data"][global_time - 1][
                    'episode_length'] = current_episode_length

            # (Post-episode final evaluation logic remains the same)
            last_recorded_step = global_time - 1
            if (
                    do_eval and
                    (global_time != last_eval_step) and
                    (episode == num_episodes - 1)
            ):
                eval_seeds = [
                    seed + 1000 + i
                    for i in range(eval_num_episodes)
                ]

                eval_agent = target_agent or behavioral_agent

                eval_func = (
                    parallel_evaluate
                    if parallel_eval
                    else sequential_evaluate
                )

                evaluation_results = eval_func(
                    env_builder=env_builder, agent=eval_agent,
                    seeds=eval_seeds, T=T,
                    reward_shaper=reward_shaper, state_bins=state_bins,
                    max_workers=eval_max_workers,
                    soft_eval=soft_eval, hard_eval=hard_eval, **kwargs
                )

                if last_recorded_step in episode_data["steps_data"]:
                    episode_data["steps_data"][last_recorded_step][
                        "evaluation_results"] = evaluation_results

            V0 = (
                behavioral_agent.state_value(state_0)
                if isinstance(behavioral_agent, DiscreteActionCriticStateValue)
                else None
            )

            episode_data["episode_length"] = t + 1
            episode_data["sum_raw_rewards"] = sum(
                s["raw_reward"]
                for s in episode_data["steps_data"].values()
            )
            episode_data["G0"] = G0
            episode_data["V0"] = V0
            seed_data["training_episodes"].append(episode_data)

            pbar.set_postfix(
                {
                    "G0"  : f" {G0:.2f}",
                    "V[0]": f"{V0 if V0 is not None else 0:.2f}"
                 }
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
        reward_shaper: Callable = lambda reward, state, done, t: reward,
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
        try:
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

        except Exception as e:
            print(f"\n!!!!!! ERROR: Training failed for seed {seed} !!!!!!")
            print(f"Exception: {e}\n")
            import traceback
            traceback.print_exc()
            continue

    return all_seeds_data


def parallel_train(
        env_builder: Callable[[], Env],
        num_parallel_workers: int,
        behavioral_agent_class: type,
        behavioral_agent_kwargs: dict,
        target_agent_class: Optional[type] = None,
        target_agent_kwargs: Optional[dict] = None,
        reward_shaper: Callable = lambda reward, state, done, t: reward,
        sigma_fn: Callable[[int], float] = lambda t: 1.,
        T: int = 30,
        num_episodes: int = 10,
        do_eval: bool = True,
        eval_num_episodes: int = 10,
        evaluate_frequency: int = 1,
        train_seeds: Union[Tuple[int, ...], List[int]] = (1, 2, 3, 4),
        state_bins: Optional[List[np.ndarray]] = None,
        parallel_eval: bool = False,
        soft_eval: bool = False,
        hard_eval: bool = False,
        **kwargs
) -> List[Dict[str, Any]]:
    """
    Trains an agent over multiple seeds in parallel using joblib.
    """

    def safe_single_train(*args, **kwargs):
        seed = kwargs.get('seed', 'N/A')
        try:
            return train_single_seed(*args, **kwargs)
        except Exception as e:
            print(f"\n!!!!!! ERROR: Training failed for seed {seed} !!!!!!")
            print(f"Exception: {e}\n")
            import traceback
            traceback.print_exc()
            return None

    total_cpus = os.cpu_count() or 1
    available_for_eval = total_cpus - num_parallel_workers
    actual_eval_workers = max(1, available_for_eval)

    # Use tqdm for the progress bar directly on the iterable
    all_seeds_data = Parallel(n_jobs=num_parallel_workers)(
        delayed(safe_single_train)(
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

    successful_results = [r for r in all_seeds_data if r is not None]

    return successful_results

# endregion train



# ************************************************************************** #
# Continuing Task
# ************************************************************************** #

def print_continuing_health_report(
        training_results: Dict[str, pd.DataFrame],
        evaluation_results: Dict[str, pd.DataFrame],
        distribution_results: Optional[Dict[str, pd.DataFrame]] = None,
        episode_length_results: Optional[Dict[str, pd.DataFrame]] = None
):
    """
    Analyzes preprocessed continuing task results and prints a data health report.
    """
    print("\n" + "=" * 70)
    print(" " * 18 + "CONTINUING TASK EXPERIMENT HEALTH REPORT")
    print("=" * 70)

    all_algorithms = set(training_results.keys())
    all_algorithms.update(evaluation_results.keys())
    if distribution_results:
        all_algorithms.update(distribution_results.keys())
    if episode_length_results:
        all_algorithms.update(episode_length_results.keys())

    for name in sorted(list(all_algorithms)):
        print(f"\n--- Algorithm: {name} ---")

        # --- Training Metrics Report ---
        print("  Training Data:")
        if name not in training_results or training_results[name].empty:
            print("    No data found.")
        else:
            df = training_results[name]
            metrics_to_check = [
                'raw_reward', 'shaped_reward', 'estimated_avg_reward',
                'td_error', 'agent_loss', 'mean_entropy'
            ]
            for metric in metrics_to_check:
                if metric in df.columns:
                    if pd.api.types.is_numeric_dtype(df[metric]):
                        total_points = len(df)
                        inf_count = np.isinf(df[metric]).sum()
                        nan_count = df[metric].isnull().sum()
                        numeric_count = total_points - inf_count - nan_count
                        print(
                            f"    - {metric:<35}: Numeric: {numeric_count:>6}, Missing (NaN/None): {nan_count:>6}, Infs: {inf_count:>6}")
                    else:
                        total_points = len(df)
                        nan_count = df[metric].isnull().sum()
                        valid_count = total_points - nan_count
                        print(
                            f"    - {metric:<35}: Valid (non-numeric): {valid_count:>6}, Missing (NaN/None): {nan_count:>6}")

        # --- Evaluation Metrics Report ---
        print("  Evaluation Data:")
        if name not in evaluation_results or evaluation_results[name].empty:
            print("    No data found.")
        else:
            df = evaluation_results[name]
            metrics_to_check = [
                'mean_soft_eval_mean_raw_reward', 'mean_hard_eval_mean_raw_reward',
                'mean_soft_eval_mean_shaped_rewards', 'mean_hard_eval_mean_shaped_rewards',
                'mean_soft_eval_total_steps', 'mean_hard_eval_total_steps'
            ]
            for metric in metrics_to_check:
                if metric in df.columns:
                    total_points = len(df)
                    inf_count = np.isinf(df[metric]).sum()
                    nan_count = df[metric].isnull().sum()
                    numeric_count = total_points - inf_count - nan_count
                    print(
                        f"    - {metric:<35}: Numeric: {numeric_count:>6}, Missing (NaN/None): {nan_count:>6}, Infs: {inf_count:>6}")

        # --- Distribution Metrics Report ---
        print("  Distribution Data:")
        if not distribution_results or name not in distribution_results or distribution_results[name].empty:
            print("    No data found.")
        else:
            df = distribution_results[name]
            metrics_to_check = ['eval_action_distribution']
            for metric in metrics_to_check:
                if metric in df.columns:
                    total_points = len(df)
                    nan_count = df[metric].isnull().sum()
                    valid_count = total_points - nan_count
                    print(
                        f"    - {metric:<35}: Valid: {valid_count:>6}, Missing (NaN/None): {nan_count:>6}")

        # --- Episode Length Data Report ---
        print("  Episode Length Data:")
        if not episode_length_results or name not in episode_length_results or episode_length_results[name].empty:
            print("    No data found.")
        else:
            df = episode_length_results[name]
            metrics_to_check = ['last_episode_length']
            for metric in metrics_to_check:
                if metric in df.columns:
                    total_points = len(df)
                    inf_count = np.isinf(df[metric]).sum()
                    nan_count = df[metric].isnull().sum()
                    numeric_count = total_points - inf_count - nan_count
                    print(
                        f"    - {metric:<35}: Numeric: {numeric_count:>6}, Missing (NaN/None): {nan_count:>6}, Infs: {inf_count:>6}")

    print("\n" + "=" * 70 + "\n")


def preprocess_for_continuing_training_plots(
        all_seeds_data: List[Dict[str, Any]],
        algorithm_name: str
) -> pd.DataFrame:
    """
    Creates a DataFrame with only per-step training metrics for continuing tasks.
    """
    records = []
    for seed_run in all_seeds_data:
        seed = seed_run['seed']
        for step, metrics in seed_run.get('training_steps', {}).items():
            record = {
                'algorithm': algorithm_name,
                'seed': seed,
                'train_steps': step,
                'estimated_avg_reward': metrics.get('estimated_avg_reward'),
                'td_error': metrics.get('td_error'),
                'agent_loss': metrics.get('agent_loss'),
                'raw_reward': metrics.get('raw_reward'),
                'shaped_reward': metrics.get('shaped_reward'),
                'mean_entropy': metrics.get('mean_entropy')
            }
            records.append(record)
    return pd.DataFrame(records)


def preprocess_for_continuing_evaluation_plots(
        all_seeds_data: List[Dict[str, Any]],
        algorithm_name: str
) -> pd.DataFrame:
    """
    Creates a DataFrame with only evaluation metrics for continuing tasks,
    one record per evaluation event.
    """
    records = []
    for seed_run in all_seeds_data:
        seed = seed_run['seed']
        for step, metrics in seed_run.get('training_steps', {}).items():
            eval_results_dict = metrics.get("evaluation_results")
            if eval_results_dict:
                record = {'algorithm'  : algorithm_name, 'seed': seed,
                          'train_steps': step
                }

                for eval_type in ['soft_eval', 'hard_eval']:
                    if eval_type in eval_results_dict:
                        eval_data = eval_results_dict[eval_type]
                        eval_df = pd.DataFrame(eval_data["seed_runs"])
                        prefix = f"{eval_type.split('_')[0]}_"

                        if not eval_df.empty:
                            record[f'mean_{prefix}eval_mean_raw_reward'] = \
                                eval_df['mean_raw_reward'].mean()
                            record[f'mean_{prefix}eval_mean_shaped_rewards'] = \
                                eval_df['mean_shaped_rewards'].mean()
                            record[f'mean_{prefix}eval_total_steps'] = eval_df[
                                'total_steps'].mean()

                records.append(record)

    return pd.DataFrame(records)


def preprocess_for_continuing_distribution_plots(
        all_seeds_data: List[Dict[str, Any]],
        algorithm_name: str
) -> pd.DataFrame:
    """
    Creates a DataFrame for distribution plots from continuing task evaluations.
    """
    records = []
    for seed_run in all_seeds_data:
        seed = seed_run['seed']
        for step, metrics in seed_run.get('training_steps', {}).items():
            eval_results_dict = metrics.get("evaluation_results")
            if eval_results_dict:
                for eval_type in ['soft_eval', 'hard_eval']:
                    if eval_type in eval_results_dict:
                        # Iterate through each individual run within the evaluation
                        for eval_run in eval_results_dict[eval_type]["seed_runs"]:
                            record = {
                                'algorithm'                 : algorithm_name,
                                'seed'                      : seed,
                                'train_steps'               : step,
                                'eval_type'                 : eval_type.split('_')[0],
                                'eval_action_distribution'  : eval_run.get('action_distribution')
                            }
                            records.append(record)

    return pd.DataFrame(records)


def evaluate_single_seed_continuing(
        env_builder: Callable[[], Env],
        agent: Any,
        seed: int,
        eval_max_steps: int,
        greedy_eval: bool = True,
        reward_shaper: Callable = lambda reward, state, done, t: reward,
        **kwargs
) -> Dict[str, Any]:
    """
    Runs a single pseudo-continuing evaluation for a fixed number of steps.
    Resets the environment upon termination or truncation to continue the run.
    No learning occurs.
    """
    env = env_builder()

    random.seed(seed)
    np.random.seed(seed)
    state, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    raw_rewards = []
    shaped_rewards = []
    action_counts = Counter()

    for t in range(eval_max_steps):
        action, _ = (
            agent.get_greedy_action(state)
            if greedy_eval
            else agent.act(state)
        )
        action_counts.update([action])

        next_state, raw_reward, terminated, truncated, info = env.step(action)
        raw_rewards.append(raw_reward)

        done = terminated or truncated

        shaped_reward = reward_shaper(
            reward=raw_reward,
            state=next_state,
            done=done,
            t=t
        )

        shaped_rewards.append(shaped_reward)

        if done:
            state, info = env.reset()
        else:
            state = next_state

    env.close()

    return {
        "seed"               : seed,
        "total_steps"        : len(raw_rewards),
        "mean_raw_reward"    : np.mean(raw_rewards) if raw_rewards else 0,
        "mean_shaped_rewards": np.mean(
            shaped_rewards) if shaped_rewards else 0,
        "action_distribution": dict(action_counts),
    }


def sequential_evaluate_continuing(
        env_builder: Callable[[], Env],
        agent: Any,
        seeds: List[int],
        eval_max_steps: int,
        reward_shaper: Callable = lambda reward, state, done, t: reward,
        soft_eval: bool = False,
        hard_eval: bool = False,
        **kwargs
) -> Dict[str, Any]:
    """
    Runs sequential evaluations for a continuing task over multiple seeds.
    """
    results = {}

    for is_greedy in [False, True]:
        if is_greedy and not hard_eval:
            continue
        elif not is_greedy and not soft_eval:
            continue

        eval_type = 'hard_eval' if is_greedy else 'soft_eval'

        seed_runs = []
        desc = f'Sequential Eval ({eval_type})'
        for seed in tqdm(seeds, desc=desc, ncols=100):
            try:
                run_data = evaluate_single_seed_continuing(
                    env_builder=env_builder,
                    agent=copy.deepcopy(agent),
                    seed=seed,
                    eval_max_steps=eval_max_steps,
                    greedy_eval=is_greedy,
                    reward_shaper=reward_shaper,
                    **kwargs
                )
                seed_runs.append(run_data)
            except Exception as e:
                print(
                    f"\n!!!!!! ERROR: Continuing evaluation failed for seed {seed} !!!!!!")
                print(f"Exception: {e}\n")
                continue

        results[eval_type] = {"seed_runs": seed_runs}

    return results


def parallel_evaluate_continuing(
        env_builder: Callable[[], Env],
        agent: Any,
        seeds: List[int],
        eval_max_steps: int,
        reward_shaper: Callable = lambda reward, state, done, t: reward,
        max_workers: Optional[int] = None,
        soft_eval: bool = False,
        hard_eval: bool = False,
        **kwargs
) -> Dict[str, Any]:
    """
    Runs parallel evaluations for a continuing task over multiple seeds.
    """
    def safe_single_seed_run_continuing(*args, **kwargs):
        seed = kwargs.get('seed', 'N/A')
        try:
            return evaluate_single_seed_continuing(*args, **kwargs)
        except Exception as e:
            print(f"\n!!!!!! ERROR: Continuing evaluation failed for seed {seed} !!!!!!")
            print(f"Exception: {e}\n")
            return None

    results = {}
    n_jobs = max_workers if max_workers is not None else -1

    for is_greedy in [False, True]:
        if is_greedy and not hard_eval:
            continue
        elif not is_greedy and not soft_eval:
            continue

        eval_type = 'hard_eval' if is_greedy else 'soft_eval'

        seed_runs = Parallel(n_jobs=n_jobs)(
            delayed(safe_single_seed_run_continuing)(
                env_builder=env_builder,
                agent=copy.deepcopy(agent),
                seed=seed,
                eval_max_steps=eval_max_steps,
                greedy_eval=is_greedy,
                reward_shaper=reward_shaper,
                **kwargs
            ) for seed in tqdm(seeds, desc=f"Parallel Eval ({eval_type})")
        )

        results[eval_type] = {"seed_runs": [r for r in seed_runs if r is not None]}

    return results


def train_single_seed_continuing(
        seed: int,
        env_builder: Callable[[], Env],
        behavioral_agent_class: type,
        behavioral_agent_kwargs: dict,
        T: int,
        reward_shaper: Callable = lambda reward, state, done, t: reward,
        do_eval: bool = True,
        eval_num_episodes: int = 10,
        eval_max_steps: int = 200,
        evaluate_frequency: int = 1000,
        parallel_eval: bool = False,
        soft_eval: bool = False,
        hard_eval: bool = False,
        eval_max_workers: Optional[int] = None,
        log_frequency: int = 100,
        **kwargs
) -> Dict[str, Any]:
    """
        Runs a full training and evaluation process for a single seed in a
        continuing task setting.
    """
    env = env_builder()

    random.seed(seed)
    np.random.seed(seed)
    state, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    # --- Agent Construction ---
    bkwargs = behavioral_agent_kwargs.copy()
    bkwargs['seed'] = seed
    for key in list(bkwargs.keys()):
        if key.endswith('_builder'):
            base_key = key.replace('_builder', '')
            builder = bkwargs.pop(key)
            builder_kwargs = bkwargs.pop(f"{base_key}_kwargs", {})
            bkwargs[base_key] = builder(**builder_kwargs)
    behavioral_agent = behavioral_agent_class(**bkwargs)
    behavioral_agent.initialize()

    seed_data = {"seed": seed, "training_steps": {}}
    last_eval_step = 0
    steps_in_current_episode = 0

    with tqdm(
            total=T,
            desc=f'Train (Continuing) - seed_{seed}',
            ncols=100
    ) as pbar:
        for t in range(T):
            steps_in_current_episode += 1

            action, p = behavioral_agent.act(state)
            (
                next_state,
                raw_reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            done = terminated or truncated

            shaped_reward = reward_shaper(
                reward=raw_reward,
                state=next_state,
                done=done,
                t=t
            )

            if done:
                # Fake continuing
                next_state_for_agent, info = env.reset()
                next_action, next_p = behavioral_agent.act(
                    next_state_for_agent)
            else:
                next_action, next_p = behavioral_agent.act(next_state)

            experience = Experience(
                s=state,
                a=action,
                p=p,
                r=shaped_reward,
                sp=next_state,
                ap=next_action,
                pp=next_p,
                done=int(terminated),
                t=t
            )

            step_metrics = behavioral_agent.step(experience)

            if not isinstance(step_metrics, dict):
                step_metrics = {'agent_loss': step_metrics}

            # --- Evaluation Logic ---
            evaluation_results = None
            if (
                    do_eval and
                    (t - last_eval_step >= evaluate_frequency or t == T - 1)
            ):
                eval_seeds = [
                    seed + 1000 + i
                    for i in range(eval_num_episodes)
                ]

                eval_func = (
                    parallel_evaluate_continuing
                    if parallel_eval
                    else sequential_evaluate_continuing
                )

                evaluation_results = eval_func(
                    env_builder=env_builder,
                    agent=behavioral_agent,
                    seeds=eval_seeds,
                    eval_max_steps=eval_max_steps,
                    reward_shaper=reward_shaper,
                    max_workers=eval_max_workers,
                    soft_eval=soft_eval,
                    hard_eval=hard_eval,
                    **kwargs
                )

                last_eval_step = t

            # --- Log All Metrics for This Step ---
            step_metrics['raw_reward'] = raw_reward
            step_metrics['shaped_reward'] = shaped_reward
            step_metrics['evaluation_results'] = evaluation_results
            step_metrics['mean_entropy'] = behavioral_agent.entropy(state)

            # The agent can also return this. That's why we're checking here
            if ('estimated_avg_reward' not in step_metrics) and (
            hasattr(behavioral_agent, 'R_bar')):
                step_metrics['estimated_avg_reward'] = float(
                    behavioral_agent.R_bar.detach().cpu()
                )

            if ('td_error' not in step_metrics) and (
            hasattr(behavioral_agent, 'delta')):
                step_metrics['td_error'] = float(
                    behavioral_agent.delta.detach().cpu()
                )

            if done:
                step_metrics['episode_length'] = steps_in_current_episode
            seed_data["training_steps"][t] = step_metrics

            if done:
                state = next_state_for_agent
            else:
                state = next_state

            if t % log_frequency == 0:
                postfix_data = {}
                if 'estimated_avg_reward' in step_metrics:
                    postfix_data['R_bar'] = (
                        f"{step_metrics['estimated_avg_reward']:.3f}"
                    )
                pbar.set_postfix(postfix_data)
            pbar.update(1)

    env.close()

    return seed_data


def sequential_train_continuing(
        env_builder: Callable[[], Env],
        behavioral_agent_class: type,
        behavioral_agent_kwargs: dict,
        T: int,
        train_seeds: Union[Tuple[int, ...], List[int]] = (1, 2, 3, 4),
        reward_shaper: Callable = lambda reward, state, done, t: reward,
        do_eval: bool = True,
        eval_num_episodes: int = 10,
        eval_max_steps: int = 200,
        evaluate_frequency: int = 1000,
        soft_eval: bool = False,
        hard_eval: bool = False,
        **kwargs
) -> List[Dict[str, Any]]:
    """
    Trains an agent on a continuing task over multiple seeds sequentially.
    """
    all_seeds_data = []

    for seed in train_seeds:
        try:
            seed_data = train_single_seed_continuing(
                seed=seed,
                env_builder=env_builder,
                behavioral_agent_class=behavioral_agent_class,
                behavioral_agent_kwargs=behavioral_agent_kwargs,
                T=T,
                reward_shaper=reward_shaper,
                do_eval=do_eval,
                eval_num_episodes=eval_num_episodes,
                eval_max_steps=eval_max_steps,
                evaluate_frequency=evaluate_frequency,
                soft_eval=soft_eval,
                hard_eval=hard_eval,
                **kwargs
            )

            all_seeds_data.append(seed_data)

        except Exception as e:
            print(f"\n!!!!!! ERROR: Continuing training failed for seed {seed} !!!!!!")
            print(f"Exception: {e}\n")
            import traceback
            traceback.print_exc()
            continue

    return all_seeds_data



def parallel_train_continuing(
        env_builder: Callable[[], Env],
        num_parallel_workers: int,
        behavioral_agent_class: type,
        behavioral_agent_kwargs: dict,
        T: int,
        reward_shaper: Callable = lambda reward, state, done, t: reward,
        train_seeds: Union[Tuple[int, ...], List[int]] = (1, 2, 3, 4),
        do_eval: bool = True,
        eval_num_episodes: int = 10,
        eval_max_steps: int = 200,
        evaluate_frequency: int = 1000,
        parallel_eval: bool = False,
        soft_eval: bool = False,
        hard_eval: bool = False,
        **kwargs
) -> List[Dict[str, Any]]:
    """
    Trains an agent on a continuing task over multiple seeds in parallel.
    """

    def safe_single_train_continuing(*args, **kwargs):
        seed = kwargs.get('seed', 'N/A')
        try:
            return train_single_seed_continuing(*args, **kwargs)
        except Exception as e:
            print(f"\n!!!!!! ERROR: Continuing training failed for seed {seed} !!!!!!")
            print(f"Exception: {e}\n")
            import traceback
            traceback.print_exc()
            return None

    # Determine how many workers to allocate for nested parallel evaluations
    total_cpus = os.cpu_count() or 1
    # Ensure at least 1 worker for the main training loop and 1 for evaluation
    actual_eval_workers = max(1, total_cpus - num_parallel_workers)

    all_seeds_data = Parallel(n_jobs=num_parallel_workers)(
        delayed(safe_single_train_continuing)(
            seed=seed,
            env_builder=env_builder,
            behavioral_agent_class=behavioral_agent_class,
            behavioral_agent_kwargs=behavioral_agent_kwargs,
            T=T,
            reward_shaper=reward_shaper,
            do_eval=do_eval,
            eval_num_episodes=eval_num_episodes,
            eval_max_steps=eval_max_steps,
            evaluate_frequency=evaluate_frequency,
            parallel_eval=parallel_eval,
            soft_eval=soft_eval,
            hard_eval=hard_eval,
            eval_max_workers=actual_eval_workers,
            **kwargs
        ) for seed in tqdm(train_seeds, desc="Total Progress (Seeds)")
    )

    successful_results = [r for r in all_seeds_data if r is not None]
    return successful_results