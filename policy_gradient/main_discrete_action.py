import os
import random
import copy
from typing import List, Callable
from joblib import Parallel, delayed

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import warnings


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*env\.shape to get variables from other wrappers is deprecated.*"
)
import gymnasium as gym
from gymnasium import Env

from shared.utils import LinearSchedule

from approximate_methods.utils import (
    DiscreteActionAgent,
    SoftPolicy,
    Experience,
    TileCodingFeature
)

from policy_gradient.agents import (
    Reinforce_LA,
    Reinforce,
    ReinforceBaseline,
    OneStepAC,
    ACWithEligibilityTraces,
    ACWithEligibilityTracesContinuing
)

global ENV_NAME
global RENDER
global MAX_EPISODE_STEPS


#region plots
def plot_experiments(
    result_list,
    exp_alphas,
    title=None,
    xlabel='Episode',
    ylabel='Total reward',
    figsize=(8, 5),
    save_fig=False,
    filename="experiment_rewards.png",
    save_dir="./images"
):
    """
    Plot each experiment's per-episode rewards, one full-color line per alpha.

    Parameters
    ----------
    result_list : List[List[float]] or List[np.ndarray]
        Each entry is a sequence of rewards (one per episode) for a given alpha.
    exp_alphas : List[float]
        List of alpha values, only used for legend labels.
    title : str, optional
        Plot title.
    xlabel : str, default 'Episode'
        Label for the x-axis.
    ylabel : str, default 'Total reward'
        Label for the y-axis.
    figsize : tuple, default (8, 5)
        Figure size.
    save_fig : bool, default False
        If True, save the figure to disk.
    filename : str, default "experiment_rewards.png"
        Filename for the saved figure.
    save_dir : str, default "./images"
        Directory in which to save the figure.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    exp_arrays = [np.asarray(exp).flatten() for exp in result_list]
    n_exp = len(exp_arrays)
    assert len(exp_alphas) == n_exp, "exp_alphas length must match number of experiments"

    fig, ax = plt.subplots(figsize=figsize)

    for arr, alpha in zip(exp_arrays, exp_alphas):
        episodes = np.arange(1, len(arr) + 1)
        ax.plot(
            episodes, arr,
            linewidth=2,
            label=f"α={alpha: .0e}"
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc='best', title='LR')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig, ax

#endregion plots

def entropy_from_list(lst, base=np.e):
    """
    Compute the entropy of a list of integers using NumPy,
    based on the frequency distribution of the integers.
    """
    # Count occurrences
    values, counts = np.unique(lst, return_counts=True)
    probs = counts / counts.sum()
    # Compute entropy
    entropy = -np.sum(probs * np.log(probs))
    # Change base if needed
    if base != np.e:
        entropy /= np.log(base)

    return entropy


def build_env() -> Env:
    global ENV_NAME
    global RENDER
    global MAX_EPISODE_STEPS

    if ENV_NAME.lower() == 'MountainCar'.lower():
        env = gym.make(
            'MountainCar-v0',
            render_mode="human" if RENDER else None)
        env._max_episode_steps = MAX_EPISODE_STEPS
    elif ENV_NAME.lower() == 'AirRaid'.lower():
        env = gym.make(
            "ALE/AirRaid-v5",
            obs_type="rgb",
            render_mode="human" if RENDER else None
        )
        env._max_episode_steps = MAX_EPISODE_STEPS
    elif ENV_NAME.lower() == 'LunarLander'.lower():
        env = gym.make(
            "LunarLander-v2",
            render_mode="human" if RENDER else None
        )
        env._max_episode_steps = MAX_EPISODE_STEPS
    elif ENV_NAME.lower() == 'CartPole'.lower():
        env = gym.make(
            "CartPole-v1",
            render_mode="human" if RENDER else None
        )

        env._max_episode_steps = MAX_EPISODE_STEPS
    else:
        raise NotImplementedError

    return copy.deepcopy(env)


def eval_env_episodic(
        env: Env,
        agent: DiscreteActionAgent,
        T: int = 30,
        num_episodes: int = 1,
        greedy: bool = True,
        seeds = None
):
    steps_per_episode = []
    sum_of_rewards_per_episode = []
    R0_over_episodes = []

    for ei, episode in enumerate(range(num_episodes)):
        # For seeds:
        # 1 -   None
        # 2 -   Single value
        # 3 -   Same as number of episodes
        seed = None
        if seeds is not None:
            if hasattr(seeds, '__getitem__'):
                assert len(seeds) == num_episodes
                seed = seeds[ei]
            elif isinstance(seeds, (float, int)):
                seed = seeds
            else:
                raise Exception("Seed format not recognized", str(seeds))

        # Set the seed for this episode
        random.seed(seed)
        np.random.seed(seed)

        # gymnasium v26 requires users to set seed
        # when resetting the environment
        s, info = env.reset(seed=seed)  # s[0]
        if greedy:
            a, p = agent.get_greedy_action(s)  # a[0]
        else:
            a, p = agent.act(s)

        R = 0

        for t in range(T):
            sp, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated or (t + 1 == T)
            ap, pp = agent.get_greedy_action(sp)

            R += r

            if done:
                steps_per_episode.append(t)
                sum_of_rewards_per_episode.append(R)
                break
            else:
                s = sp
                a = ap
                p = pp

        R0_over_episodes.append(R)

    return steps_per_episode, sum_of_rewards_per_episode, R0_over_episodes


def run_env_episodic(
        env: Env,
        behavioral_agent: DiscreteActionAgent,
        target_agent: DiscreteActionAgent = None,
        reward_shaper: Callable = lambda reward, state, done, t: reward,
        T: int = 30,
        num_episodes: int = 10,
        seeds=None,
        do_eval = False,
        evaluate_frequency = None,
        eval_num_episodes: int = 1,
        greedy_eval: bool = True
):
    global EPISODIC

    steps_per_episode = []
    sum_of_rewards_per_episode = []
    R0_over_episodes = []
    avg_rewards_over_episodes = []
    eval_steps_per_episode = []
    eval_sum_of_rewards_per_episode = []
    eval_R0_over_episodes = []

    try:
        # ----- Unlearn ----- #
        behavioral_agent.initialize()
        if target_agent is not None:
            target_agent.initialize()

        pbar = tqdm(range(num_episodes), desc="Training Episode", leave=False)

        for ei in pbar:
            # For seeds:
            # 1 -   None
            # 2 -   Single value
            # 3 -   Same as number of episodes
            seed = None
            if seeds is not None:
                if hasattr(seeds, '__getitem__'):
                    assert len(seeds) == num_episodes
                    seed = seeds[ei]
                elif isinstance(seeds, (float, int)):
                    seed = seeds
                else:
                    raise Exception("Seed format not recognized", str(seeds))

            # Set the seed for this episode
            random.seed(seed)
            np.random.seed(seed)

            # Noise state reset (not exploration level).
            # Clear any trajectories.
            # Clear any eligibility traces
            behavioral_agent.reset()
            if target_agent is not None:
                target_agent.reset()

            # gymnasium v26 requires users to set seed
            # when resetting the environment
            s, info = env.reset(seed=seed)  # s[0]
            a, p = behavioral_agent.act(s)  # a[0]

            rho = None
            if (target_agent is not None) and (
                    isinstance(target_agent, SoftPolicy)
            ):
                target_p = target_agent.get_sa_probability(s, a)
                rho = target_p / p

            R = 0
            avg_rewards = [0]
            entropies = []
            actions = [a]

            # For the pseudo- continuing case
            pbar_time = None
            if T > 1e4:
                pbar_time = tqdm(range(T), desc="Training", leave=False)

            for t in pbar_time if pbar_time is not None else range(T):
                sp, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated or (t + 1 == T)

                reward = reward_shaper(reward=r, state=sp, done=done, t=t)
                ap, pp = behavioral_agent.act(sp)
                entropy = behavioral_agent.entropy(sp)

                # R += r
                R += reward

                entropies.append(entropy)
                actions.append(ap)
                avg_rewards.append(R / (t + 1))

                # If off-policy, capture target probabilities
                rhop = None
                if (target_agent is not None) and (
                        isinstance(target_agent, SoftPolicy)
                ):
                    target_pp = target_agent.get_sa_probability(sp, ap)
                    rhop = target_pp / pp

                e = Experience(
                    s=s,
                    a=a,
                    p=p,
                    r=reward,
                    sp=sp,
                    ap=ap,
                    pp=pp,
                    done=int(done),
                    rho=rho,
                    rhop=rhop,
                    t=t)

                behavioral_agent.step(e)

                if target_agent is not None:
                    target_agent.step(e)

                if (t < (T - 1)) and (not EPISODIC) and done:
                    s, info = env.reset(seed=seed)
                    a, p = behavioral_agent.act(s)
                elif done:
                    steps_per_episode.append(t)
                    sum_of_rewards_per_episode.append(R)
                    break
                else:
                    s = sp
                    a = ap
                    p = pp
                    rho = rhop

                if pbar_time is not None:
                    pbar_time.set_postfix_str(f"Train Sum(r)={R:.2e}, Avg(r)={avg_rewards[-1]:.2e}, Entropy={entropy: .2f}")


            R0_over_episodes.append(R)
            avg_rewards_over_episodes.append(avg_rewards)

            eval_stat = ''

            if (
                    do_eval and
                    (
                            (
                                    (ei % evaluate_frequency == 0) and
                                    (eval_num_episodes > 0)
                            ) or
                            (ei + 1 == num_episodes)  # evaluate last episode
                    )
            ):
                eval_agent = behavioral_agent

                if target_agent is not None:
                    eval_agent = target_agent

                steps, sum_r, r0 = eval_env_episodic(
                    env,
                    agent=eval_agent,
                    seeds=seed,
                    T=T,
                    num_episodes=eval_num_episodes,
                    greedy=greedy_eval,
                )

                eval_steps_per_episode += steps
                eval_sum_of_rewards_per_episode += sum_r
                eval_R0_over_episodes.append(np.mean(r0))
                eval_stat = f'eval score={np.mean(r0):.2e}'


            pbar.set_postfix_str(
                f"train score={R:.2e}, " +
                eval_stat +
                f", Policy entropy: {np.mean(entropies): .2e}, " +
                f"Actions entropy: {entropy_from_list(actions): .2e}"
            )

        return dict(
            steps_per_episode=steps_per_episode,
            sum_of_rewards_per_episode=sum_of_rewards_per_episode,
            R0_over_episodes=R0_over_episodes,
            avg_rewards_over_episodes=avg_rewards_over_episodes,
            eval_steps_per_episode=eval_steps_per_episode,
            eval_sum_of_rewards_per_episode=eval_sum_of_rewards_per_episode,
            eval_R0_over_episodes=eval_R0_over_episodes
        )

    except Exception as ex:
        return dict(
            steps_per_episode=[-np.inf] * num_episodes,
            sum_of_rewards_per_episode=[-np.inf] * num_episodes,
            R0_over_episodes=[-np.inf] * num_episodes,
            avg_rewards_over_episodes=[-np.inf] * num_episodes,
            eval_steps_per_episode=[-np.inf] * num_episodes,
            eval_sum_of_rewards_per_episode=[-np.inf] * num_episodes,
            eval_R0_over_episodes=[-np.inf] * num_episodes
        )


def run_one_experiment(
        model: str,
        num_episodes,
        T,
        alpha,
        seeds,
        alpha_builder = lambda _a : _a,
        reward_shaper: Callable = lambda reward, state, done, t: reward,
):
    global NORM_GRAD

    env = build_env()

    if model == 'Reinforce_LA':
        do_eval = True
        target_agent = None

        num_tilings = 8
        num_tiles = 8
        max_size = 4096

        x0_low, x1_low = env.observation_space.low
        x0_high, x1_high = env.observation_space.high

        '''
            From Section 10.1:
                We used 8 tilings, with each tile covering 1/8th of 
                the bounded distance in each dimension
        '''
        _feature_fn = TileCodingFeature(
            max_size, num_tiles, num_tilings, x0_low, x1_low, x0_high, x1_high)

        feature_fn = lambda _s: _feature_fn(_s, -1)
        agent = Reinforce_LA(
            feature_size=max_size,
            action_space_dims=int(env.action_space.n),
            update_coefficient=alpha_builder(alpha),
            policy_feature_fn=feature_fn,
            discount=0.99)
    elif model == 'Reinforce':
        do_eval = True
        target_agent = None

        state_size = env.unwrapped.observation_space.shape[0]
        action_size = int(env.unwrapped.action_space.n)

        h = 4096 // (state_size + action_size)
        agent = Reinforce(
            state_size=state_size,
            action_space_dims=action_size,
            update_coefficient=alpha_builder(alpha),
            hidden_dims=(h,),
            discount=0.99,
            norm_grad=NORM_GRAD
        )
    elif model == 'ReinforceBaseline':
        do_eval = True
        target_agent = None

        state_size = env.unwrapped.observation_space.shape[0]
        action_size = int(env.unwrapped.action_space.n)

        h = 4096 // (state_size + action_size)
        agent = ReinforceBaseline(
            state_size=state_size,
            action_space_dims=action_size,
            update_coefficient_policy=alpha_builder(alpha),
            update_coefficient_baseline=alpha_builder(1.2 * alpha),
            hidden_dims=(h, ),
            discount=0.99,
            norm_grad=NORM_GRAD
        )
    elif model == 'OneStepAC':
        do_eval = True
        target_agent = None

        state_size = env.unwrapped.observation_space.shape[0]
        action_size = int(env.unwrapped.action_space.n)

        h = 4096 // (state_size + action_size)

        agent = OneStepAC(
            state_size=state_size,
            action_space_dims=action_size,
            update_coefficient_policy=alpha_builder(alpha),
            update_coefficient_critic=alpha_builder(25 * alpha),
            hidden_dims=(h,),
            discount=0.99,
            norm_grad=NORM_GRAD)
    elif model == 'ACWithEligibilityTraces':
        do_eval = True
        target_agent = None

        state_size = env.unwrapped.observation_space.shape[0]
        action_size = int(env.unwrapped.action_space.n)

        h = 4096 // (state_size + action_size)

        agent = ACWithEligibilityTraces(
            state_size=state_size,
            action_space_dims=action_size,
            update_coefficient_policy=alpha_builder(alpha),
            lam_policy= 0.5,
            update_coefficient_critic=alpha_builder(25 * alpha),
            lam_critic=0.5,
            hidden_dims=(h,),
            discount=0.99,
            norm_grad=NORM_GRAD)
    elif model == 'ACWithEligibilityTracesContinuing':
        do_eval = False
        target_agent = None

        state_size = env.unwrapped.observation_space.shape[0]
        action_size = int(env.unwrapped.action_space.n)

        h = 4096 // (state_size + action_size)

        agent = ACWithEligibilityTracesContinuing(
            state_size=state_size,
            action_space_dims=action_size,
            update_coefficient_policy=alpha_builder(alpha),
            lam_policy= 0.8,
            update_coefficient_critic=alpha_builder(alpha),
            lam_critic=0.8,
            update_coefficient_avg_reward=alpha_builder(alpha),
            hidden_dims=(h,),
            norm_grad=NORM_GRAD)

    else:
        raise NotImplemented(f'{model} model not recognized')

    res = run_env_episodic(
        env=env,
        behavioral_agent=agent,
        target_agent=target_agent,
        reward_shaper=reward_shaper,
        T=T,
        num_episodes=num_episodes,
        do_eval=do_eval,
        eval_num_episodes=1,
        evaluate_frequency=1,
        seeds=seeds)

    del agent

    return dict(
        steps_per_episode=res['steps_per_episode'],
        sum_of_rewards_per_episode=res['sum_of_rewards_per_episode'],
        R0_over_episodes=res['R0_over_episodes'],
        eval_steps_per_episode = res['eval_steps_per_episode'],
        eval_sum_of_rewards_per_episode = res['eval_sum_of_rewards_per_episode'],
        eval_R0_over_episodes = res['eval_R0_over_episodes'],
        avg_rewards_over_episodes=res['avg_rewards_over_episodes']
    )


def experiments_parallel(
        model: str,
        num_episodes,
        T,
        reward_shaper: Callable,
        alphas = np.linspace(0, 1, num=50),
        alpha_builder: Callable = lambda _a: _a,
        num_experiments: int = 10,
        seeds=(1, 2)
):
    global EPISODIC
    global NORM_GRAD

    # Prepare a 2D array to store final means
    steps_per_episode = np.zeros((len(alphas), 1), dtype=np.float64)
    sum_of_rewards_per_episode = np.zeros((len(alphas), 1), dtype=np.float64)
    eval_steps_per_episode = np.zeros((len(alphas), 1), dtype=np.float64)
    eval_sum_of_rewards_per_episode = np.zeros((len(alphas), 1), dtype=np.float64)

    # Offset the original seeds
    seeds_per_experiment = [
        [s + e * len(seeds) for s in seeds]
        for e in range(num_experiments)
    ]

    # Loop over α's in serial
    completed = 0
    total = len(alphas)
    R0_over_alphas = []
    eval_R0_over_alphas = []
    avg_rewards_over_alphas = []

    for ia, alpha in enumerate(alphas):
        #    Launch n_experiments calls of run_one_experiment(...) *in parallel*
        #    Each call returns a single‐experiment‐average‐RMS.
        #    We use n_jobs=-1 to utilize all CPU cores by default.

        res = Parallel(n_jobs=-1)(
            delayed(run_one_experiment)(
                model=model,
                num_episodes=num_episodes,
                T=T,
                alpha=alpha,
                alpha_builder=alpha_builder,
                reward_shaper=reward_shaper,
                seeds=seeds_per_experiment[e])

            for e in range(num_experiments)
        )

        completed += 1
        steps = [r['steps_per_episode'] for r in res]
        sum_rewards = [r['sum_of_rewards_per_episode'] for r in res]
        R0_over_episodes = [r['R0_over_episodes'] for r in res]
        eval_R0_over_episodes = [r['eval_R0_over_episodes'] for r in res]
        avg_rewards_over_episodes = [r['avg_rewards_over_episodes'][0] for r in res]

        eval_steps = [
            r['eval_steps_per_episode'] for r in res
            if (r['eval_steps_per_episode'] is not None) and (len(r['eval_steps_per_episode']) > 0)
        ]

        eval_sum_rewards = [
            r['eval_sum_of_rewards_per_episode'] for r in res
            if (r['eval_sum_of_rewards_per_episode'] is not None) and (len(r['eval_sum_of_rewards_per_episode']) > 0)
        ]

        # 4) Average those n_experiments results to fill results_array
        valid_idx = [i for i, s in enumerate(steps)
                     if not ((np.inf in s) or (-np.inf) in s)]

        steps_per_episode[ia] = np.mean(
            [steps[vi] for vi in valid_idx]) if len(
            valid_idx) > 0 else np.inf

        sum_of_rewards_per_episode[ia] = np.mean(
            [sum_rewards[vi] for vi in valid_idx]) if len(
            valid_idx) > 0 else -np.inf

        r0 = [R0_over_episodes[vi] for vi in valid_idx] if len(
            valid_idx) > 0 else -np.inf

        if r0 != -np.inf:
            R0_over_alphas.append(np.mean(r0, axis=0))

        if not EPISODIC:
            avg_r = [
                avg_rewards_over_episodes[vi] for vi in valid_idx] if len(
                valid_idx) > 0 else -np.inf

            if avg_r != -np.inf:
                avg_rewards_over_alphas.append(np.mean(avg_r, axis=0)[1:])

        if EPISODIC:
            if len(eval_steps) > 0:
                eval_steps_per_episode[ ia] = np.mean(
                    [eval_steps[vi] for vi in valid_idx]) if len(
                    valid_idx) > 0 else np.inf

            if len(eval_sum_rewards) > 0:
                eval_sum_of_rewards_per_episode[ia] = np.mean(
                    [eval_sum_rewards[vi] for vi in valid_idx]) if len(
                    valid_idx) > 0 else -np.inf

        if len(eval_R0_over_episodes) > 0:
            eval_r0 = [eval_R0_over_episodes[vi] for vi in valid_idx] if len(
                valid_idx) > 0 else -np.inf

            if eval_r0 != -np.inf:
                eval_R0_over_alphas.append(np.mean(eval_r0, axis=0))

        print(
            f"\n{model} - Completed {100 * completed / total:.1f} %, "
            f"$\\alpha$: {alpha:.2e}, "
            f"steps/episode: {steps_per_episode[ia][0]:.2f}, "
            f"sum(r)/episode: {sum_of_rewards_per_episode[ia][0]:.2f}, "
            f"eval steps/episode: {eval_steps_per_episode[ia][0]:.2f}, "
            f"eval sum(r)/episode: {eval_sum_of_rewards_per_episode[ia][0]:.2f}"
        )

    # ACWithEligibilityTracesContinuing
    if model == 'Reinforce_LA':
        title = 'Reinforce_LA: $\sum_t R_t$'
    elif model == 'Reinforce':
        title = 'Reinforce: $\mathbb{E}[\sum_t R_t]$'
    elif model == 'ReinforceBaseline':
        title = 'Reinforce-w-Baseline: $\mathbb{E}[\sum_t R_t]$'
    elif model == 'OneStepAC':
        title = 'One-Step Actor-Critic: $\mathbb{E}[\sum_t R_t]$'
    elif model == 'ACWithEligibilityTraces':
        title = 'Actor-Critic with Eligibility Traces: $\mathbb{E}[\sum_t R_t]$'
    elif model == 'ACWithEligibilityTracesContinuing':
        title = 'Continuing: AC with Eligibility Traces: $\mathbb{E}[\\frac{R_t}{step}]$'
    else:
        raise NotImplemented

    if not EPISODIC:
        ng_str = '_normgrad' if NORM_GRAD else ''

        plot_experiments(
            avg_rewards_over_alphas,
            exp_alphas=alphas.tolist(),
            title=title + " - Train",
            xlabel='Step',
            ylabel='Avg. Reward / Step',
            save_fig=True,
            filename=f'{model}_avg_R_train{ng_str}.png'
        )
    else:

        ng_str = '_normgrad' if NORM_GRAD else ''

        plot_experiments(
            R0_over_alphas,
            exp_alphas=alphas.tolist(),
            title=title + " - Train",
            xlabel='Episode',
            save_fig=True,
            filename=f'{model}_G0_train{ng_str}.png',
        )

        if len(eval_R0_over_alphas):
            plot_experiments(
                eval_R0_over_alphas,
                exp_alphas=alphas.tolist(),
                title=title + " - Eval",
                xlabel='Episode',
                save_fig=True,
                filename=f'{model}_G0_eval{ng_str}.png'
            )

def reinforce_la(
        num_episodes,
        T,
        reward_shaper: Callable,
        alphas = np.linspace(0, 1, num=50),
        alpha_builder = lambda _a : _a,
        seeds=(1, 2)
):
    env = build_env()
    steps_per_episode = np.zeros((len(alphas), ), dtype=np.float64)
    sum_of_rewards_per_episode = np.zeros((len(alphas), ), dtype=np.float64)
    eval_steps_per_episode = np.zeros((len(alphas), ), dtype=np.float64)
    eval_sum_of_rewards_per_episode = np.zeros((len(alphas), ), dtype=np.float64)

    num_tilings = 8
    num_tiles = 8
    max_size = 4096

    x0_low, x1_low = env.unwrapped.observation_space.low
    x0_high, x1_high = env.unwrapped.observation_space.high

    '''
        From Section 10.1:
            We used 8 tilings, with each tile covering 1/8th of 
            the bounded distance in each dimension
    '''
    _feature_fn = TileCodingFeature(
        max_size, num_tiles, num_tilings, x0_low, x1_low, x0_high, x1_high)

    feature_fn = lambda _s: _feature_fn(_s, -1)

    for ia, alpha in enumerate(alphas):

        agent = Reinforce_LA(
            feature_size=max_size,
            action_space_dims=int(env.unwrapped.action_space.n),
            update_coefficient=alpha_builder(alpha),
            policy_feature_fn=feature_fn,
            discount=0.99
        )

        res = run_env_episodic(
            env=env,
            behavioral_agent=agent,
            reward_shaper=reward_shaper,
            T=T,
            num_episodes=num_episodes,
            seeds=seeds,
            eval_num_episodes=1,
            do_eval=True,
            evaluate_frequency=1,
            greedy_eval=True
        )

        steps = res['steps_per_episode']
        sum_rewards = res['sum_of_rewards_per_episode']
        eval_steps = res['eval_steps_per_episode']
        eval_sum_rewards = res['eval_sum_of_rewards_per_episode']

        # Average those n_experiments results to fill results_array
        steps_per_episode[ia] = np.mean(steps)
        sum_of_rewards_per_episode[ia] = np.mean(sum_rewards)
        eval_steps_per_episode[ia] = np.mean(eval_steps)
        eval_sum_of_rewards_per_episode[ia] = np.mean(eval_sum_rewards)

        print(
            f"\nReinforce_LA: Done α={alpha:.3e}, "
            f" steps/episode: {steps_per_episode[ia]:.4f}, "
            f"sum(r)/episode: {sum_of_rewards_per_episode[ia]:.4f}"
            f" eval steps/episode: {eval_steps_per_episode[ia]:.4f}, "
            f"eval sum(r)/episode: {eval_sum_of_rewards_per_episode[ia]:.4f}"
        )


def reinforce(
        num_episodes,
        T,
        reward_shaper: Callable,
        alphas = np.linspace(0, 1, num=50),
        alpha_builder = lambda _a : _a,
        seeds=(1, 2)
):
    env = build_env()
    steps_per_episode = np.zeros((len(alphas), ), dtype=np.float64)
    sum_of_rewards_per_episode = np.zeros((len(alphas), ), dtype=np.float64)
    eval_steps_per_episode = np.zeros((len(alphas), ), dtype=np.float64)
    eval_sum_of_rewards_per_episode = np.zeros((len(alphas), ), dtype=np.float64)

    state_size = env.unwrapped.observation_space.shape[0]
    action_size = int(env.unwrapped.action_space.n)

    # Using 4096 as the reference of total parameters used for the
    # linear approximation method.
    # total = state_size * h + h * action_size
    # => h = total / (state_size + action_size)
    h = 4096 // (state_size + action_size)

    for ia, alpha in enumerate(alphas):
        agent = Reinforce(
            state_size=state_size,
            action_space_dims=action_size,
            update_coefficient=alpha_builder(alpha),
            hidden_dims=(h, ),
            discount=0.99
        )

        res = run_env_episodic(
            env=env,
            behavioral_agent=agent,
            reward_shaper=reward_shaper,
            T=T,
            num_episodes=num_episodes,
            seeds=seeds,
            eval_num_episodes=1,
            do_eval=True,
            evaluate_frequency=1,
            greedy_eval=True
        )

        del agent

        steps = res['steps_per_episode']
        sum_rewards = res['sum_of_rewards_per_episode']
        eval_steps = res['eval_steps_per_episode']
        eval_sum_rewards = res['eval_sum_of_rewards_per_episode']

        # Average those n_experiments results to fill results_array
        steps_per_episode[ia] = np.mean(steps)
        sum_of_rewards_per_episode[ia] = np.mean(sum_rewards)
        eval_steps_per_episode[ia] = np.mean(eval_steps)
        eval_sum_of_rewards_per_episode[ia] = np.mean(eval_sum_rewards)

        print(
            f"\nReinforce: Done α={alpha:.3e}, "
            f" steps/episode: {steps_per_episode[ia]:.4f}, "
            f"sum(r)/episode: {sum_of_rewards_per_episode[ia]:.4f}"
            f" eval steps/episode: {eval_steps_per_episode[ia]:.4f}, "
            f"eval sum(r)/episode: {eval_sum_of_rewards_per_episode[ia]:.4f}"
        )


def reinforce_baseline(
        num_episodes,
        T,
        reward_shaper: Callable,
        alphas = np.linspace(0, 1, num=50),
        alpha_builder = lambda _a : _a,
        seeds=(1, 2),
        num_experiments=1
):
    env = build_env()
    steps_per_episode = np.zeros((len(alphas), ), dtype=np.float64)
    sum_of_rewards_per_episode = np.zeros((len(alphas), ), dtype=np.float64)
    eval_steps_per_episode = np.zeros((len(alphas), ), dtype=np.float64)
    eval_sum_of_rewards_per_episode = np.zeros((len(alphas), ), dtype=np.float64)

    state_size = env.unwrapped.observation_space.shape[0]
    action_size = int(env.unwrapped.action_space.n)

    # Using 4096 as the reference of total parameters used for the
    # linear approximation method.
    # total = state_size * h + h * action_size
    # => h = total / (state_size + action_size)
    h = 4096 // (state_size + action_size)

    seeds_per_experiment = [
        [s + e * len(seeds) for s in seeds]
        for e in range(num_experiments)
    ]

    total = len(alphas) * num_experiments
    count = 0
    for ia, alpha in enumerate(alphas):
        steps = []
        sum_rewards = []
        eval_steps = []
        eval_sum_rewards = []

        for exp in range(num_experiments):
            count += 1
            agent = ReinforceBaseline(
                state_size=state_size,
                action_space_dims=action_size,
                update_coefficient_policy=alpha_builder(alpha),
                update_coefficient_baseline=alpha_builder(1.2 * alpha),
                hidden_dims=(h, ),
                discount=0.99
            )

            res = run_env_episodic(
                env=env,
                behavioral_agent=agent,
                reward_shaper=reward_shaper,
                T=T,
                num_episodes=num_episodes,
                seeds=seeds_per_experiment[exp],
                eval_num_episodes=1,
                do_eval=True,
                evaluate_frequency=1,
                greedy_eval=True
            )

            del agent

            steps += res['steps_per_episode']
            sum_rewards += res['sum_of_rewards_per_episode']
            eval_steps += res['eval_steps_per_episode']
            eval_sum_rewards += res['eval_sum_of_rewards_per_episode']

        # Average those n_experiments results to fill results_array
        steps_per_episode[ia] = np.mean(steps)
        sum_of_rewards_per_episode[ia] = np.mean(sum_rewards)
        eval_steps_per_episode[ia] = np.mean(eval_steps)
        eval_sum_of_rewards_per_episode[ia] = np.mean(eval_sum_rewards)

        print(
            f"\nReinforce_Baseline: Completed = {100 * count / total:.0f}%"
            f" α={alpha:.3e}, "
            f" steps/episode: {steps_per_episode[ia]:.4f}, "
            f"sum(r)/episode: {sum_of_rewards_per_episode[ia]:.4f}"
            f" eval steps/episode: {eval_steps_per_episode[ia]:.4f}, "
            f"eval sum(r)/episode: {eval_sum_of_rewards_per_episode[ia]:.4f}"
        )


def one_step_ac(
        num_episodes,
        T,
        reward_shaper: Callable,
        alphas = np.linspace(0, 1, num=50),
        alpha_builder = lambda _a : _a,
        seeds=(1, 2),
        num_experiments=1
):
    env = build_env()
    steps_per_episode = np.zeros((len(alphas), ), dtype=np.float64)
    sum_of_rewards_per_episode = np.zeros((len(alphas), ), dtype=np.float64)
    eval_steps_per_episode = np.zeros((len(alphas), ), dtype=np.float64)
    eval_sum_of_rewards_per_episode = np.zeros((len(alphas), ), dtype=np.float64)

    state_size = env.unwrapped.observation_space.shape[0]
    action_size = int(env.unwrapped.action_space.n)

    # Using 4096 as the reference of total parameters used for the
    # linear approximation method.
    # total = state_size * h + h * action_size
    # => h = total / (state_size + action_size)
    h = 4096 // (state_size + action_size)

    seeds_per_experiment = [
        [s + e * len(seeds) for s in seeds]
        for e in range(num_experiments)
    ]

    total = len(alphas) * num_experiments
    count = 0
    for ia, alpha in enumerate(alphas):
        steps = []
        sum_rewards = []
        eval_steps = []
        eval_sum_rewards = []

        for exp in range(num_experiments):
            count += 1
            agent = OneStepAC(
                state_size=state_size,
                action_space_dims=action_size,
                update_coefficient_policy=alpha_builder(alpha),
                update_coefficient_critic=alpha_builder(1.2 * alpha),
                hidden_dims=(h, ),
                discount=0.99,
                norm_grad=False
            )

            res = run_env_episodic(
                env=env,
                behavioral_agent=agent,
                reward_shaper=reward_shaper,
                T=T,
                num_episodes=num_episodes,
                seeds=seeds_per_experiment[exp],
                eval_num_episodes=1,
                do_eval=True,
                evaluate_frequency=1,
                greedy_eval=True
            )

            del agent

            steps += res['steps_per_episode']
            sum_rewards += res['sum_of_rewards_per_episode']
            eval_steps += res['eval_steps_per_episode']
            eval_sum_rewards += res['eval_sum_of_rewards_per_episode']

        # Average those n_experiments results to fill results_array
        steps_per_episode[ia] = np.mean(steps)
        sum_of_rewards_per_episode[ia] = np.mean(sum_rewards)
        eval_steps_per_episode[ia] = np.mean(eval_steps)
        eval_sum_of_rewards_per_episode[ia] = np.mean(eval_sum_rewards)

        print(
            f"\nOne-Step AC: Completed = {100 * count / total:.0f}%"
            f" α={alpha:.3e}, "
            f" steps/episode: {steps_per_episode[ia]:.4f}, "
            f"sum(r)/episode: {sum_of_rewards_per_episode[ia]:.4f}"
            f" eval steps/episode: {eval_steps_per_episode[ia]:.4f}, "
            f"eval sum(r)/episode: {eval_sum_of_rewards_per_episode[ia]:.4f}"
        )


def ac_with_eligibility_traces(
        num_episodes,
        T,
        reward_shaper: Callable,
        alphas = np.linspace(0, 1, num=50),
        alpha_builder = lambda _a : _a,
        seeds=(1, 2),
        num_experiments=1
):
    env = build_env()
    steps_per_episode = np.zeros((len(alphas), ), dtype=np.float64)
    sum_of_rewards_per_episode = np.zeros((len(alphas), ), dtype=np.float64)
    eval_steps_per_episode = np.zeros((len(alphas), ), dtype=np.float64)
    eval_sum_of_rewards_per_episode = np.zeros((len(alphas), ), dtype=np.float64)

    state_size = env.unwrapped.observation_space.shape[0]
    action_size = int(env.unwrapped.action_space.n)

    # Using 4096 as the reference of total parameters used for the
    # linear approximation method.
    # total = state_size * h + h * action_size
    # => h = total / (state_size + action_size)
    h = 4096 // (state_size + action_size)

    seeds_per_experiment = [
        [s + e * len(seeds) for s in seeds]
        for e in range(num_experiments)
    ]

    total = len(alphas) * num_experiments
    count = 0
    for ia, alpha in enumerate(alphas):
        steps = []
        sum_rewards = []
        eval_steps = []
        eval_sum_rewards = []

        for exp in range(num_experiments):
            count += 1
            agent = ACWithEligibilityTraces(
                state_size=state_size,
                action_space_dims=action_size,
                update_coefficient_policy=alpha_builder(alpha),
                lam_policy=0.5,
                update_coefficient_critic=alpha_builder(25 * alpha),
                lam_critic=0.5,
                hidden_dims=(h, ),
                discount=0.99,
                norm_grad=False
            )

            res = run_env_episodic(
                env=env,
                behavioral_agent=agent,
                reward_shaper=reward_shaper,
                T=T,
                num_episodes=num_episodes,
                seeds=seeds_per_experiment[exp],
                eval_num_episodes=1,
                do_eval=True,
                evaluate_frequency=1,
                greedy_eval=True
            )

            del agent

            steps += res['steps_per_episode']
            sum_rewards += res['sum_of_rewards_per_episode']
            eval_steps += res['eval_steps_per_episode']
            eval_sum_rewards += res['eval_sum_of_rewards_per_episode']

        # Average those n_experiments results to fill results_array
        steps_per_episode[ia] = np.mean(steps)
        sum_of_rewards_per_episode[ia] = np.mean(sum_rewards)
        eval_steps_per_episode[ia] = np.mean(eval_steps)
        eval_sum_of_rewards_per_episode[ia] = np.mean(eval_sum_rewards)

        print(
            f"\nAC with ET: Completed = {100 * count / total:.0f}%"
            f" α={alpha:.3e}, "
            f" steps/episode: {steps_per_episode[ia]:.4f}, "
            f"sum(r)/episode: {sum_of_rewards_per_episode[ia]:.4f}"
            f" eval steps/episode: {eval_steps_per_episode[ia]:.4f}, "
            f"eval sum(r)/episode: {eval_sum_of_rewards_per_episode[ia]:.4f}"
        )


def ac_continuing_with_eligibility_traces(
        num_episodes,
        T,
        reward_shaper: Callable,
        alphas = np.linspace(0, 1, num=50),
        alpha_builder = lambda _a : _a,
        seeds=(1, 2),
        num_experiments=1
):
    env = build_env()
    steps_per_episode = np.zeros((len(alphas), ), dtype=np.float64)
    sum_of_rewards_per_episode = np.zeros((len(alphas), ), dtype=np.float64)
    eval_steps_per_episode = np.zeros((len(alphas), ), dtype=np.float64)
    eval_sum_of_rewards_per_episode = np.zeros((len(alphas), ), dtype=np.float64)

    state_size = env.unwrapped.observation_space.shape[0]
    action_size = int(env.unwrapped.action_space.n)

    # Using 4096 as the reference of total parameters used for the
    # linear approximation method.
    # total = state_size * h + h * action_size
    # => h = total / (state_size + action_size)
    h = 4096 // (state_size + action_size)

    seeds_per_experiment = [
        [s + e * len(seeds) for s in seeds]
        for e in range(num_experiments)
    ]

    total = len(alphas) * num_experiments
    count = 0
    for ia, alpha in enumerate(alphas):
        steps = []
        sum_rewards = []
        eval_steps = []
        eval_sum_rewards = []

        for exp in range(num_experiments):
            count += 1
            agent = ACWithEligibilityTracesContinuing(
                state_size=state_size,
                action_space_dims=action_size,
                update_coefficient_policy=alpha_builder(alpha),
                lam_policy=0.8,
                update_coefficient_critic=alpha_builder(alpha),
                update_coefficient_avg_reward=alpha_builder(alpha),
                lam_critic=0.8,
                hidden_dims=(h, ),
                norm_grad=False
            )

            res = run_env_episodic(
                env=env,
                behavioral_agent=agent,
                reward_shaper=reward_shaper,
                T=T,
                num_episodes=num_episodes,
                seeds=seeds_per_experiment[exp],
                eval_num_episodes=1,
                do_eval=False,
                evaluate_frequency=1,
                greedy_eval=True
            )

            del agent

            steps += res['steps_per_episode']
            sum_rewards += res['sum_of_rewards_per_episode']
            eval_steps += res['eval_steps_per_episode']
            eval_sum_rewards += res['eval_sum_of_rewards_per_episode']

        # Average those n_experiments results to fill results_array
        steps_per_episode[ia] = np.mean(steps)
        sum_of_rewards_per_episode[ia] = np.mean(sum_rewards)
        eval_steps_per_episode[ia] = np.mean(eval_steps)
        eval_sum_of_rewards_per_episode[ia] = np.mean(eval_sum_rewards)

        print(
            f"\nContinuing AC - ET: Completed = {100 * count / total:.0f}%"
            f" α={alpha:.3e}, "
            f" steps/episode: {steps_per_episode[ia]:.4f}, "
            f"sum(r)/episode: {sum_of_rewards_per_episode[ia]:.4f}"
            f" eval steps/episode: {eval_steps_per_episode[ia]:.4f}, "
            f"eval sum(r)/episode: {eval_sum_of_rewards_per_episode[ia]:.4f}"
        )


if __name__ == '__main__':
    do_log = False
    RENDER = False

    EPISODIC = False
    NORM_GRAD = False

    if EPISODIC:
        ENV_NAME = 'CartPole'
    else:
        ENV_NAME = 'MountainCar'

    if EPISODIC:
        num_experiments = 100
    else:
        num_experiments = 12

    num_episodes = None
    T = None
    if ENV_NAME == 'MountainCar':
        if EPISODIC:
            num_episodes = 150
            T = 999
        else:
            T = 5 * int(1e4)
            num_episodes = 1

        MAX_EPISODE_STEPS = T
    elif ENV_NAME == 'CartPole':
        if EPISODIC:
            num_episodes = 150
            T = 500
        else:
            T = 5 * int(1e5)
            num_episodes = 1

        MAX_EPISODE_STEPS = T
    elif ENV_NAME == 'LunarLander':
        if EPISODIC:
            num_episodes = 100
            T = 1000
        else:
            T = int(1e5) # 40000
            num_episodes = 1

        MAX_EPISODE_STEPS = T
    else:
        raise NotImplementedError(f"{ENV_NAME} not implemented")

    def build_temp_sched(start,  steps=max(1, num_episodes * (T // 3))):
        return LinearSchedule(start, end=1.0, steps=steps)

    def build_alpha_sched(start, steps=max(1, num_episodes * (T // 3))):
        return LinearSchedule(start, end=0.001 * start, steps=steps)

    def base_reward(reward: float, state: np.ndarray, done: bool, t: int):
        return reward

    def mountaincar_continuous_reward(reward: float, state: np.ndarray, done: bool, t: int):
        pos = state[0]
        bonus = (1 - abs(pos - 0.5) / 1.8)
        if bonus > 0.9:
            bonus *= 10
        return reward + bonus

    alphas = np.linspace(1e-6, 1e-2, num=5)

    # ac_continuing_with_eligibility_traces(
    #     num_episodes=num_episodes,
    #     T=T,
    #     reward_shaper=mountaincar_reward,
    #     alphas=alphas,
    #     alpha_builder=build_alpha_sched,
    #     num_experiments=num_experiments,
    #     seeds=[i for i in range(num_episodes)]
    # )
    #
    # exit(0)

    if EPISODIC:
        for model in ['Reinforce', 'ReinforceBaseline', 'OneStepAC', 'ACWithEligibilityTraces']:
            experiments_parallel(
                model=model,
                num_episodes=num_episodes,
                T=T,
                reward_shaper=base_reward,
                alphas=alphas,
                alpha_builder=build_alpha_sched,
                num_experiments=num_experiments,
                seeds=[i for i in range(num_episodes)]
            )
    else:
        for model in ['ACWithEligibilityTracesContinuing']:
            experiments_parallel(
                model=model,
                num_episodes=num_episodes,
                T=T,
                reward_shaper=mountaincar_continuous_reward,
                alphas=alphas,
                alpha_builder=build_alpha_sched,
                num_experiments=num_experiments,
                seeds=[i for i in range(num_episodes)]
            )

    exit(0)