import os.path
from typing import Callable
import pygame
import numpy as np
import pickle
from scipy.stats import randint

from planning.agents import TabularDynaQAgent
from envMaze import ExtendedMazeEnvironment

from utils import Experience, LinearEpsSchedule


##############################################


def create_new_maze():
    env = ExtendedMazeEnvironment(
        save_maze=True,
        load_maze=False
    )

    _ = env.reset(randomize_start_goal=True)

    # display maze and wait for user SPACE for confirmation
    while not env.space_clicked:
        env.handle_event()
        env.drawGrid()
        env.showChar()
        pygame.display.update()

    env.save_maze()


def get_environment(
        randomize_start_goal=False,
        skip_visual_confirmation: bool = True):
    """
        skip_visual_confirmation: no user confirmation

        Returns: environment, s0
    """

    # Setup the environment dynamically. Spacebar to confirm environment
    env = ExtendedMazeEnvironment(
        save_maze=False,
        load_maze=True
    )

    env.load_maze()
    s0 = env.reset(randomize_start_goal=randomize_start_goal)

    # display maze and wait for user SPACE for confirmation
    while (not env.space_clicked) and (not skip_visual_confirmation):
        env.handle_event()
        env.drawGrid()
        env.showChar()
        pygame.display.update()

    return env, s0


def run(
        model_steps,
        maxEpisodes=200,
        max_steps_per_episode=0,
        randomize_start_goal=False
):
    """
        n: num simulation steps
        save_load:  True on fresh maze setup
    """

    def build_greedy_eps_sched(start):
        """
            Sarsa requires pi --> greedy as one of the conditions for
            convergence.
        :param start:
        :return:
        """
        return LinearEpsSchedule(
            start,
            end=0.0,
            steps=(maxEpisodes // 4) * max(max_steps_per_episode, 100)
        )

    env, s = get_environment(skip_visual_confirmation=False)
    n_actions = env.num_actions
    n_states = env.num_states

    agent = TabularDynaQAgent(
        action_space_dims=n_actions,
        obs_space_dims=n_states,
        update_coefficient=0.2,
        model_steps=model_steps,
        eps=build_greedy_eps_sched(0.2)
    )

    agent.initialize()

    for episode in range(maxEpisodes):
        # Redundant, but environment needs to be reset
        env, s = get_environment(
            randomize_start_goal=randomize_start_goal,
            skip_visual_confirmation=True)

        agent.reset()

        terminal = False
        while (not terminal and
               (
                       (max_steps_per_episode <= 0) or
                       (env._num_ep_steps < max_steps_per_episode)
               ) and not env.close_clicked
        ):

            env.handle_event()
            env.drawGrid(state_val_fn=agent.state_value)
            env.showChar()

            a, p = agent.act(s)
            r, sp, terminal = env.step(a)

            env.drawBlackBox(sp)
            pygame.display.update()

            agent.step(
                Experience(
                    s=s,
                    a=a,
                    r=r,
                    sp=sp,
                    p=p,
                    done=int(terminal)
                )
            )

            if not terminal:
                s = sp
            else:
                print(f'Episode: {episode}, steps taken: {env.num_steps()}')


def run_dynaq(do_create_new_maze: bool = False):
    model_steps = 5  # model planning steps

    if do_create_new_maze:
        create_new_maze()
    run(model_steps,
        max_steps_per_episode=100,
        randomize_start_goal=False)


if __name__ == '__main__':
    run_dynaq(do_create_new_maze=False)

    exit(0)
