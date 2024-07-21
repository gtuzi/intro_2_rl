"""
    Extended from:
    https://github.com/konantian/Dyna-Maze-Game/blob/master/Codes/envMaze.py
"""

"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018,
  University of Alberta.
  Gambler's problem environment using RLGlue.
"""

from rl_glue import BaseEnvironment


class MazeEnvironment(BaseEnvironment):

    def __init__(self):
        """Declare environment variables."""

    def env_init(self, maze):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize environment variables necessary for run.
        """
        self.state = None
        self.terminal = None
        self.update_wall(maze)

    def env_start(self, maze, start, goal):
        """
        Arguments: Nothing
        Returns: state - numpy array
        Hint: Sample the starting state necessary for exploring starts and return.
        """

        self.state = start
        self.terminal = goal
        return self.state

    def env_step(self, action):
        """
        Arguments: action - integer
        Returns: reward - float, state - numpy array - terminal - boolean
        Hint: Take a step in the environment based on dynamics; also checking for action validity in
        state may help handle any rogue agents.
        """
        #calculate next state based on the action taken
        testState = tuple(map(sum, zip(self.state, action)))
        x, y = testState

        if testState not in self.wall and 0 <= x <= 8 and 0 <= y <= 5:

            self.state = testState

            if self.state == self.terminal:
                return 1, self.state, True

        return 0, self.state, False

    def env_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: response based on in_message
        This function is complete. You do not need to add code here.
        """
        if in_message == "return":
            return self.state

    def update_wall(self, maze):
        self.wall = set([])
        for row in range(len(maze)):
            for col in range(len(maze[0])):
                if maze[row][col] == 1:
                    self.wall.add((row, col))

    def update_start_gola(self, start, goal):
        self.state = start
        self.terminal = goal


######### Extension of Environment ########

import os
from typing import Callable
import pygame
import pickle
import numpy as np

class ExtendedMazeEnvironment():
    def __init__(self, save_maze: bool = False, load_maze: bool = False):
        self._environment = MazeEnvironment()
        self.actions = ["left", "right", "up", "down"]
        self.coords = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self.returns = dict(zip(self.actions, self.coords))
        self._save_maze = save_maze
        self._load_maze = load_maze

        # useful statistics
        self._total_reward = None
        self._num_steps = None  # number of steps in entire experiment
        self._num_episodes = None  # number of episodes in entire experiment
        self._num_ep_steps = None  # number of steps in this episode

        #################attributes of Game
        self.surface = self.create_window()
        self.bg_color = pygame.Color('black')
        self.pause_time = 0.04
        self.close_clicked = False
        self.space_clicked = False
        self.continue_game = True
        self.normal_color = pygame.Color('white')
        self.wall_color = pygame.Color('gray')
        self.w = 60
        self.margin = 1
        self.maze = [[0] * 6 for n in range(9)]

        color_tags = [hex(0xffffff), hex(0xedffed), hex(0xdbffdb),
                      hex(0xc9ffc9), hex(0xb7ffb7), hex(0xa5ffa5),
                      hex(0x93ff93), hex(0x81ff81), hex(0x6fff6f),
                      hex(0x5dff5d)]
        self.color = dict(zip(list(range(10)), color_tags))

    def __del__(self):
        if self._save_maze:
            self.save_maze()

    ###############
    def create_window(self):
        title = "Dyna Maze"
        size = (550, 370)
        pygame.init()
        surface = pygame.display.set_mode(size, 0, 0)
        pygame.display.set_caption(title)
        return surface

    def handle_event(self):
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            self.close_clicked = True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            y = pos[0] // (self.w + self.margin)
            x = pos[1] // (self.w + self.margin)
            if self.maze[y][x] == 0:
                self.maze[y][x] = 1
            elif self.maze[y][x] == 1:
                self.maze[y][x] = 0
            self._environment.update_wall(self.maze)
        elif event.type == pygame.KEYUP:  # spacebar for macs ?
            self.space_clicked = True
        elif event.type == pygame.WINDOWCLOSE:
            exit(0)

    def drawGrid(self, state_val_fn: Callable = lambda s: 0):
        self.surface.fill(self.bg_color)
        for row in range(9):
            for col in range(6):
                grid = [(self.margin + self.w) * row + self.margin,
                        (self.margin + self.w) * col + self.margin, self.w,
                        self.w]
                if self.maze[row][col] == 1:
                    pygame.draw.rect(self.surface, self.wall_color, grid)
                else:
                    # value = (self._agent.calValue((row, col)) * 100) // 10
                    value = (state_val_fn((row, col)) * 100) // 10

                    # Clip to 9
                    value = min(value, 9)

                    pygame.draw.rect(
                        self.surface,
                        pygame.Color(self.color[value]),
                        grid
                    )

    def showChar(self):
        x_offset = 20
        y_offset = 5

        pygame.font.init()
        myfont = pygame.font.SysFont('Comic Sans MS', 35)
        start = myfont.render('S', False, (0, 0, 0))
        self.surface.blit(
            start,
            (self.start[0] * (self.w + self.margin) + x_offset,
             self.start[1] * (self.w + self.margin) + y_offset)
        )

        goal = myfont.render('G', False, (0, 0, 0))
        self.surface.blit(
            goal,
            (self.goal[0] * (self.w + self.margin) + x_offset,
             self.goal[1] * (self.w + self.margin) + y_offset)
        )

    def drawBlackBox(self, pos):
        x, y = pos
        grid = [(self.margin + self.w) * x + self.margin,
                (self.margin + self.w) * y + self.margin, self.w, self.w]
        pygame.draw.rect(self.surface, self.bg_color, grid)

    ################
    def save_maze(self):
        data = dict(maze=self.maze, start=self.start, goal=self.goal)
        with open('maze_config.pkl', 'wb') as f:
            pickle.dump(data, f)
            print('Maze saved to maze_config.pkl')

    def load_maze(self):
        if os.path.isfile('maze_config.pkl'):
            with open('maze_config.pkl', 'rb') as f:
                data = pickle.load(f)
                print('Maze loaded from maze_config.pkl')
                self.maze = data['maze']
                self.start = data['start']
                self.goal = data['goal']
        else:
            print('Could not load maze_config.pkl')

    ###############
    def total_reward(self):
        return self._total_reward

    def num_steps(self):
        return self._num_steps

    def num_episodes(self):
        return self._num_episodes

    def num_ep_steps(self):
        return self._num_ep_steps

    ###############

    def reset(self,  randomize_start_goal: bool = True, seed=None) -> int:
        # reset statistics
        self._total_reward = 0
        self._num_steps = 0
        self._num_episodes = 0
        self._num_ep_steps = 0

        if self._load_maze:
            self.load_maze()

        st0 = None
        if seed is not None:
            st0 = np.random.get_state()
            np.random.seed(seed)

        # Generate start/goal such that they are not on the blocks
        while randomize_start_goal:
            self.start = (
                np.random.randint(0, 9),
                np.random.randint(0, 6)
            )

            if self.maze[self.start[0]][self.start[1]] != 1:
                break

        while randomize_start_goal:
            self.goal = (
                np.random.randint(0, 9),
                np.random.randint(0, 6)
            )

            if self.maze[self.goal[0]][self.goal[1]] != 1:
                break

        self._environment.env_init(self.maze)

        if seed is not None:
            np.random.set_state(st0)

        # A state is the available free cell
        self.num_states = sum(
            [0 if cell == 1 else 1 for cols in self.maze for cell in cols]
        )
        self.num_actions = len(self.actions)

        self._num_ep_steps = 1
        # state is a tuple
        state = self._environment.env_start(self.maze, self.start, self.goal)
        return state

    def step(self, action):
        reward, state, terminal = self._environment.env_step(self.coords[action])

        self._total_reward += reward

        if terminal:
            self._num_episodes += 1
        else:
            self._num_ep_steps += 1
            self._num_steps += 1

        return reward, state, terminal
