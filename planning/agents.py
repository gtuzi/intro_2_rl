from typing import Union, Optional
from collections import defaultdict
import numpy as np
from utils import Experience, NoiseSchedule, QEpsGreedyAgent


class TabularDynaQAgent(QEpsGreedyAgent):

    """
        Implementation of Tabular Dyna-Q in Section 8.2
    """

    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            update_coefficient: float,
            discount: float = 0.9,
            eps: Union[float, NoiseSchedule] = 0.01,
            qval_init: float = 0.,
            model_steps: int = 0
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)
        assert 0. < update_coefficient < 1.

        if not isinstance(eps, NoiseSchedule):
            assert 0 <= eps <= 1

        super().__init__(
            obs_space_dims=obs_space_dims,
            action_space_dims=action_space_dims,
            discount=discount,
            eps=eps
        )

        self.t = 0
        self.update_coefficient = update_coefficient
        self.qval_init = qval_init
        self.model: Optional[defaultdict] = None
        self.model_steps = model_steps

    def initialize(self):
        self.t = 0

        if isinstance(self.eps, NoiseSchedule):
            # Reset noise to starting exploration
            self.eps.initialize()

        # Initialize Q[si][aj] = qval_init
        self.Q = defaultdict(lambda: [self.qval_init] * self.action_space_dims)
        self.Q_update_count = defaultdict(lambda: [0] * self.action_space_dims)

        # Initialize model[si][aj] where None indicates an unvisited pair
        self.model = defaultdict(lambda: [(None, None)] * self.action_space_dims)

    def step(self, experience: Experience, **kwargs):
        self.direct_rl_learn(experience)
        self.model_learn(experience)
        self.planning()

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

    def direct_rl_learn(self, experience: Experience):
        self._td_update(experience)

    def model_learn(self, experience: Experience):
        s, a, r, sp, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.done
        )

        # deterministic model
        self.model[s][a] = (sp, r)

    def planning(self):
        for step in range(self.model_steps):
            # List of states that have been encountered
            states = list(self.model.keys())

            # Random sample from the already encountered
            s = states[int(np.random.randint(len(states), size=1))]

            # Obtain actions that have been taken for this state
            actions = [
                _a for _a in range(self.action_space_dims)
                if self.model[s][_a] != (None, None)
            ]

            # Random sample from the available actions
            a = actions[int(np.random.randint(len(actions), size=1))]

            sp, r = self.model[s][a]

            experience = Experience(
                s=s,
                a=a,
                r=r,
                sp=sp,
                done=0
            )

            self.direct_rl_learn(experience)

    def _td_update(self, experience: Experience, **kwargs):
        s, a, r, sp, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.done
        )

        # Directly estimate q*
        tgt = r + self.discount * max(self.Q[sp]) * (1 - done)
        td_error = tgt - self.Q[s][a]

        self.Q[s][a] += self.update_coefficient * td_error
        self.Q_update_count[s][a] += 1

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0
        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()