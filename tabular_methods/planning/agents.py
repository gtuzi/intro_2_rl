import math
from typing import Union, Optional, List, Tuple, Iterable, Any
from collections import defaultdict
import numpy as np
from tabular_methods.utils import Experience, NoiseSchedule, QEpsGreedyAgent


class TabularDynaQAgent(QEpsGreedyAgent):
    """
        Implementation of Tabular Dyna-Q in Section 8.2
        The method in the book uses QLearning / SarsaMax.

        Authors claim that other one-step sample updates can also be used
        (Section 8.5)

        Added support for the Direct RL path:
         - Sarsa
         - ExpectedSarsa
    """

    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            update_coefficient: float,
            discount: float = 0.95,
            eps: Union[float, NoiseSchedule] = 0.1,
            qval_init: float = 0.,
            model_steps: int = 0,
            dynaq_plus_k: Optional[float] = None,  # Dyna-Q+ exploration bonus reward coefficient
            td_update_type: str = 'qlearning'
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)
        assert 0. < update_coefficient < 1.

        if dynaq_plus_k is not None:
            assert dynaq_plus_k >= 0.

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
        self.state_action_visit_count: Optional[defaultdict] = None
        self.model_steps = model_steps
        self.dynaq_plus_k = dynaq_plus_k

        self.td_update_type = td_update_type
        self.requires_next_step_before_update = False
        self._agent_requires_next_step()

    def _agent_requires_next_step(self):
        if self.td_update_type.lower() == 'qlearning':
            self.requires_next_step_before_update = False
        elif self.td_update_type.lower() == 'sarsa':
            self.requires_next_step_before_update = True
        elif self.td_update_type.lower() == 'expected_sarsa':
            self.requires_next_step_before_update = False
        else:
            raise NotImplementedError(f'{self.td_update_type} not supported')

    def initialize(self):
        self.t = 0

        if isinstance(self.eps, NoiseSchedule):
            # Reset noise to starting exploration
            self.eps.initialize()

        # Initialize Q[si][aj] = qval_init
        self.Q = defaultdict(lambda: [self.qval_init] * self.action_space_dims)
        self.Q_update_count = defaultdict(lambda: [0] * self.action_space_dims)

        # Initialize model[si][aj] where None indicates an unvisited pair
        self.model = defaultdict(
            lambda: [(None, None, None)] * self.action_space_dims)

        if self.dynaq_plus_k is not None:
            self.state_action_visit_count = defaultdict(
                lambda: [0] * self.action_space_dims)

    def step(self, experience: Experience, **kwargs):
        self.direct_rl_learn(experience)
        self.model_learn(experience)
        self.planning()

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

        self.t += 1

    def direct_rl_learn(self, experience: Experience):
        self._td_update(experience)

    def model_learn(self, experience: Experience, **kwargs):
        s, a, r, sp, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.done
        )

        # deterministic model
        self.model[s][a] = (sp, r, done)

        if self.state_action_visit_count is not None:
            self.state_action_visit_count[s][a] = self.t

    def planning(self, **kwargs):
        for step in range(self.model_steps):
            ###### Search control ######
            """
                search control to refer to the process that selects the 
                starting states and actions for the simulated experiences 
                generated by the model
            """
            # List of states that have been encountered
            states = list(self.model.keys())

            # Random sample from the already encountered states.
            s = states[int(np.random.randint(len(states), size=1))]

            # Obtain actions that have been taken for this state
            visited_actions = [
                _a for _a in range(self.action_space_dims)
                if self.model[s][_a] != (None, None, None)
            ]

            if self.state_action_visit_count is None:
                # We're not tracking the (state,action) visitation counts
                # Random sample from already taken actions for this state
                a = visited_actions[
                    int(np.random.randint(len(visited_actions), size=1))
                ]

                sp, r, done = self.model[s][a]
            else:
                # In Section 8.3 (page 168) footnote:
                # 1 - actions that had never been tried before from a state
                #     were allowed to be considered in the planning step
                # 2 - the initial model for such actions was that they would
                #     lead back to the same state with a reward of zero

                # Based on (1) random sample from ALL actions -
                # not just the visited actions for this state.
                a = int(np.random.randint(self.action_space_dims, size=1))

                # If the selected action has been taken before, use the
                # model (next-state, reward, terminal)
                if a in visited_actions:
                    sp, r, done = self.model[s][a]
                else:
                    # Otherwise, per (2), reward is zero and next step is the
                    # starting step
                    sp, r, done = s, 0, 0

            if self.state_action_visit_count is not None:
                # Encourage exploration for stale state-actions
                # For actions that have not been visited, r = 0, so
                # the reward is purely time
                dt = self.t - self.state_action_visit_count[s][a]
                r += self.dynaq_plus_k * math.sqrt(dt)

            # Algos, such as sarsa, require the next step the policy
            # would have taken.
            ap, pp = None, None
            if self.requires_next_step_before_update:
                ap, pp = self.act(sp)

            experience = Experience(
                s=s,
                a=a,
                r=r,
                sp=sp,
                ap=ap,
                pp=pp,
                done=done
            )

            ###### Planning Update ######
            self.direct_rl_learn(experience)

    def _td_update(self, experience: Experience, **kwargs):
        if self.td_update_type == 'qlearning':
            self._q_learning(experience)
        elif self.td_update_type == 'sarsa':
            self._sarsa(experience)
        elif self.td_update_type == 'expected_sarsa':
            self._expected_sarsa(experience)
        else:
            raise NotImplementedError

    def _q_learning(self,  experience: Experience, **kwargs):
        s, a, r, sp, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.done
        )

        # Directly estimate q* (Q-learning / SarsaMax)
        tgt = r + self.discount * max(self.Q[sp]) * (1 - done)
        td_error = tgt - self.Q[s][a]

        self.Q[s][a] += self.update_coefficient * td_error
        self.Q_update_count[s][a] += 1

    def _sarsa(self, experience: Experience, **kwargs):
        s, a, r, sp, ap, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap, experience.done
        )

        tgt = r + self.discount * self.Q[sp][ap] * (1 - done)
        td_error = tgt - self.Q[s][a]

        self.Q[s][a] += self.update_coefficient * td_error
        self.Q_update_count[s][a] += 1

    def _expected_sarsa(self,  experience: Experience, **kwargs):
        s, a, r, sp, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.done
        )

        qp_expected = sum(
            [
                self.get_sa_probability(s=sp, a=_a) * self.Q[sp][_a]
                for _a in range(self.action_space_dims)
            ]
        )

        tgt = r + self.discount * qp_expected * (1 - done)

        td_error = tgt - self.Q[s][a]

        self.Q[s][a] += self.update_coefficient * td_error
        self.Q_update_count[s][a] += 1

    def reset(self):
            # The agent here is prepared for a new episode
            self.t = 0

            # Reset the visitation counts
            if self.dynaq_plus_k is not None:
                self.state_action_visit_count = defaultdict(
                    lambda: [0] * self.action_space_dims)

            if isinstance(self.eps, NoiseSchedule):
                self.eps.reset()


class PrioritizedQueue:
    def __init__(self):
        self.priorities: List[float] = []
        self.state_actions: List[Any] = []

    def insert(self, sa: Any, priority: float):
        """
            Insert <v> according to its <priority> value.
            Priorities are descending
        """
        if len(self.priorities) == 0:
            self.priorities.append(priority)
            self.state_actions.append(sa)
        else:
            # Remove existing pairs
            if sa in self.state_actions:
                idc_2_remove = [i for i, val in enumerate(self.state_actions) if val == sa]
                _ = [self.state_actions.pop(i) for i in idc_2_remove]
                _ = [self.priorities.pop(i) for i in idc_2_remove]

            # Priorities are kept descending in value
            idx = next((i for i, p in enumerate(self.priorities) if p < priority), None)

            # If all the existing priorities are greater,
            # insert at the end of queue
            if idx is None:
                idx = len(self.priorities)

            self.priorities.insert(idx, priority)
            self.state_actions.insert(idx, sa)

        assert len(self.priorities) == len(self.state_actions)

    def pop_first(self) -> Tuple[float, Any]:
        """ Pop the pop_first in queue """
        return self.priorities.pop(0), self.state_actions.pop(0)

    def is_empty(self) -> bool:
        return len(self.priorities) == 0

    def __len__(self) -> int:
        return len(self.priorities)

    def clear(self):
        self.priorities.clear()
        self.state_actions.clear()


class DeterministicTabularModel:
    def __init__(self, action_space_dims: int):
        self.model: Optional[defaultdict] = None
        self.state_action_visit_count: Optional[defaultdict] = None
        self.action_space_dims = action_space_dims

        # Tracks s' <- (s, a)
        self.inverse_state_transitions: Optional[defaultdict] = None

    def initialize(self):
        # Initialize model[si][aj] where None indicates an unvisited pair
        self.model = defaultdict(
            lambda: [(None, None, None)] * self.action_space_dims)

        # Model tracks the visit counts
        self.state_action_visit_count = defaultdict(
            lambda: [0] * self.action_space_dims)

        # Assuming deterministic environment: sp -> [.., (s, a), ...]
        self.inverse_state_transitions = defaultdict(lambda: set())

    def reset(self):
        self.state_action_visit_count = defaultdict(
            lambda: [0] * self.action_space_dims)

    def learn(self, experience: Experience, t: int):
        s, a, r, sp, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.done
        )

        # deterministic model
        self.model[s][a] = (sp, r, done)
        self.state_action_visit_count[s][a] = t

        # Track (s,a) that lead to sp
        self.inverse_state_transitions[sp].add((s, a))

    def get_sa_list_that_lead_to_s(self, s) -> Iterable[Tuple[int, int]]:
        """ Obtain the (s*, a*) list that lead to s """
        return list(self.inverse_state_transitions[s])

    def get_nex_state_reward_terminal(self, s, a) -> Tuple[Any, float, int]:
        return self.model[s][a]


class TabularPrioritizedSweepingAgent(QEpsGreedyAgent):
    """
        Implementation of "Prioritized sweeping for a deterministic environment"
        in Section 8.4
    """
    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            update_coefficient: float,
            discount: float = 0.95,
            eps: Union[float, NoiseSchedule] = 0.1,
            qval_init: float = 0.,
            model_steps: int = 0,
            priority_threshold: float = 0.1  # theta
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
        self.model = DeterministicTabularModel(self.action_space_dims)
        self.model_steps = model_steps
        self.priority_threshold = priority_threshold
        self.queue: Optional[PrioritizedQueue] = None
        self.requires_next_step_before_update = False

    def initialize(self):
        self.t = 0

        if isinstance(self.eps, NoiseSchedule):
            # Reset noise to starting exploration
            self.eps.initialize()

        # Initialize Q[si][aj] = qval_init
        self.Q = defaultdict(lambda: [self.qval_init] * self.action_space_dims)
        self.Q_update_count = defaultdict(lambda: [0] * self.action_space_dims)

        # Initialize model[si][aj] where None indicates an unvisited pair
        self.model.initialize()
        self.queue = PrioritizedQueue()

    def step(self, experience: Experience, **kwargs):
        assert isinstance(experience.done, (int, bool))

        self.model_learn(experience)

        # TODO: clear ??
        # self.queue.clear()

        self.planning(experience)

        if isinstance(self.eps, NoiseSchedule):
            self.eps.step()

        self.t += 1

    def planning(self, experience: Experience):
        """ Prioritized sweeping """
        s, a, r, sp, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.done
        )

        priority = abs(self._td_error(s, a, r, sp, done))

        if priority > self.priority_threshold:
            self.queue.insert((s, a), priority)

        simu_step = 0

        while (simu_step < self.model_steps) and (not self.queue.is_empty()):
            _, (s, a) = self.queue.pop_first()
            sp, r, done = self.model.get_nex_state_reward_terminal(s, a)

            self._td_update(
                Experience(
                    s=s,
                    a=a,
                    r=r,
                    sp=sp,
                    done=done
                )
            )

            # Loop for all S ̄, A ̄ predicted to lead to S
            sa_bars = self.model.get_sa_list_that_lead_to_s(s)

            for sb, ab in sa_bars:
                # R ̄ predicted reward for S ̄, A ̄, S
                # For the deterministic model case, just poll the model.
                # This is not specified in the algorithm of the book, but
                # given that in the following paragraph, the author mentions
                # extensions for a "stochastic" environment, it means that
                # for the deterministic environment we've already captured
                # the reward = model[S ̄][A ̄] (or 0 if no experience yet).
                # If stochastic model, we need to estimate the expectation
                # of the reward: E[r] = some_predictor(S ̄, A ̄, S)
                spb, rb, doneb = self.model.get_nex_state_reward_terminal(s=sb, a=ab)

                assert spb == s, 'Deterministic model not so deterministic'
                assert doneb == 0

                # This is the backward step, this cannot be terminal
                priority = abs(self._td_error(sb, ab, rb, s, doneb))
                if priority > self.priority_threshold:
                    # Learn backwards the state-actions that lead to this state
                    self.queue.insert((sb, ab), priority)

            simu_step += 1

        # if simu_step > 0:
        #     print('Simu Step: ', simu_step, 'Queue Size: ', len(self.queue))

    def _td_update(self, experience: Experience, **kwargs):
        s, a, r, sp, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.done
        )

        # Directly estimate q* (Q-learning / SarsaMax)
        td_error = self._td_error(s, a, r, sp, done)
        self.Q[s][a] += self.update_coefficient * td_error
        self.Q_update_count[s][a] += 1

    def _td_error(self, s, a, r, sp, done):
        # Q-learning td-error
        tgt = r + self.discount * max(self.Q[sp]) * (1 - done)
        td_error = tgt - self.Q[s][a]
        return td_error

    def model_learn(self, experience: Experience, **kwargs):
        self.model.learn(experience, self.t)

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0
        self.model.reset()

        if isinstance(self.eps, NoiseSchedule):
            self.eps.reset()

