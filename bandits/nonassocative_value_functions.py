from abc import ABC, abstractmethod
from typing import List, Union, Optional

from tools.coefficients import Coefficient, ConstantCoefficient
from tools.moving_averages import MovingAverage, CummulativeMovingAverage, ExponentialMovingAverage


class Q(ABC):
    def __init__(self):
        self._action_values = []
        self._action_visit_counts = []
        self._t = None

    # Action values keeps the reward value of an action
    @property
    def action_values(self) -> List[MovingAverage]:
        return self._action_values

    @action_values.setter
    def action_values(self, values: List[MovingAverage]):
        assert len([v for v in values if isinstance(v, MovingAverage)]) > 0, 'action_values must be MovingAverage type'
        self._action_values = values

    # Action visit counts the times an action has been taken
    @property
    def action_visit_counts(self) -> List[int]:
        return self._action_visit_counts

    @action_visit_counts.setter
    def action_visit_counts(self, values: List[int]):
        assert len([v for v in values if isinstance(v, int)]) > 0, 'action_visit_counts must be int type'
        self._action_visit_counts = values

    def __call__(self, a: Optional[int] = None) -> Union[float, List[float]]:
        """ [.., Q(ai, N), ... ]: the value of one or all actions """

        if a is not None:  # Value of action at this step
            assert 0 <= a < len(self), 'Action is out of bounds'
            return self.action_values[a].current_value
        else:  # Values of all actions at this step
            return [a_val.current_value for a_val in self.action_values]

    def __len__(self):
        return len(self.action_values)

    @abstractmethod
    def step(self, step: int, action: int, reward: float):
        """ Q(a, N + 1) = f(Q(a, N), r, N): Updates the action-value at step """
        raise NotImplementedError


class QMonteCarlo(Q):
    def __init__(self, n_actions: int, initial_action_value: Union[float, List] = 0.0):
        super().__init__()
        if not isinstance(initial_action_value, List):
            initial_action_value = [initial_action_value] * n_actions
        self.action_values = [CummulativeMovingAverage(initial_value=iv) for iv in initial_action_value]
        self.action_visit_counts = [0 for _ in range(n_actions)]

    def step(self, step: int, action: int, reward: float):
        """ Update the value of action """
        self._t = step
        self.action_visit_counts[action] += 1
        _ = self.action_values[action].step(target=reward)


class QCoefficientMovingAverage(Q):
    def __init__(self,
                 n_actions: int,
                 coefficient: Coefficient = ConstantCoefficient(0.1),
                 initial_action_value: float = 0.0):
        super().__init__()
        self.action_values = [
            ExponentialMovingAverage(initial_value=initial_action_value, coefficient=coefficient)
            for _ in range(n_actions)
        ]
        self.action_visit_counts = [0 for _ in range(n_actions)]

    def step(self, step: int, action: int, reward: float):
        """ Update the value of action """
        self.action_visit_counts[action] += 1
        _ = self.action_values[action].step(target=reward)

