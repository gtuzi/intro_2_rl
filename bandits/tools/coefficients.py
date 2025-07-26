import math
import numpy as np
from abc import ABC, abstractmethod


class Coefficient(ABC):
    '''
        Family of objects which generates a single value, and depends only on step() call count.
    '''

    def __init__(self, base_value: float):
        self.base_value = base_value
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def step(self, **kwargs) -> float:
        """ Update coefficient and return new value"""
        self._call_count += 1
        self._update(**kwargs)
        return self.current_value

    @abstractmethod
    def _update(self, **kwargs):
        """ Update state parameters """
        raise NotImplementedError

    @property
    @abstractmethod
    def current_value(self) -> float:
        raise NotImplementedError


class ConstantCoefficient(Coefficient):
    """ Constant coefficient callable """

    def __init__(self, base_value: float):
        super(ConstantCoefficient, self).__init__(base_value=base_value)

    def _update(self, **kwargs):
        ''' Constant coefficient. No state value to update '''
        pass

    @property
    def current_value(self) -> float:
        return self.base_value


class EarlyWeightedCoefficient(Coefficient):
    """
        A coefficient used to reduce the influence of the initial value (bias) in the early steps of a moving average.

        Use case (background):
        The debiasing coefficient weighs current samples values more than the moving average - which is initially more biased towards the
        initial value. With time, the moving average gets weighted more as the debiasing coefficient
        y(t+1) = y(t) + b(t)(target - y(t))

        This implementation
        b(t+1) = b(t) + alpha * (1 - b(t))
        Debiasing coefficient starts large reaches base value over time.
        Value returned will be clipped in [min, max] range.

        Ref: Sutton RL book, 2d ed:  (2.9)
    """

    def __init__(self, base_value: float, min_val: float = 0.0, max_val: float = 1.0):
        super(EarlyWeightedCoefficient, self).__init__(base_value=base_value)
        assert min_val <= base_value <= max_val, 'Base value must be between min/max values'
        self.base_value = base_value
        self.moving_coefficient = 0.0
        self.alpha = 0.1
        self.min_val = min_val
        self.max_val = max_val

    def _update(self, **kwargs):
        self.moving_coefficient += self.alpha * (1 - self.moving_coefficient)

    @property
    def current_value(self) -> float:
        """ Value returned will be clipped in [min, max] range. """
        return float(
            np.clip(self.base_value / (self.moving_coefficient + 1e-7), a_min=self.min_val, a_max=self.max_val))


class CosineDecayCoefficient(Coefficient):
    def __init__(self, base_value: float, decay_steps: int, final_value: float = 1e-8, alpha: float = 0.0):
        super(CosineDecayCoefficient, self).__init__(base_value=base_value)
        assert decay_steps > 0
        self.decay_steps = int(decay_steps)
        self.alpha = alpha
        self.final_value = final_value
        self.moving_coefficient = base_value

    def _update(self, **kwargs):
        step = min(self._call_count, self.decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        self.moving_coefficient = self.base_value * decayed + (1 - decayed) * self.final_value

    @property
    def current_value(self) -> float:
        return self.moving_coefficient


if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt

    n_steps = 100
    cc = CosineDecayCoefficient(2.0, n_steps, final_value=0.7)

    samples = [cc.step() for i in range(n_steps)]

    plt.plot(samples)
    plt.xlabel('Steps')
    plt.grid()
    plt.show()
    exit(0)
