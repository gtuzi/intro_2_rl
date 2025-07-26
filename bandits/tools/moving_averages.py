import numpy as np
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt

from tools.coefficients import Coefficient, ConstantCoefficient, EarlyWeightedCoefficient


class MovingAverage(ABC):
    '''
        Family of objects which estimate the expected value of a sequence.

         Generalized form of update:
         val(t + 1) = val(t) + coefficient * f(target, ...)
    '''

    def __init__(self, initial_value: float):
        self._moving_average = initial_value
        self._call_count = 0

    @property
    def current_value(self) -> float:
        return self._moving_average

    def step(self, target: float = None) -> float:
        """
            Return the state value of the object.
        :param target: sequence value. If target == None, moving average is not updated
        :return: moving average value.
        """
        self._call_count += 1
        if target is not None:
            self._update(target=target)
        return self.current_value

    @abstractmethod
    def _update(self, target: float):
        raise NotImplementedError


class SimpleMovingAverage(MovingAverage):
    def __init__(self, initial_value: float, coefficient: Coefficient = ConstantCoefficient(0.1)):
        super(SimpleMovingAverage, self).__init__(initial_value=initial_value)
        assert 0 < coefficient.current_value <= 1.0, 'Invalid coefficient value. Must be 0 < coefficient <= 1'
        self.coefficient = coefficient

    def _update(self, target: float):
        c = self.coefficient.step()  # Update the coefficient
        assert c >= 0.0, 'Coefficient cannot be negative'
        self._moving_average += c * target


class CummulativeMovingAverage(MovingAverage):
    """
        Sample moving average:
        Y_mean(N+1) = (1/(N + 1)) * Sum_N(Yi) = Y(N) + (1/(N + 1)) * (target - Y(N))

        Ref: https://en.wikipedia.org/wiki/Moving_average
    """

    def __init__(self, initial_value: float):
        super(CummulativeMovingAverage, self).__init__(initial_value=initial_value)

    def _update(self, target: float):
        self._moving_average += (1. / (self._call_count + 1)) * (target - self._moving_average)


class ExponentialMovingAverage(MovingAverage):
    """

        Y_mean(N+1) = Y(N) + c(N) * (target - Y(N)).
                    = (1 - c(N)) * Y(N) + c(N) * target

        Ref: https://en.wikipedia.org/wiki/Moving_average
    """

    def __init__(self, initial_value, coefficient: Coefficient = ConstantCoefficient(0.1)):
        super(ExponentialMovingAverage, self).__init__(initial_value=initial_value)
        assert 0 < coefficient.current_value <= 1.0, 'Invalid coefficient value. Must be 0 < coefficient <= 1'
        self.coefficient = coefficient

    def _update(self, target: float):
        c = self.coefficient.step()  # Update the coefficient
        assert c >= 0.0, 'Coefficient cannot be negative'
        self._moving_average += c * (target - self._moving_average)


def exponential_moving_average_experiment():
    n_steps = 100
    constant_coefficient = ConstantCoefficient(0.05)
    samples = [1.] * n_steps
    ma = ExponentialMovingAverage(initial_value=0.0, coefficient=constant_coefficient)
    d_biased = [ma.step(target=s) for s in samples]
    uc = EarlyWeightedCoefficient(base_value=0.05)
    samples = [1.] * n_steps
    ma = ExponentialMovingAverage(initial_value=0.0, coefficient=uc)
    d_unbiased = [ma.step(target=s) for s in samples]
    _ = plt.figure()
    plt.plot(samples)
    plt.plot(d_biased, linestyle='--')
    plt.plot(d_unbiased, linestyle='-.')
    plt.legend(['Target', 'Biased Est', 'Unbiased Est'])
    plt.title('ExponentialMovingAverage')
    plt.grid()
    plt.show()


def moving_avg_experiment():
    n_steps = 2000
    n_trials = 500
    mean = 1.0
    constant_coefficient = ConstantCoefficient(0.1)

    ######## Test the quality of the final value of estimation #########
    data = list()
    for trial in range(n_trials):
        rand_samples = mean + np.random.normal(0.0, 0.1, size=n_steps)
        ma = CummulativeMovingAverage(-10.)
        d = [ma.step(target=s) for s in rand_samples]
        data.extend(d[-n_steps // 3:])
    print(f'Weighted average E[estimation_error] at step {n_steps}: ', mean - np.mean(data))

    rand_samples = mean + np.random.normal(0.0, 0.1, size=n_steps)
    ma = CummulativeMovingAverage(-0.5)
    d = [ma.step(target=s) for s in rand_samples]
    _ = plt.figure()
    plt.plot(rand_samples, linestyle=':')
    plt.plot(d, linestyle='-.')
    plt.legend(['Random', 'Moving average'])
    plt.title('Sample Moving Average')
    plt.grid()
    plt.show()

    data = list()
    for trial in range(n_trials):
        rand_samples = mean + np.random.normal(0.0, 0.1, size=n_steps)
        ma = ExponentialMovingAverage(initial_value=100., coefficient=constant_coefficient)
        d = [ma.step(target=s) for s in rand_samples]
        data.append(d[-n_steps // 3:])
    print(f'Exponential moving average E[estimation_error] at step {n_steps}: ', mean - np.mean(data))


if __name__ == "__main__":

    ########## Testing the bias ############
    '''
        Compare the performance between 
        * constant coefficient:
            ** biased
            ** val(t + 1) = val(t) + c * (target - val(t))
        * early weighted coefficient
            ** unbiased
            ** m(t+1) = m(t) + k*(1 - m(t)) : early weighted coefficient
            ** val(t + 1) = val(t) + m(t) * (target - val(t))
    '''

    exponential_moving_average_experiment()

    exit(0)
