import math
from abc import ABC, abstractmethod
from typing import Union, List
import numpy as np
from functools import partial


def to_np(v):
    if isinstance(v, np.ndarray):
        return v
    elif isinstance(v, (int, float)):
        return np.array([v])
    else:
        return np.array(v)


FloatListNPArray = Union[float, List[float], np.ndarray]


class RandomWalk(ABC):
    '''
        Y(n+1) = scale*Y(n) + noisy_step
        Def (2.1): https://www.math.ucla.edu/~biskup/PDFs/PCMI/PCMI-notes-1

        Assumption: E[Y(n+1)] = E[scale*Y(n) + noisy_step] = scale*Y(n) + E[noise]
                              = scale*Y(n) + 0

        * Subclasses can override E[noise]
        * Subclasses are to implement their noise sigma

        * Parameters are shaped (N,)
    '''
    def __init__(self, scale: FloatListNPArray = 1.0, initial_value: FloatListNPArray = 0.0,
                 clip_min=-np.inf, clip_max=np.inf):
        self.scale: np.ndarray = to_np(scale)
        self.value: np.ndarray = to_np(initial_value)
        assert len(self.scale.shape) == 1
        assert self.scale.shape == self.value.shape, 'Random walk parameters do not match'
        self.sample_size = self.scale.shape[0]
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, noise: FloatListNPArray) -> np.ndarray:
        self.value = np.clip(self.scale * self.value + to_np(noise), a_min=self.clip_min, a_max=self.clip_max)
        return self.value

    @property
    def current_mean(self) -> np.ndarray:
        return self.scale * self.value # Assuming E[noise] = 0

    @property
    @abstractmethod
    def sigma(self) -> np.ndarray:
        raise NotImplementedError


class NormalRandomWalk(RandomWalk):
    """ Random walk with normally distributed step """
    def __init__(self,
                 scale: FloatListNPArray = 1.0,
                 initial_value: FloatListNPArray = 0.0,
                 mean: FloatListNPArray = 0.0,
                 sig: FloatListNPArray = 1.0,
                 clip_min=-np.inf,
                 clip_max=np.inf):
        super(NormalRandomWalk, self).__init__(
            scale=scale,
            initial_value=initial_value,
            clip_min=clip_min,
            clip_max=clip_max)

        self._noise_mean = to_np(mean)
        self._noise_sigma = to_np(sig)
        self._rand_step = partial(np.random.normal, loc=mean, scale=sig)
        assert len(self._noise_mean.shape) == 1
        assert self._noise_mean.shape == self._noise_sigma.shape, 'Noise parameters do not match'

    def __call__(self, size: int = 1) -> np.ndarray:
        assert size == self.sample_size, 'Sample size mismatch with experiment'
        sample = super(NormalRandomWalk, self).__call__(self._rand_step(size=(size,)))
        return sample

    @property
    def current_mean(self) -> np.ndarray:
        return super(NormalRandomWalk, self).current_mean + self._noise_mean

    @property
    def sigma(self) -> np.ndarray:
        return self._noise_sigma


class BernoulliRandomWalk(RandomWalk):
    """ Random walk with Bernoulli distributed step """
    def __init__(self,
                 scale: FloatListNPArray = 1.0,
                 initial_value: FloatListNPArray = 0.0,
                 step_size: FloatListNPArray = 1.,
                 p: FloatListNPArray = 0.5,
                 clip_min=-np.inf,
                 clip_max=np.inf):
        super(BernoulliRandomWalk, self).__init__(
            scale=scale,
            initial_value=initial_value,
            clip_max=clip_max,
            clip_min=clip_min)

        self.p = to_np(p)
        self.step_size = to_np(step_size)
        assert len(self.p.shape) == 1
        assert self.p.shape == self.step_size.shape, 'Noise size and step size shapes must match'

    def _rand_step(self, size: int) -> np.ndarray:
        mask = 2 * (np.random.uniform(size=(size,)) >= self.p).astype(float) - 1 # {-1, 1}
        return self.step_size * mask

    def __call__(self, size: int = 1) -> np.ndarray:
        assert size == self.sample_size, 'Sample size mismatch with experiment'
        sample = super(BernoulliRandomWalk, self).__call__(noise=self._rand_step(size=size))
        return sample

    @property
    def current_mean(self) -> np.ndarray:
        return super(BernoulliRandomWalk, self).current_mean + self.p

    @property
    def sigma(self) -> np.ndarray:
        """https://en.wikipedia.org/wiki/Bernoulli_distribution"""
        return self.p*(1. - self.p)


if __name__ == "__main__":
    from tqdm import tqdm
    from matplotlib import pyplot as plt

    initial_value = 7.
    n_trials = 200
    steps = int(1e3)
    step_size_brw = 1.
    seeds = list(range(n_trials))
    collector = []

    nrw_constructor = lambda: NormalRandomWalk(scale=1.0, initial_value=initial_value, mean=0.0, sig=1.0)
    brnl_constructor = lambda: BernoulliRandomWalk(scale=1.0, initial_value=initial_value, p=0.5, step_size=step_size_brw)

    for t in tqdm(range(n_trials)):
        np.random.seed(seeds[t])
        rw = brnl_constructor()
        vals = np.array([rw() for _ in range(steps)]).squeeze()
        collector.append(vals)

    plt.plot(np.mean(collector, axis=0)) # Mean across time steps
    plt.title('{..., E[A(i|t)], E[A(i|t+1)], ...}')
    plt.grid()
    plt.show()

    _ = plt.figure()
    for i in range(3):
        plt.plot(collector[i])
    plt.grid()
    plt.title('Sample trajectories')
    plt.show()


    ##### parallel random walks

    n_walks = n_trials
    rwn_constructor = lambda : NormalRandomWalk(
        scale=[1.0]*n_walks,
        initial_value=[initial_value]*n_walks,
        mean=[0.0]*n_walks,
        sig=[1.0]*n_walks)

    brnl_constructor = lambda: BernoulliRandomWalk(
        scale=[1.0]*n_walks,
        initial_value=[initial_value]*n_walks,
        p=[0.5]*n_walks,
        step_size=[step_size_brw]*n_walks)

    rw = brnl_constructor()
    vals = np.array([rw(size=n_walks) for _ in range(steps)])
    plt.plot(np.mean(vals, axis=1)) # Mean across time steps
    plt.title('{..., E[A(i|t)], E[A(i|t+1)], ...}')
    plt.grid()
    plt.show()

    _ = plt.figure()
    for i in range(3):
        plt.plot(vals[..., i])
    plt.grid()
    plt.title('Sample trajectories')
    plt.show()

    ### Clipped parallel trajectories

    n_walks = n_trials
    clip_min, clip_max = -10, 15

    rwn_constructor = lambda: NormalRandomWalk(
        scale=[1.0] * n_walks,
        initial_value=[initial_value] * n_walks,
        mean=[0.0] * n_walks,
        sig=[1.0] * n_walks,
        clip_min=clip_min, clip_max=clip_max)

    brnl_constructor = lambda: BernoulliRandomWalk(
        scale=[1.0] * n_walks,
        initial_value=[initial_value] * n_walks,
        p=[0.5] * n_walks,
        step_size=[step_size_brw] * n_walks,
        clip_min=clip_min, clip_max=clip_max)

    rw = brnl_constructor()
    vals = np.array([rw(size=n_walks) for _ in range(steps)])
    plt.plot(np.mean(vals, axis=1))  # Mean across time steps
    plt.title('Clipped\n{..., E[A(i|t)], E[A(i|t+1)], ...}')
    plt.grid()
    plt.show()

    _ = plt.figure()
    for i in range(3):
        plt.plot(vals[..., i])
    plt.grid()
    plt.title('Clipped Sample trajectories')
    plt.show()

    exit(0)