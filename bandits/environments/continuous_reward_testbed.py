from abc import ABC, abstractmethod
import numpy as np
from typing import List
from functools import partial

from bandits.environments.testbed import NonAssocativeTestBed
from bandits.tools.random_walks import NormalRandomWalk, BernoulliRandomWalk, RandomWalk


class ContinuousRewardBandit(ABC):
    @property
    @abstractmethod
    def reward_mean(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def reward_sigma(self) -> float:
        raise NotImplementedError


class StationaryRewardBandit(ContinuousRewardBandit):
    """
        Otherwise known as the "stochastic bandit". Reward distribution is stationary
    """
    def __init__(self, mean: float = 0.0, reward_sigma: float = 1.0, dist: str = 'normal'):
        assert dist.lower() == 'normal', f'{dist} not implemented'

        self._reward_mean = mean
        self._reward_sigma = reward_sigma
        # This is a normal distribution bandit
        self.rand = partial(np.random.normal, loc=mean, scale=reward_sigma)

    def __call__(self, n_samples: int = 1):
        """ Step in time """
        assert n_samples > 0
        return self.rand(size=n_samples).tolist()

    @property
    def reward_mean(self) -> float:
        return self._reward_mean

    @property
    def reward_sigma(self) -> float:
        return self._reward_sigma


class NonStationaryRewardBandit(ContinuousRewardBandit):
    def __init__(self, initial_reward: float = 0.0, randomness_scale: float = 0.01, dist: str = 'normal'):
        if dist.lower() == 'normal':
            self.rw: RandomWalk = NormalRandomWalk(initial_value=initial_reward, sig=randomness_scale)
        elif dist.lower() == 'bernoulli':
            self.rw: RandomWalk = BernoulliRandomWalk(initial_value=initial_reward, p=randomness_scale)
        else:
            raise Exception('Distribution not recognized')

    def __call__(self, n_samples: int = 1) -> List:
        assert n_samples > 0
        return self.rw(size=n_samples).tolist()

    @property
    def reward_mean(self) -> float:
        return float(self.rw.current_mean)

    @property
    def reward_sigma(self) -> float:
        return float(self.rw.sigma)


class ContinuousValueRewardTestBed(NonAssocativeTestBed):
    def __init__(self, reward_means: List, reward_randomness_scales: List, dist='normal', stationary: bool = True):
        assert len(reward_means) == len(reward_randomness_scales)

        if stationary:
            self.bandits = [
                StationaryRewardBandit(
                    mean=r_mean,
                    reward_sigma=r_sig,
                    dist=dist)
                for r_mean, r_sig in zip(reward_means, reward_randomness_scales)
            ]
        else:
            self.bandits = [
                NonStationaryRewardBandit(
                    initial_reward=r_mean,
                    randomness_scale=r_scale,
                    dist=dist)
                for r_mean, r_scale in zip(reward_means, reward_randomness_scales)
            ]

    def __call__(self, action: int) -> float:
        return self.bandits[action](n_samples=1)[0]

    @property
    def best_mean(self) -> float:
        return max([b.reward_mean for b in self.bandits])

    @property
    def best_arm(self) -> float:
        reward_means = [b.reward_mean for b in self.bandits]
        return reward_means.index(max(reward_means))

    def arm_mean(self, a) -> float:
        return self.bandits[a].reward_mean


if __name__ == "__main__":
    # TODO: test cases
    pass
