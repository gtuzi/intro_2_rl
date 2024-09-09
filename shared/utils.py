from typing import Dict, Union, Any, List
import scipy


class NoiseSchedule:
    def initialize(self):
        """ Initialization of exploration to starting point """
        pass

    def reset(self):
        """
            State reset to starting point (eg, mu). In random walks
            this resets the mu to initial value.
            Exploration doesn't change  (e.g. sigma/eps)
        """
        pass

    def step(self, *args, **kwargs):
        pass

    @property
    def value(self):
        raise NotImplementedError


class ConstantSchedule(NoiseSchedule):
    def __init__(self, val):
        self.val = val

    @property
    def value(self):
        return self.val


class LinearEpsSchedule(NoiseSchedule):
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1

        self.inc = (end - start) / float(steps)

        self.start = start
        self.current = start
        self.end = end

        if end > start:
            self.bound = min
        else:
            self.bound = max

    def initialize(self):
        self.current = self.start

    def step(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)

    def reset(self):
        pass

    @property
    def value(self):
        return self.current


class PDFSampler:
    def __init__(
            self,
            distribution: Union[
                scipy.stats.rv_continuous, scipy.stats.rv_discrete
            ],
            params: Dict
    ):
        """
            Initialize the PDFSampler with a given distribution
            from scipy.stats and its parameters.
        """
        self.distribution = distribution
        self.params = params
        self.is_discrete = hasattr(distribution, 'pmf')

    def sample(self, size: int = 1) -> Union[Any, List[Any]]:
        """
        Generate a sample from the distribution.
        """
        addl = {}
        if 'size' not in self.params:
            addl.update({'size': size})
        sample = self.distribution.rvs(**{**self.params, **addl})

        if size == 1:
            return sample.tolist()[0]
        else:
            return sample.tolist()

    def probability(self, x: Union[Any, List[Any]]) -> Union[Any, List[Any]]:
        """
            Return the probability (or probability density for
            continuous distributions) of the given sample.
        """
        if self.is_discrete:
            p = self.distribution.pmf(x, **self.params)
        else:
            p = self.distribution.pdf(x, **self.params)

        if isinstance(x, List):
            return p.tolist()
        return p


class Experience:
    def __init__(self, **kwargs):
        self.s = None
        self.a = None
        self.p = None
        self.r = None
        self.sp = None
        self.ap = None
        self.pp = None
        self.done = None
        self.t = None
        self.sigma = None
        self.sigmap = None
        self.rho = None
        self.rhop = None

        for k, v in kwargs.items():
            if k == 's':  # state 0
                self.s = v
            elif k == 'a':  # action
                self.a = v
            elif k == 'p':  # action probability
                self.p = v
            elif k == 'r':  # reward
                self.r = v
            elif k == 'sp':  # next state (s')
                self.sp = v
            elif k == 'ap':  # next action ~ pi(s')
                self.ap = v
            elif k == 'pp':  # next action ~ pi(s')'s probability
                self.pp = v
            elif k == 'done':
                self.done = v
            elif k == 't':
                self.t = v
            elif k == 'sigma':
                self.sigma = v
            elif k == 'sigmap':
                self.sigmap = v
            elif k == 'rho':
                self.rho = v
            elif k == 'rhop':
                self.rhop = v