from typing import Dict, Union, Any, List
import numpy as np
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


class LinearSchedule(NoiseSchedule):
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


class CosineDecaySchedule(NoiseSchedule):
    def __init__(self, initial_value, final_value, decay_steps):
        """
        Args:
            initial_value (float): The starting value of the noise.
            final_value (float): The value to which the noise decays.
            decay_steps (int): The number of steps over which the decay occurs before reset.
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_steps = decay_steps
        self.current_step = 0
        self._value = self.initial_value  # Start at the initial value

    def initialize(self):
        """ Initialization of exploration to the starting point """
        self.current_step = 0
        self._value = self.initial_value

    def reset(self):
        """
        Reset the schedule to the initial state (used to reset between cycles).
        """
        self.current_step = 0
        self._value = self.initial_value

    def step(self):
        """
        Step the schedule forward by one. This will update the current value based on a
        cosine decay function with periodic resets after 'decay_steps' steps.
        """
        # Reset the step counter periodically
        cycle_step = self.current_step % self.decay_steps

        # Apply the cosine decay function
        cosine_decay = 0.5 * (1 + np.cos(np.pi * cycle_step / self.decay_steps))
        decayed_value = self.final_value + (self.initial_value - self.final_value) * cosine_decay

        # Update the current value and step
        self._value = decayed_value
        self.current_step += 1

    @property
    def value(self):
        """ Returns the current value of the schedule. """
        return self._value


class CosineDecayWithHoldSchedule(NoiseSchedule):
    def __init__(self, initial_value, final_value, decay_steps,
                 initial_hold_steps, final_hold_cycles):
        """
        Args:
            initial_value (float): The starting value of the noise.
            final_value (float): The value to which the noise decays.
            decay_steps (int): The number of steps over which the decay occurs before resetting.
            initial_hold_steps (int): The number of steps to hold the initial value before decay starts.
            final_hold_cycles (int): The number of cycles (complete decay periods) before holding the final value indefinitely.
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_steps = decay_steps
        self.initial_hold_steps = initial_hold_steps  # Number of steps to hold the initial value
        self.final_hold_cycles = final_hold_cycles  # Number of cycles before final hold

        self.current_step = 0
        self._value = self.initial_value  # Start with initial value
        self.hold_final = False

    def initialize(self):
        """ Initialization of exploration to starting point """
        self.current_step = 0
        self._value = self.initial_value
        self.hold_final = False

    def reset(self):
        """ Reset the schedule to the initial state (used to reset between cycles). """
        self.current_step = 0
        self._value = self.initial_value
        self.hold_final = False

    def step(self):
        """
        Step the schedule forward by one. This updates the current value based on a
        cosine decay function with periodic resets after 'decay_steps' steps, and includes
        an initial hold at the beginning and a final hold after cycling ends.
        """
        if self.current_step < self.initial_hold_steps:
            # Initial hold phase
            self._value = self.initial_value

        elif not self.hold_final:
            # Cycle step: cosine decay with reset after decay_steps
            cycle_step = (self.current_step - self.initial_hold_steps) % self.decay_steps
            cycle_num = (self.current_step - self.initial_hold_steps) // self.decay_steps

            if cycle_num >= self.final_hold_cycles:
                # Final hold phase after all cycles
                self._value = self.final_value
                self.hold_final = True
            else:
                # Apply cosine decay during cycling
                cosine_decay = 0.5 * (
                            1 + np.cos(np.pi * cycle_step / self.decay_steps))
                self._value = self.final_value + (
                            self.initial_value - self.final_value) * cosine_decay

        # Increment the step
        self.current_step += 1

    @property
    def value(self):
        """ Returns the current value of the schedule. """
        return self._value


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