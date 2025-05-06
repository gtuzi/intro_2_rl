from typing import Union, Callable, Any
import numpy as np

from approximate_methods.utils import (
    LinearQEpsGreedyAgent,
    NoiseSchedule,
    Experience)
from shared.utils import LinearSchedule


from torch.utils.tensorboard import SummaryWriter


class LambdaReturn:
    def __init__(self):
        raise NotImplemented
