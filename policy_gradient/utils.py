from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional, List
import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from shared.utils import (LinearSchedule, SoftPolicy)


class DiscreteActionAgent:
    def __init__(
            self,
            feature_size: int,
            action_space_dims: int
    ):
        self.feature_size = feature_size
        self.action_space_dims = action_space_dims

    def act(self, s, **kwargs) -> Tuple[int, float]:
        """ Return discrete action and its probability """
        raise NotImplementedError

    def initialize(self, **kwargs):
        pass

    def reset(self, **kwargs):
        pass

    def step(self, *args, **kwargs):
        """ Learn """
        pass


class DiscreteActionSoftPolicy(DiscreteActionAgent, SoftPolicy):
    def __init__(
            self,
            state_size: int,
            action_space_dims: int,
            discount: float,
            temp: Union[float, LinearSchedule] = 1.,
            seed: Optional[int] = None
    ):
        super().__init__(
            feature_size=state_size,
            action_space_dims=action_space_dims)

        self.discount = discount
        self.temp = temp
        self.rng = np.random.default_rng(seed)
        self._writer: Optional[SummaryWriter] = None
        torch.manual_seed(seed)

    @property
    def temperature(self) -> float:
        t = self.temp
        if isinstance(t, LinearSchedule):
            t = t.value
        return t

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    @writer.setter
    def writer(self, w: SummaryWriter):
        if w is not None:
            assert isinstance(w, SummaryWriter)
        self._writer = w

    @abstractmethod
    def logp(self, s, **kwargs) -> float:
        raise NotImplementedError

    @abstractmethod
    def prob(self, s, **kwargs) -> float:
        raise NotImplementedError


class DiscreteActionCriticStateValue(ABC):
    @abstractmethod
    def state_value(self, s, **kwargs) -> float:
        # V(s)
        raise NotImplementedError


class DiscreteActionCriticActionValue(DiscreteActionCriticStateValue):

    @abstractmethod
    def get_state_action_value(self, s, a, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def optimal_state_value(self, s, **kwargs) -> float:
        # Value of greedy action: max_a(Q[s][a])
        raise NotImplementedError


class ContinuousActionAgent:
    def __init__(
            self,
            feature_size: int,
            action_size: int,
            action_mins: Optional[
                Union[Tuple[float, ...], List[float]]] = None,
            action_maxs: Optional[
                Union[Tuple[float, ...], List[float]]] = None,
    ):
        self.feature_size = feature_size
        self.action_size = action_size
        self.action_mins = action_mins
        self.action_maxs = action_maxs

    def act(self, s) -> Tuple[int, float]:
        """ Return the action and probability """
        raise NotImplementedError

    def initialize(self):
        pass

    def reset(self):
        pass

    def step(self, *args, **kwargs):
        pass


class ContinuousActionSoftPolicy(ContinuousActionAgent, SoftPolicy):
    def __init__(
            self,
            state_size: int,
            action_size: int,
            discount: float,
            action_mins: Optional[
                Union[Tuple[float, ...], List[float]]] = None,
            action_maxs: Optional[
                Union[Tuple[float, ...], List[float]]] = None,
            seed: Optional[int] = None
    ):
        super().__init__(
            feature_size=state_size,
            action_size=action_size,
            action_mins=action_mins,
            action_maxs=action_maxs
        )

        self.discount = discount
        self.rng = np.random.default_rng(seed)
        self._writer: Optional[SummaryWriter] = None
        torch.manual_seed(seed)
        self._writer: Optional[SummaryWriter] = None

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    @writer.setter
    def writer(self, w: SummaryWriter):
        if w is not None:
            assert isinstance(w, SummaryWriter)
        self._writer = w

    @abstractmethod
    def pd(self, s):
        raise NotImplementedError

    @abstractmethod
    def logp_sa(self, s, a, **kwargs) -> float:
        raise NotImplementedError


class ContinuousActionCriticStateValue(ABC):
    @abstractmethod
    def state_value(self, s, **kwargs) -> float:
        # V(s)
        raise NotImplementedError


##############################################################
# Utilities
##############################################################

# --- Feature Extractors for vision tasks
def get_mobilenet_feature_extractor(name: str, normalize: bool = True):
    net = None
    preprocess = None

    if name == 'mobilenet-v3':
        # Load pre-trained EfficientNet-B0 and remove the classifier head
        net = models.mobilenet_v3_small(pretrained=True)

        # Remove the final classification layer
        net = nn.Sequential(*list(net.children())[:-1])

        # Preprocess input images to match MobileNetV2 input requirements
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f'Unknown EfficientNet model: {name}')

    net.eval()

    # Get the output shape of the model:
    dummy_input = torch.randn(1, 3, 224, 224)

    # Pass the dummy input through the modified EfficientNet
    with torch.no_grad():
        output = net(dummy_input)

    assert len(output.squeeze().shape) == 1

    feature_size = output.squeeze().shape[0]

    def extract_features(state):
        # Gym provides images as (H, W, C), we need to reshape to (C, H, W)
        # state = np.transpose(state, (2, 0, 1))
        state_tensor = preprocess(state).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = net(state_tensor).squeeze()[..., None]

        if normalize:
            features = F.normalize(features, p=2, dim=0)

        return features.detach().numpy()

    return extract_features, feature_size


def get_efficientnet_feature_extractor(name: str, normalize: bool = True):
    net = None
    if name == 'efficientnet-b0':
        # Load pre-trained EfficientNet-B0 and remove the classifier head
        net = models.efficientnet_b0(pretrained=True)

        # Remove the final classification layer
        net = nn.Sequential(
            *list(net.children())[:-1]
        )
    else:
        raise ValueError(f'Unknown EfficientNet model: {name}')

    net.eval()

    # Get the output shape of the model:
    dummy_input = torch.randn(1, 3, 224, 224)

    # Pass the dummy input through the modified EfficientNet
    with torch.no_grad():
        output = net(dummy_input)

    assert len(output.squeeze().shape) == 1

    feature_size = output.squeeze().shape[0]

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])

    # Function to extract features using EfficientNet
    def extract_features(state):
        # Gym provides images as (H, W, C), we need to reshape to (C, H, W)
        # state = np.transpose(state, (2, 0, 1))
        state_tensor = preprocess(state).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = net(state_tensor).squeeze()[..., None]

        if normalize:
            features = F.normalize(features, p=2, dim=0)
            # features = F.softmax(features, dim=0)

        return features.detach().numpy()

    return extract_features, feature_size


def get_nn_based_feature_extractor(
        num_actions: int,
        nn_name: str = 'efficientnet-b0',
        normalize: bool = True
):
    if nn_name.startswith('efficientnet'):
        fe, nn_out_size = get_efficientnet_feature_extractor(nn_name, normalize)
    elif nn_name.startswith('mobilenet'):
        fe, nn_out_size = get_mobilenet_feature_extractor(nn_name, normalize)
    else:
        raise ValueError(f'Unknown model: {nn_name}')

    def action_feature_extractor(a: int):
        assert isinstance(a, int)
        oh = F.one_hot(
            torch.tensor(a), num_classes=num_actions
        ).squeeze()[..., None].float()

        return oh.detach().numpy()

    def feature_extractor(state: np.ndarray, action: int):
        state_features = fe(state)
        action_features = action_feature_extractor(action)
        return np.concatenate([state_features, action_features], axis=0)

    # Action is one-hot encoded
    return feature_extractor, nn_out_size + num_actions