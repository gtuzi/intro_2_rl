from typing import Tuple, Union, Callable, Any, List
import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F

from shared.policies import SoftPolicy
from shared.utils import NoiseSchedule, Experience

class DiscreteActionAgent:
    def __init__(
            self,
            feature_size: int,
            action_space_dims: int
    ):
        self.feature_size = feature_size
        self.action_space_dims = action_space_dims

    def act(self, s) -> Tuple[int, float]:
        """ Return the action and probability """
        raise NotImplementedError

    def initialize(self):
        pass

    def reset(self):
        pass

    def step(self, *args, **kwargs):
        """ Learn: Qk+1 = somefunction(Qk) """
        pass


class LinearQEpsGreedyAgent(DiscreteActionAgent, SoftPolicy):

    def __init__(
            self,
            feature_size: int,
            action_space_dims: int,
            discount: float,
            feature_fn: Callable[[Any, int], np.ndarray], # state, action(int) --> np.ndarray
            eps: Union[float, NoiseSchedule] = 0.01
    ):
        DiscreteActionAgent.__init__(self, feature_size, action_space_dims)
        SoftPolicy.__init__(self)
        self.discount = discount
        self.eps = eps
        self.w = None
        self.feature_fn = feature_fn
        self.init_weights()

    def init_weights(self, *args, **kwargs):
        if 'init' in kwargs:
            self.w = kwargs['init']((self.feature_size, 1))
        else:
            self.w = np.zeros((self.feature_size, 1), dtype=np.float32) #[features|actions]

    def action_values(self, s) -> np.ndarray:
        """ Q[s, .., a[i], ... | w] """

        # Quick shape check
        r = self.feature_fn(s, 0)
        assert 0 < len(r.shape) <= 2
        assert r.shape[0] == self.w.shape[0]
        assert r.shape[1] == 1 if len(r.shape) == 2 else None

        if len(r.shape) == 2:
            res =  np.array([
                np.dot(self.w.T, self.feature_fn(s, a)).squeeze()
                for a in range(self.action_space_dims)
            ])
        else:
            res = np.array([
                np.dot(self.w.T, self.feature_fn(s, a)[..., None]).squeeze()
                for a in range(self.action_space_dims)
            ])

        return res

    def state_action_value(self, s: Any, a: int) -> float:
        """ Q[s, a | w] """

        x = self.feature_fn(s, a)
        assert x.shape == self.w.shape

        return np.dot(self.w.T, x).squeeze()

    def get_greedy_action(self, s) -> Tuple[int, float]:
        """
            Get the greedy action and its **conditional** prob.
            If a single action: conditional probability is 1.
            If multiple actions compete for being picked, they are randomly
            tie-broken. This mean's that their probability is 1/|argmax_a|
        """
        av = self.action_values(s)
        max_vals = np.amax(av)
        idc = np.argwhere(av == max_vals).squeeze().tolist()

        if isinstance(idc, list):
            # Random tie-breaking
            return int(np.random.choice(idc)), 1. / len(idc)
        else:
            assert isinstance(idc, int)
            return idc, 1.

    def get_sa_probability(self, s, a) -> float:
        if isinstance(self.eps, NoiseSchedule):
            eps = self.eps.value
        else:
            eps = self.eps

        # Check if action is greedy for this state
        av = self.action_values(s)
        max_vals = np.amax(av)
        idc = np.argwhere(av == max_vals).squeeze().tolist()

        if isinstance(idc, list) and (a in idc):
            return ((1. - eps) / len(idc)) + eps / self.action_space_dims
        elif isinstance(idc, int) and (a == idc):
            assert isinstance(idc, int)
            return 1. - eps + eps / self.action_space_dims
        # Action is not greedy.
        else:
            return eps / self.action_space_dims

    def act(self, s) -> Tuple[int, float]:
        """
            eps-greedy policy
        :param s: state
        :return: (action, probability of action)
        """
        if isinstance(self.eps, NoiseSchedule):
            eps = self.eps.value
        else:
            eps = self.eps

        # We need to know which one is the greedy action and/or it's prob
        a_greedy, pcond_greedy = self.get_greedy_action(s)
        p_greedy = 1. - eps + eps / self.action_space_dims  # If 1 argmax_action
        # If there are > 1 argmax_a's this is scaled according to its
        # conditional uniform pmf
        p_greedy *= pcond_greedy

        greedy = np.random.choice([True, False], p=[1. - eps, eps])

        if greedy:
            return a_greedy, p_greedy
        else:
            a = int(np.random.choice(self.action_space_dims))
            # The greedy action can still be picked
            if a == a_greedy:
                return a, p_greedy
            else:
                if pcond_greedy > (1. - 0.0001):
                    # We're guaranteed that the greedy action was not tie-broken
                    # so the non-greedy has eps/num_actions probability
                    return a, eps / self.action_space_dims

                # The non-greedy action here, may have been one of the
                # randomly tie-broken greedy actions. This means that the
                # probability of this action may not exactly
                # eps / action_space_dims but rather p_greedy, where we've
                # already scaled it with pcond_greedy

                av = self.action_values(s)
                max_vals = np.amax(av)
                idc = np.argwhere(av == max_vals).squeeze().tolist()
                if a in idc:
                    return a, p_greedy
                else:
                    return a, eps / self.action_space_dims

    def state_value(self, s):
        """ V[s] """
        probs = [
            self.get_sa_probability(s, a)
            for a in range(self.action_space_dims)
        ]

        av = self.action_values(s)

        return sum([p * q for p, q in zip(probs, av)])

    def optimal_state_value(self, s):
        a, _ = self.get_greedy_action(s)
        av = self.action_values(s)
        return av[a]


from approximate_methods.tiles3 import IHT, tiles


# ------ Feature Extractors ------

class TileCodingFeature:
    def __init__(
            self,
            max_size: int,
            num_tiles: int,
            num_tilings: int,
            x0_low: float,
            x1_low: float,
            x0_high: float,
            x1_high: float):

        self.iht = IHT(max_size)
        self.x0_low = x0_low
        self.x1_low = x1_low
        self.x0_high = x0_high
        self.x1_high  = x1_high
        self.num_tiles = num_tiles
        self.num_tilings = num_tilings

    def __call__(
            self,
            x: Union[List, np.ndarray],
            a: int, **kwargs
    ) -> np.ndarray:

        x0, x1 = x[0], x[1]
        feats = tiles(
            ihtORsize=self.iht,
            numtilings=self.num_tilings,
            floats=[
                self.num_tiles * x0 / (self.x0_high - self.x0_low),
                self.num_tiles * x1 / (self.x1_high - self.x1_low)
            ],
            ints=[a]
        )

        feats = np.array(feats)

        res = np.zeros((self.iht.size, 1), dtype=np.float32)
        res[feats] = 1.
        return res


class TileCodingNFeature:
    def __init__(
            self,
            max_size: int,
            num_tiles: int,
            num_tilings: int,
            lows: List[float],
            highs: List[float]):

        assert isinstance(lows, (list, tuple))
        assert isinstance(highs, (list, tuple))
        assert len(lows) == len(highs)

        self.iht = IHT(max_size)
        self.lows = lows
        self.highs = highs
        self.num_tiles = num_tiles
        self.num_tilings = num_tilings

    def __call__(
            self,
            x: Union[List, np.ndarray],
            a: int, **kwargs
    ) -> np.ndarray:

        assert len(x) == len(self.lows) == len(self.highs)

        floats = [
            self.num_tiles * xx / (h - l)
            for xx, l, h in zip(x, self.lows, self.highs)
        ]

        feats = tiles(
            ihtORsize=self.iht,
            numtilings=self.num_tilings,
            floats=floats,
            ints=[a]
        )

        feats = np.array(feats)

        res = np.zeros((self.iht.size, 1), dtype=np.float32)
        res[feats] = 1.
        return res


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

